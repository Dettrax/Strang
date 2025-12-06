from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- series helpers ----------------
@torch.no_grad()
def _factorials_up_to(K: int, device):
    fac = torch.zeros(K + 1, dtype=torch.float32, device=device)
    fac[0] = 1.0
    for k in range(1, K + 1):
        fac[k] = fac[k - 1] * k
    return fac


def heat_series_apply(
    L_sym: torch.Tensor,
    X: torch.Tensor,
    t: float,
    K: int,
    fac: torch.Tensor,
) -> torch.Tensor:
    """
    Approximate exp(-t L_sym) X via truncated power series:
        exp(-t L) X  ≈  Σ_{k=0}^K (-t)^k / k!  (L^k X)
    """
    assert L_sym.is_sparse
    Y = X
    term = X
    for k in range(1, K + 1):
        term = torch.sparse.mm(L_sym, term)
        Y = Y + ((-t) ** k) / float(fac[k].item()) * term
    return Y


def complex_rot_series_apply(
    S: torch.Tensor,
    Zr: torch.Tensor,
    Zi: torch.Tensor,
    t: float,
    K: int,
    fac: torch.Tensor,
) -> (torch.Tensor, torch.Tensor):
    """
    Approximate exp(-i t S) Z where Z = Zr + i Zi and S is real skew-symmetric.
    This corresponds to the unitary evolution under the imaginary part of the Laplacian.
    
    Series: exp(-i t S) = Σ_{k=0}^K (-i t)^k / k! S^k
    """
    assert S.is_sparse
    
    Yr = Zr.clone()
    Yi = Zi.clone()
    
    Tr = Zr
    Ti = Zi
    
    for k in range(1, K + 1):
        # T_k = S * T_{k-1}
        Tr = torch.sparse.mm(S, Tr)
        Ti = torch.sparse.mm(S, Ti)
        
        inv_fact = 1.0 / float(fac[k].item())
        tk = (t ** k) * inv_fact
        
        rem = k % 4
        if rem == 1:   # (-i)^1 = -i  =>  -i * (Tr + i Ti) = Ti - i Tr
            Yr = Yr + tk * Ti
            Yi = Yi - tk * Tr
        elif rem == 2: # (-i)^2 = -1  =>  -1 * (Tr + i Ti) = -Tr - i Ti
            Yr = Yr - tk * Tr
            Yi = Yi - tk * Ti
        elif rem == 3: # (-i)^3 = i   =>   i * (Tr + i Ti) = -Ti + i Tr
            Yr = Yr - tk * Ti
            Yi = Yi + tk * Tr
        elif rem == 0: # (-i)^4 = 1   =>   1 * (Tr + i Ti) = Tr + i Ti
            Yr = Yr + tk * Tr
            Yi = Yi + tk * Ti
            
    return Yr, Yi


# ---------------- loss / metric ----------------
def label_smoothing_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.0,
    class_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Cross-entropy with label smoothing and optional class weights.
    """
    C = logits.size(-1)
    logp = F.log_softmax(logits, dim=-1)

    with torch.no_grad():
        y_true = torch.zeros_like(logits).scatter_(1, y.view(-1, 1), 1.0)
        y_smooth = (1.0 - eps) * y_true + eps / C

    loss_i = -(y_smooth * logp).sum(dim=1)
    if class_weight is not None:
        w = class_weight[y]
        return (loss_i * w).sum() / (w.sum() + 1e-12)
    return loss_i.mean()


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == y).float().mean().item()


# ---------------- model ----------------
class StrangMagHeatNet(nn.Module):
    """
    Multi-scale magnetic heat diffusion with Strang splitting.

    For each time scale t:
      1) Z_t =  Heat(Lsym, t/2) -> Rot(Lskew, t) -> Heat(Lsym, t/2)  on Xp.
      2) Low-pass:  L_t = Z_t
         High-pass: H_t = Xp - Z_t
      3) Concatenate [L_t, H_t] and encode with a per-t MLP -> h_t.

    Aggregation:
      - Concatenate [h_t] over all t along feature dimension.
      - Classifier on the concatenated multi-scale representation.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        heat_scales: List[float],
        K_heat: int = 12,
        K_rot: int = 6,
        x_proj_dim: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.5,
        tau_t: float = 0.7,  # kept in signature for compatibility, not used
    ):
        super().__init__()
        self.heat_scales = [float(t) for t in heat_scales]
        self.K_heat = K_heat
        self.K_rot = K_rot

        # Base projection of node features
        self.x_proj = nn.Linear(in_dim, x_proj_dim)
        self.norm = nn.LayerNorm(x_proj_dim)
        self.dropout = nn.Dropout(p=dropout)

        # Per-time encoder: input is [Re(L), Im(L), Re(H), Im(H)] -> 4 * x_proj_dim
        per_t_in = 4 * x_proj_dim
        self.t_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(per_t_in, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                )
                for _ in self.heat_scales
            ]
        )

        # Concatenation-based aggregation
        self.classifier = nn.Linear(hidden_dim * len(self.heat_scales), num_classes)

    def _strang_step(
        self,
        Lsym: torch.Tensor,
        Lskew: torch.Tensor,
        Xr: torch.Tensor,
        Xi: torch.Tensor,
        t: float,
        fac_heat: torch.Tensor,
        fac_rot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        One Strang splitting step in complex domain:
            Heat(t/2) -> Rot(t) -> Heat(t/2)
        """
        # 1. Heat diffusion (real symmetric) - acts on Re and Im independently
        # exp(-t/2 L_sym)
        Zr = heat_series_apply(Lsym, Xr, t * 0.5, self.K_heat, fac_heat)
        Zi = heat_series_apply(Lsym, Xi, t * 0.5, self.K_heat, fac_heat)
        
        # 2. Rotation (imaginary skew-symmetric) - mixes Re and Im
        # exp(-i t L_skew)
        Zr, Zi = complex_rot_series_apply(Lskew, Zr, Zi, t, self.K_rot, fac_rot)
        
        # 3. Heat diffusion again
        Zr = heat_series_apply(Lsym, Zr, t * 0.5, self.K_heat, fac_heat)
        Zi = heat_series_apply(Lsym, Zi, t * 0.5, self.K_heat, fac_heat)
        
        return Zr, Zi

    def forward(
        self,
        X: torch.Tensor,
        Lsym: torch.Tensor,
        Lskew: torch.Tensor,
    ) -> torch.Tensor:
        """
        X:     [N, in_dim]
        Lsym:  sparse symmetric Laplacian
        Lskew: sparse skew-symmetric Laplacian

        Returns:
            logits: [N, num_classes]
        """
        device = X.device

        # 1) Project & normalize features
        Xp = self.x_proj(X)
        Xp = self.norm(Xp)
        Xp = F.relu(Xp, inplace=True)
        Xp = self.dropout(Xp)
        
        # Initialize complex features (Real=Xp, Imag=0)
        Xp_r = Xp
        Xp_i = torch.zeros_like(Xp)

        # 2) Precompute factorials for truncated series
        fac_heat = _factorials_up_to(self.K_heat, device=device)
        fac_rot = _factorials_up_to(self.K_rot, device=device)

        Lsym = Lsym.coalesce()
        Lskew = Lskew.coalesce()

        # 3) Multi-scale features for each t
        per_t_h = []
        for ti, t in enumerate(self.heat_scales):
            Zt_r, Zt_i = self._strang_step(Lsym, Lskew, Xp_r, Xp_i, float(t), fac_heat, fac_rot)

            # Concatenate Real and Imaginary parts of Low-pass and High-pass
            # Low-pass: Zt
            # High-pass: Xp - Zt
            
            low_r = Zt_r
            low_i = Zt_i
            high_r = Xp_r - Zt_r
            high_i = Xp_i - Zt_i
            
            # Concatenate all components: [Re(L), Im(L), Re(H), Im(H)]
            feat_t = torch.cat([low_r, low_i, high_r, high_i], dim=-1)  # [N, 4 * x_proj_dim]

            h_t = self.t_mlps[ti](feat_t)  # [N, hidden_dim]
            per_t_h.append(h_t)

        # 4) Concatenate across time scales
        H_concat = torch.cat(per_t_h, dim=-1)  # [N, T * hidden_dim]
        H_concat = self.dropout(H_concat)

        logits = self.classifier(H_concat)
        return logits
