import math
import numpy as np
import scipy.sparse as sp
import torch

def _edge_index_to_csr(edge_index: torch.Tensor, num_nodes: int) -> sp.csr_matrix:
    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()
    data = np.ones_like(row, dtype=np.float32)
    A = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    return A.tocsr()

def bidirectional_ratio(edge_index: torch.Tensor, num_nodes: int) -> float:
    A = _edge_index_to_csr(edge_index, num_nodes)
    pairs = A.nonzero()
    total = pairs[0].shape[0]
    rev = (A.T.multiply(A) > 0).sum()  # counts bidirectional twice; fix by //2
    return float((rev // 2) / max(1, total // 2))

def build_magnetic_operators(edge_index: torch.Tensor,
                             num_nodes: int,
                             q: float,
                             device: torch.device):
    """
    Build magnetic Laplacian L = I - D^{-1/2} H D^{-1/2}, split into real/imag.
    H_ij = a_s,ij * exp(i * theta_ij), with a_s from symmetric magnitude,
    and theta_ij = 2π q * (A_ij - A_ji) ∈ { -2πq, 0, +2πq }.
    We then return:
      L_sym_real = (Re(L) + Re(L)^T)/2   (Hermitian real part)  --> for heat diffusion
      L_skew_im  = (Im(L) - Im(L)^T)/2   (skew imag part)       --> directional features
    All returned as torch.sparse_coo on the given device.
    """
    A = _edge_index_to_csr(edge_index, num_nodes)
    As = (A + A.T) * 0.5  # symmetric magnitude
    deg = np.asarray(As.sum(axis=1)).reshape(-1)
    deg[deg == 0.0] = 1.0

    # quick lookup for directed presence
    A_bool = (A > 0).astype(np.float32)
    A_lookup = { (i, j): 1.0 for i, j in zip(*A_bool.nonzero()) }

    # assemble L (real, imag) in COO lists
    rows, cols, vals_real, vals_imag = [], [], [], []

    # diagonal = 1 + 0i
    for i in range(num_nodes):
        rows.append(i); cols.append(i)
        vals_real.append(1.0); vals_imag.append(0.0)

    As_coo = As.tocoo()
    for i, j, a_s in zip(As_coo.row, As_coo.col, As_coo.data):
        if i == j or a_s == 0.0:
            continue
        w_ij = A_lookup.get((i, j), 0.0)
        w_ji = A_lookup.get((j, i), 0.0)
        theta = 2.0 * math.pi * q * (w_ij - w_ji)
        c, s = math.cos(theta), math.sin(theta)

        h_real = a_s * c
        h_imag = a_s * s

        norm = math.sqrt(deg[i] * deg[j])  # normalized operator
        h_real /= norm
        h_imag /= norm

        # L_ij = - H_ij (off-diagonal)
        rows.append(i); cols.append(j)
        vals_real.append(-h_real)
        vals_imag.append(-h_imag)

    idx = torch.tensor([rows, cols], dtype=torch.long, device=device)
    Lr = torch.sparse_coo_tensor(idx, torch.tensor(vals_real, dtype=torch.float32, device=device),
                                 (num_nodes, num_nodes)).coalesce()
    Li = torch.sparse_coo_tensor(idx, torch.tensor(vals_imag, dtype=torch.float32, device=device),
                                 (num_nodes, num_nodes)).coalesce()

    # Hermitian real part for diffusion: (Lr + Lr^T)/2
    Lr_T = torch.sparse_coo_tensor(torch.stack([idx[1], idx[0]]),
                                   Lr.values(), (num_nodes, num_nodes)).coalesce()
    L_sym_real = _sparse_average(Lr, Lr_T)

    # Skew imaginary (directional) part: (Li - Li^T)/2
    Li_T = torch.sparse_coo_tensor(torch.stack([idx[1], idx[0]]),
                                   Li.values(), (num_nodes, num_nodes)).coalesce()
    L_skew_im = _sparse_halved_diff(Li, Li_T)

    return L_sym_real.coalesce(), L_skew_im.coalesce()

def _sparse_add(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return (A + B).coalesce()

def _sparse_average(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return (A * 0.5 + B * 0.5).coalesce()

def _sparse_halved_diff(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return (A * 0.5 - B * 0.5).coalesce()
