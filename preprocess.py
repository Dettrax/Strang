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
    undirected = ((A + A.T) > 0).astype(np.float32)
    both = (A.multiply(A.T) > 0).astype(np.float32)
    num_pairs = int(undirected.sum() - undirected.diagonal().sum()) // 2
    num_bidir = int(both.sum() - both.diagonal().sum()) // 2
    if num_pairs == 0:
        return 0.0
    return num_bidir / num_pairs

def build_magnetic_operators(edge_index: torch.Tensor,
                             num_nodes: int,
                             q: float,
                             device=None):
    """
    Magnetic Laplacian L = I - D^{-1/2} H D^{-1/2}
      H_ij = a_s,ij * exp(i * theta_ij)
      a_s from symmetric magnitude As = (A + A^T)/2
      theta_ij = 2π q * (A_ij - A_ji)  ∈ {-2πq, 0, +2πq}
    Return:
      L_sym_real = (Re(L) + Re(L)^T)/2   (symmetric real)   --> heat
      L_skew_im  = (Im(L) - Im(L)^T)/2   (skew imaginary)   --> rotation
    """
    if device is None:
        device = edge_index.device

    A = _edge_index_to_csr(edge_index, num_nodes)
    As = (A + A.T) * 0.5
    deg = np.asarray(As.sum(axis=1)).reshape(-1).astype(np.float32)
    deg[deg == 0.0] = 1.0

    A_bool = (A > 0).astype(np.float32)
    A_lookup = {(i, j): 1.0 for i, j in zip(*A_bool.nonzero())}

    rows, cols, vr, vi = [], [], [], []

    # diagonal = 1
    for i in range(num_nodes):
        rows.append(i); cols.append(i)
        vr.append(1.0); vi.append(0.0)

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

        norm = math.sqrt(deg[i] * deg[j])
        h_real /= norm
        h_imag /= norm

        rows.append(i); cols.append(j)
        vr.append(-h_real)  # L_ij = -H_ij (off-diagonal)
        vi.append(-h_imag)

    idx = torch.tensor([rows, cols], dtype=torch.long, device=device)
    vals_r = torch.tensor(vr, dtype=torch.float32, device=device)
    vals_i = torch.tensor(vi, dtype=torch.float32, device=device)

    N = num_nodes
    Lr = torch.sparse_coo_tensor(idx, vals_r, (N, N)).coalesce()
    Li = torch.sparse_coo_tensor(idx, vals_i, (N, N)).coalesce()

    idxT = torch.stack([idx[1], idx[0]], dim=0)
    Lr_T = torch.sparse_coo_tensor(idxT, vals_r, (N, N)).coalesce()
    Li_T = torch.sparse_coo_tensor(idxT, vals_i, (N, N)).coalesce()

    L_sym_real = (Lr * 0.5 + Lr_T * 0.5).coalesce()
    L_skew_im  = (Li * 0.5 - Li_T * 0.5).coalesce()
    return L_sym_real, L_skew_im
