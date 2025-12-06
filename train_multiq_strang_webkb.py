import argparse
import random
import numpy as np
import torch
from torch import optim
from torch_geometric.datasets import WebKB
from torch_geometric.transforms import NormalizeFeatures

from preprocess import build_magnetic_operators, bidirectional_ratio
from strang_heat_net import StrangMagHeatNet, label_smoothing_loss, accuracy
from utils.Citation import citation_datasets


# ---------------- utils ----------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def random_split(num_nodes, train_ratio=0.6, val_ratio=0.2, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=g)

    n_train = int(train_ratio * num_nodes)
    n_val = int(val_ratio * num_nodes)
    n_test = num_nodes - n_train - n_val

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def _extract_split_masks(data, split_id=None, fallback_seed=None):
    has_tm = hasattr(data, "train_mask") and data.train_mask is not None
    has_vm = hasattr(data, "val_mask") and data.val_mask is not None
    has_em = hasattr(data, "test_mask") and data.test_mask is not None

    if has_tm and has_vm and has_em:
        tm, vm, em = data.train_mask, data.val_mask, data.test_mask
        if tm.dim() == 2 and vm.dim() == 2 and em.dim() == 2:
            if split_id is None:
                split_id = 0
            return tm[:, split_id].clone(), vm[:, split_id].clone(), em[:, split_id].clone()
        if tm.dim() == 1 and vm.dim() == 1 and em.dim() == 1:
            return tm.clone(), vm.clone(), em.clone()

    seed = 0 if fallback_seed is None else int(fallback_seed)
    return random_split(data.num_nodes, seed=seed)


def compute_class_weights(y, train_mask, num_classes: int):
    counts = torch.zeros(num_classes, dtype=torch.float32, device=y.device)
    for c in range(num_classes):
        counts[c] = (y[train_mask] == c).sum()
    counts = torch.clamp(counts, min=1.0)
    inv = 1.0 / counts
    inv = inv * (counts.mean() / inv.mean())
    return inv


# --------------- train one split ---------------
def train_one_split(args, data, Lsym, Lskew, split_idx, masks_seed=None):
    device = data.x.device
    num_classes = int(data.y.max().item() + 1)

    train_mask, val_mask, test_mask = _extract_split_masks(
        data, split_id=split_idx, fallback_seed=(args.seed if masks_seed is None else masks_seed)
    )

    class_weight = None
    if args.use_class_weights:
        class_weight = compute_class_weights(data.y, train_mask, num_classes)

    model = StrangMagHeatNet(
        in_dim=int(data.x.size(-1)),
        num_classes=num_classes,
        heat_scales=args.heat_scales,
        K_heat=args.K_heat,
        K_rot=args.K_rot,
        x_proj_dim=args.x_proj_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        tau_t=args.tau_t,  # ignored inside model
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = -1.0
    test_at_best = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        # ---- train step ----
        model.train()
        opt.zero_grad()
        logits = model(data.x, Lsym, Lskew)
        loss = label_smoothing_loss(
            logits[train_mask],
            data.y[train_mask],
            eps=args.label_smooth,
            class_weight=class_weight,
        )
        loss.backward()
        opt.step()

        # ---- eval step ----
        model.eval()
        with torch.no_grad():
            logits = model(data.x, Lsym, Lskew)
            tr = accuracy(logits[train_mask], data.y[train_mask])
            va = accuracy(logits[val_mask], data.y[val_mask])
            te = accuracy(logits[test_mask], data.y[test_mask])

        if va > best_val:
            best_val = va
            test_at_best = te
            best_epoch = epoch

        if epoch % 50 == 1 or epoch in [100, 200, 300, 400, 500, 600, 700, 800]:
            print(
                f"[split {split_idx:02d}] Epoch {epoch:04d} | "
                f"loss {loss.item():.4f} | train {tr:.3f} | val {va:.3f} | test {te:.3f}"
            )

    print(
        f"[RESULT] split {split_idx:02d} | best val {best_val:.3f} @ epoch {best_epoch} | "
        f"test@best {test_at_best:.3f}"
    )
    return best_val, test_at_best


# ---------------- main ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        type=str,
        default="Texas",
        choices=["Cornell", "Texas", "Wisconsin", "Washington", "cora_ml/", "citeseer/", "pubmed_ml/"],
    )
    p.add_argument("--root", type=str, default="./data")
    p.add_argument("--data_path", type=str, default="../dataset/data/tmp/")
    p.add_argument("--q", type=float, default=0.1, help="Magnetic charge (0.1 recommended for WebKB)")
    p.add_argument("--heat_scales", type=float, nargs="+", default=[0.01, 0.6,])
    p.add_argument("--K_heat", type=int, default=6)
    p.add_argument("--K_rot", type=int, default=6)
    p.add_argument("--x_proj_dim", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--tau_t", type=float, default=0.7)  # no effect but kept

    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=850)
    p.add_argument("--label_smooth", type=float, default=0.0)
    p.add_argument("--use_class_weights", action="store_true", default=True)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_splits", type=int, default=2)
    args = p.parse_args()

    print("\n=== StrangMagHeat — multi-scale concat over heat scales (no alpha gate) ===")
    print(
        f"dataset={args.dataset} seed={args.seed} q={args.q} "
        f"heat_scales={args.heat_scales} K_heat={args.K_heat} K_rot={args.K_rot}\n"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    if args.dataset in ["Cornell", "Texas", "Wisconsin", "Washington"]:
        dataset = WebKB(root=args.root, name=args.dataset, transform=NormalizeFeatures())
    else:
        dataset = (
            citation_datasets(root=args.data_path + args.dataset)
            if args.dataset[-1] != "/"
            else citation_datasets(root=args.data_path + args.dataset)
        )
    data = dataset[0].to(device)

    ratio = bidirectional_ratio(data.edge_index, data.num_nodes)
    print("=" * 70)
    print("DIAGNOSTIC CHECKS")
    print("=" * 70)
    print(f"\n(A) Bidirectional edge ratio: {ratio:.4f}\n")

    # precompute magnetic ops for single q
    print(f"(B) Building magnetic operators for q={args.q:.4f}...")
    Lsym, Lskew = build_magnetic_operators(data.edge_index, data.num_nodes, args.q, device)
    print(f"    L_sym nnz={Lsym._nnz()} | L_skew nnz={Lskew._nnz()}")
    print("=" * 70)
    print()

    # decide splits
    use_public_multi = (
        hasattr(data, "train_mask")
        and data.train_mask is not None
        and data.train_mask.dim() == 2
        and hasattr(data, "val_mask")
        and data.val_mask is not None
        and data.val_mask.dim() == 2
        and hasattr(data, "test_mask")
        and data.test_mask is not None
        and data.test_mask.dim() == 2
    )
    total_splits = data.train_mask.size(1) if use_public_multi else args.num_splits
    print(f"Total splits: {total_splits} ({'public' if use_public_multi else 'random'})")

    val_list, test_list = [], []
    set_seed(args.seed)
    masks_seed = None if use_public_multi else args.seed
    for split_idx in range(total_splits):
        print(f"\n=== Training on split {split_idx + 1}/{total_splits} ===")
        bv, tb = train_one_split(args, data, Lsym, Lskew, split_idx=split_idx, masks_seed=masks_seed)
        val_list.append(bv)
        test_list.append(tb)

    val_arr = np.array(val_list, dtype=np.float32)
    test_arr = np.array(test_list, dtype=np.float32)

    def _mean_std(x):
        return float(x.mean()), float(x.std(ddof=1)) if x.size > 1 else 0.0

    vmean, vstd = _mean_std(val_arr)
    tmean, tstd = _mean_std(test_arr)

    print("\n================ SUMMARY ================")
    print(f"Val@best:  mean {vmean:.3f}  std {vstd:.3f}")
    print(f"Test@best: mean {tmean:.3f}  std {tstd:.3f}")
    print("Per-split Test@best:", np.round(test_arr, 3).tolist())
    print("=========================================")


if __name__ == "__main__":
    main()
