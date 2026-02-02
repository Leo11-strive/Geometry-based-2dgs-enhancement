import numpy as np
import torch


def build_knn_indices_sklearn(xyz: torch.Tensor, k: int) -> torch.Tensor:
    """
    Build kNN indices on CPU using sklearn.

    Returns:
      idx: (N, k) int64 tensor on CPU, excluding self-neighbor.
    """
    if k <= 0:
        raise ValueError("k must be > 0")
    if xyz.numel() == 0:
        return torch.empty((0, k), dtype=torch.int64, device="cpu")

    # Local import: sklearn is optional and only needed when smoothing is enabled.
    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception as e:
        raise ImportError("sklearn is required for build_knn_indices_sklearn(). Install scikit-learn or use build_knn_graph() with an Open3D backend.") from e

    xyz_np = xyz.detach().float().cpu().numpy().astype(np.float32, copy=False)
    n = xyz_np.shape[0]
    kk = min(int(k) + 1, n)
    nn = NearestNeighbors(n_neighbors=kk, algorithm="auto", metric="euclidean")
    nn.fit(xyz_np)
    _, ind = nn.kneighbors(xyz_np, return_distance=True)

    # Drop the first neighbor (self). If kk==n, still ok.
    ind = ind[:, 1:kk]
    if ind.shape[1] < k:
        # Pad by repeating last index (rare: n <= k).
        pad = np.repeat(ind[:, -1:], k - ind.shape[1], axis=1)
        ind = np.concatenate([ind, pad], axis=1)

    return torch.from_numpy(ind.astype(np.int64, copy=False))


def build_knn_graph_sklearn(xyz: torch.Tensor, k: int, dist_sigma_sample: int = 0):
    """
    Build kNN graph on CPU using sklearn and optionally compute the median kNN distance.

    Returns:
      idx: (N, k) int64 tensor on CPU (excluding self)
      d_med: float or None (median over sampled neighbor distances, excluding self)
    """
    if k <= 0:
        raise ValueError("k must be > 0")
    if xyz.numel() == 0:
        return torch.empty((0, k), dtype=torch.int64, device="cpu"), None

    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception as e:
        raise ImportError("sklearn is required for build_knn_graph_sklearn(). Install scikit-learn or use build_knn_graph() with an Open3D backend.") from e

    xyz_np = xyz.detach().float().cpu().numpy().astype(np.float32, copy=False)
    n = xyz_np.shape[0]
    kk = min(int(k) + 1, n)
    nn = NearestNeighbors(n_neighbors=kk, algorithm="auto", metric="euclidean")
    nn.fit(xyz_np)
    # Build kNN indices for all points (this is the expensive part).
    # We intentionally avoid requesting distances for all points to save memory.
    ind = nn.kneighbors(xyz_np, return_distance=False)
    ind = ind[:, 1:kk]
    if ind.shape[1] < k:
        pad = np.repeat(ind[:, -1:], k - ind.shape[1], axis=1)
        ind = np.concatenate([ind, pad], axis=1)

    d_med = None
    sample = int(dist_sigma_sample) if dist_sigma_sample is not None else 0
    if sample < 0:
        sample = 0

    # NOTE: dist_sigma_sample is meant to reduce the cost of *sigma estimation*.
    # It does not reduce the cost of building the full kNN indices (which is still
    # done for all N points), but it avoids computing/storing all distances when
    # you only need a robust median distance.
    if sample == 0 or sample >= n:
        sel = np.arange(n, dtype=np.int64)
    else:
        sel = np.random.choice(n, size=sample, replace=False).astype(np.int64, copy=False)

    neigh = ind[sel]  # (S, k)
    # Compute neighbor distances only for the selected subset.
    # This is cheap compared to building the full kNN index.
    xs = xyz_np[sel][:, None, :]        # (S,1,3)
    xn = xyz_np[neigh]                  # (S,k,3)
    d = np.linalg.norm(xs - xn, axis=-1)  # (S,k)
    d_med = float(np.median(d))

    return torch.from_numpy(ind.astype(np.int64, copy=False)), d_med


def build_knn_graph_o3d(xyz: torch.Tensor, k: int, dist_sigma_sample: int = 0):
    """
    Build kNN graph on CPU using Open3D core NNS (fast C++ KDTree).

    Returns:
      idx: (N, k) int64 tensor on CPU (excluding self)
      d_med: float or None
    """
    if k <= 0:
        raise ValueError("k must be > 0")
    if xyz.numel() == 0:
        return torch.empty((0, k), dtype=torch.int64, device="cpu"), None

    try:
        import open3d as o3d
    except Exception as e:
        raise ImportError("open3d is required for build_knn_graph_o3d().") from e

    if not hasattr(o3d, "core") or not hasattr(o3d.core, "nns"):
        raise ImportError("This Open3D build does not expose o3d.core.nns.NearestNeighborSearch.")

    xyz_np = xyz.detach().float().cpu().numpy().astype(np.float32, copy=False)
    n = xyz_np.shape[0]
    kk = min(int(k) + 1, n)

    pts = o3d.core.Tensor(xyz_np, dtype=o3d.core.Dtype.Float32, device=o3d.core.Device("CPU:0"))
    nns = o3d.core.nns.NearestNeighborSearch(pts)
    nns.knn_index()
    idx_t, dist2_t = nns.knn_search(pts, kk)

    idx = idx_t.cpu().numpy()
    dist2 = dist2_t.cpu().numpy()

    idx = idx[:, 1:kk]
    dist2 = dist2[:, 1:kk]
    if idx.shape[1] < k:
        pad = np.repeat(idx[:, -1:], k - idx.shape[1], axis=1)
        idx = np.concatenate([idx, pad], axis=1)
        pad_d = np.repeat(dist2[:, -1:], k - dist2.shape[1], axis=1)
        dist2 = np.concatenate([dist2, pad_d], axis=1)

    d_med = None
    sample = int(dist_sigma_sample) if dist_sigma_sample is not None else 0
    if sample < 0:
        sample = 0
    if sample == 0 or sample >= n:
        d_med = float(np.median(np.sqrt(dist2)))
    else:
        sel = np.random.choice(n, size=sample, replace=False).astype(np.int64, copy=False)
        d_med = float(np.median(np.sqrt(dist2[sel])))

    return torch.from_numpy(idx.astype(np.int64, copy=False)), d_med


def build_knn_graph(xyz: torch.Tensor, k: int, dist_sigma_sample: int = 0, backend: str = "auto"):
    """
    Backend selector for kNN graph building.
      backend: 'auto' | 'sklearn' | 'open3d'
    """
    backend = str(backend).lower()
    if backend not in ("auto", "sklearn", "open3d"):
        raise ValueError(f"Unknown backend: {backend}")

    if backend in ("auto", "sklearn"):
        try:
            return build_knn_graph_sklearn(xyz, k=k, dist_sigma_sample=dist_sigma_sample)
        except Exception:
            if backend == "sklearn":
                raise
    return build_knn_graph_o3d(xyz, k=k, dist_sigma_sample=dist_sigma_sample)


def quat_to_normal(q: torch.Tensor, order: str = "wxyz") -> torch.Tensor:
    """
    Convert normalized quaternion to the rotated +Z axis (surface normal).
    q: (N,4) on CUDA
    order: "wxyz" or "xyzw"
    returns: (N,3) normalized
    """
    if order == "wxyz":
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    elif order == "xyzw":
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    else:
        raise ValueError(f"Unknown quaternion order: {order} (expected 'wxyz' or 'xyzw')")
    nx = 2.0 * (x * z + w * y)
    ny = 2.0 * (y * z - w * x)
    nz = 1.0 - 2.0 * (x * x + y * y)
    n = torch.stack([nx, ny, nz], dim=-1)
    return torch.nn.functional.normalize(n, dim=-1, eps=1e-8)


def knn_smooth_losses(
    xyz: torch.Tensor,
    quat_wxyz: torch.Tensor,
    rgb_dc: torch.Tensor,
    opacity: torch.Tensor,
    knn_idx_cpu: torch.Tensor,
    dist_sigma: float,
    normal_sigma: float,
    quat_order: str,
    chunk_size: int,
):
    """
    Compute kNN-based smoothness losses:
      - appearance: weighted L2 on rgb_dc
      - opacity: weighted L1 on opacity

    All model tensors are expected on CUDA. knn_idx_cpu is CPU int64.
    """
    device = xyz.device
    n = xyz.shape[0]
    k = knn_idx_cpu.shape[1] if knn_idx_cpu.numel() else 0
    if n == 0 or k == 0:
        z = xyz.new_tensor(0.0)
        return z, z

    dist_sigma = float(dist_sigma)
    normal_sigma = float(normal_sigma)
    if dist_sigma <= 0:
        raise ValueError("dist_sigma must be > 0")

    # Normalize shapes:
    # - rgb_dc should be (N, 3). Some models store DC as (N, 1, 3).
    if rgb_dc.ndim == 3 and rgb_dc.shape[1] == 1 and rgb_dc.shape[2] == 3:
        rgb_dc = rgb_dc[:, 0, :]
    if rgb_dc.ndim != 2 or rgb_dc.shape[0] != n or rgb_dc.shape[1] != 3:
        raise ValueError(f"rgb_dc must be (N,3), got {tuple(rgb_dc.shape)}")

    # - opacity should be (N, 1)
    if opacity.ndim == 1:
        opacity = opacity[:, None]
    if opacity.ndim != 2 or opacity.shape[0] != n or opacity.shape[1] != 1:
        raise ValueError(f"opacity must be (N,1), got {tuple(opacity.shape)}")

    # Precompute normals once per call
    normals = quat_to_normal(quat_wxyz, order=quat_order)

    total_w = xyz.new_tensor(0.0)
    total_color = xyz.new_tensor(0.0)
    total_op = xyz.new_tensor(0.0)

    cs = int(chunk_size) if int(chunk_size) > 0 else int(65536)

    for i0 in range(0, n, cs):
        i1 = min(n, i0 + cs)
        idx = knn_idx_cpu[i0:i1].to(device=device, non_blocking=True).long()

        xi = xyz[i0:i1]  # (B,3)
        xj = xyz[idx]    # (B,K,3)
        d2 = (xi[:, None, :] - xj).pow(2).sum(dim=-1)  # (B,K)
        w = torch.exp(-0.5 * d2 / (dist_sigma * dist_sigma))

        if normal_sigma > 0:
            ni = normals[i0:i1]  # (B,3)
            nj = normals[idx]    # (B,K,3)
            cos = (ni[:, None, :] * nj).sum(dim=-1).clamp(-1.0, 1.0)
            ang = 1.0 - cos
            w = w * torch.exp(-0.5 * (ang / normal_sigma).pow(2))

        # Appearance (L2)
        ci = rgb_dc[i0:i1]          # (B,3)
        cj = rgb_dc[idx]            # (B,K,3)
        cdiff = (ci[:, None, :] - cj).pow(2).sum(dim=-1).add(1e-12).sqrt()  # (B,K)
        total_color = total_color + (w * cdiff).sum()

        # Opacity (L1) - more conservative
        oi = opacity[i0:i1, 0]      # (B,)
        oj = opacity[idx, 0]        # (B,K)
        odiff = (oi[:, None] - oj).abs()
        total_op = total_op + (w * odiff).sum()

        total_w = total_w + w.sum()

    denom = torch.clamp_min(total_w, 1.0)
    return total_color / denom, total_op / denom
