import torch
import torch.nn.functional as F


def _sobel_kernels(device, dtype):
    kx = torch.tensor(
        [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
        device=device,
        dtype=dtype,
    ) / 8.0
    ky = torch.tensor(
        [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]],
        device=device,
        dtype=dtype,
    ) / 8.0
    return kx, ky


def image_grad_mag_sobel(img: torch.Tensor, gray: bool = True, norm: str = "mean") -> torch.Tensor:
    """
    Compute Sobel gradient magnitude.

    Args:
      img: (3,H,W) or (1,H,W) float tensor in [0,1] (recommended)
      gray: whether to convert RGB->luma before Sobel
      norm: 'none' | 'mean' | 'max' normalization of the gradient magnitude

    Returns:
      grad: (H,W) float tensor
    """
    if img.ndim != 3:
        raise ValueError(f"img must be (C,H,W), got {tuple(img.shape)}")
    c, h, w = img.shape
    if c not in (1, 3):
        raise ValueError(f"img channels must be 1 or 3, got {c}")

    x = img
    if c == 3 and gray:
        # Luma (approx. Rec.601)
        x = (0.2989 * x[0:1] + 0.5870 * x[1:2] + 0.1140 * x[2:3])
    elif c == 3:
        # Average channels for a stable single-channel gradient magnitude
        x = x.mean(dim=0, keepdim=True)

    x = x.unsqueeze(0)  # (1,1,H,W)
    kx, ky = _sobel_kernels(x.device, x.dtype)
    kx = kx.view(1, 1, 3, 3)
    ky = ky.view(1, 1, 3, 3)

    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    grad = torch.sqrt(gx * gx + gy * gy + 1e-12).squeeze(0).squeeze(0)  # (H,W)

    norm = str(norm).lower()
    if norm == "none":
        return grad
    if norm == "mean":
        return grad / (grad.mean() + 1e-12)
    if norm == "max":
        return grad / (grad.max() + 1e-12)
    raise ValueError(f"Unknown norm: {norm} (expected 'none'|'mean'|'max')")


def rgb_grad_weight(
    rgb: torch.Tensor,
    alpha: float = 10.0,
    gray: bool = True,
    norm: str = "mean",
    w_min: float = 0.05,
    w_max: float = 1.0,
) -> torch.Tensor:
    """
    DN-Splatter-style edge-aware weight:
      w = exp(-alpha * ||∇I||)
    Stronger (closer to 1) in low-texture regions, weaker near edges.

    Args:
      rgb: (3,H,W) float tensor
    Returns:
      w: (H,W) float tensor in [w_min, w_max]
    """
    grad = image_grad_mag_sobel(rgb, gray=gray, norm=norm)
    w = torch.exp(-float(alpha) * grad)
    return w.clamp(min=float(w_min), max=float(w_max))


def specular_mask(rgb: torch.Tensor, tV: float = 0.92, tS: float = 0.15) -> torch.Tensor:
    """
    Heuristic specular/highlight mask from GT RGB.

    Args:
      rgb: (3,H,W) float tensor in [0,1]
      tV: brightness threshold (max channel)
      tS: "whiteness" threshold (lower => more white)

    Returns:
      spec: (H,W) bool tensor
    """
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError(f"rgb must be (3,H,W), got {tuple(rgb.shape)}")
    V = rgb.max(dim=0).values
    m = rgb.min(dim=0).values
    S = (V - m) / (V + 1e-6)
    return (V > float(tV)) & (S < float(tS))
