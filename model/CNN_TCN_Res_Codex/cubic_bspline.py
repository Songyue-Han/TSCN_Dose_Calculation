from __future__ import annotations

from collections.abc import Iterable

import torch


def cubic_bspline_weights(t: torch.Tensor) -> torch.Tensor:
    """Return cardinal cubic B-spline basis weights [B0, B1, B2, B3]."""
    t2 = t * t
    t3 = t2 * t
    one_minus_t = 1.0 - t
    b0 = one_minus_t * one_minus_t * one_minus_t / 6.0
    b1 = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0
    b2 = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0
    b3 = t3 / 6.0
    return torch.stack((b0, b1, b2, b3), dim=-1)


def _as_batched_coords(coords_bev_xyz: torch.Tensor, batch_size: int) -> torch.Tensor:
    if coords_bev_xyz.ndim == 2:
        if batch_size != 1:
            raise ValueError("Unbatched coords are only valid for BATCH_SIZE=1.")
        coords_bev_xyz = coords_bev_xyz.unsqueeze(0)
    if coords_bev_xyz.ndim != 3 or coords_bev_xyz.shape[-1] != 3:
        raise ValueError(f"coords_bev_xyz must be [B,N,3] or [N,3], got {tuple(coords_bev_xyz.shape)}")
    if coords_bev_xyz.shape[0] != batch_size:
        raise ValueError(f"Coordinate batch {coords_bev_xyz.shape[0]} does not match coefficients batch {batch_size}")
    return coords_bev_xyz


def sample_cubic_bspline_points(
    coefficients: torch.Tensor,
    coords_bev_xyz: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a 3D cardinal cubic B-spline coefficient lattice at xyz BEV indices.

    Args:
        coefficients: Tensor shaped [B,Z,Y,X].
        coords_bev_xyz: Tensor shaped [B,N,3] or [N,3], continuous xyz index order.

    Returns:
        Tensor shaped [B,N]. Boundary mode is zero padding.
    """
    if coefficients.ndim != 4:
        raise ValueError(f"coefficients must be [B,Z,Y,X], got {tuple(coefficients.shape)}")
    batch_size, depth, height, width = coefficients.shape
    coords = _as_batched_coords(coords_bev_xyz.to(device=coefficients.device), batch_size)
    coords = coords.to(dtype=coefficients.dtype)

    base = torch.floor(coords).to(dtype=torch.long)
    frac = coords - base.to(dtype=coords.dtype)
    weights = cubic_bspline_weights(frac)

    x_base = base[..., 0]
    y_base = base[..., 1]
    z_base = base[..., 2]
    wx = weights[..., 0, :]
    wy = weights[..., 1, :]
    wz = weights[..., 2, :]

    batch_index = torch.arange(batch_size, device=coefficients.device)[:, None]
    out = coefficients.new_zeros((batch_size, coords.shape[1]))
    for cz in range(4):
        iz = z_base + (cz - 1)
        valid_z = (iz >= 0) & (iz < depth)
        iz_safe = iz.clamp(0, max(depth - 1, 0))
        for cy in range(4):
            iy = y_base + (cy - 1)
            valid_y = (iy >= 0) & (iy < height)
            iy_safe = iy.clamp(0, max(height - 1, 0))
            wyz = wz[..., cz] * wy[..., cy]
            valid_yz = valid_z & valid_y
            for cx in range(4):
                ix = x_base + (cx - 1)
                valid = valid_yz & (ix >= 0) & (ix < width)
                ix_safe = ix.clamp(0, max(width - 1, 0))
                values = coefficients[batch_index, iz_safe, iy_safe, ix_safe]
                weight = wyz * wx[..., cx]
                out = out + torch.where(valid, values * weight, torch.zeros_like(values))
    return out


def sample_cubic_bspline_points_unbatched(
    coefficients_zyx: torch.Tensor,
    coords_bev_xyz: torch.Tensor,
) -> torch.Tensor:
    """Fast single-sample cubic B-spline evaluation.

    Args:
        coefficients_zyx: Tensor shaped [Z,Y,X].
        coords_bev_xyz: Tensor shaped [N,3], continuous xyz BEV index order.

    Returns:
        Tensor shaped [N]. Boundary mode is zero padding.
    """
    if coefficients_zyx.ndim != 3:
        raise ValueError(f"coefficients_zyx must be [Z,Y,X], got {tuple(coefficients_zyx.shape)}")
    if coords_bev_xyz.ndim != 2 or coords_bev_xyz.shape[1] != 3:
        raise ValueError(f"coords_bev_xyz must be [N,3], got {tuple(coords_bev_xyz.shape)}")

    depth, height, width = coefficients_zyx.shape
    coords = coords_bev_xyz.to(device=coefficients_zyx.device, dtype=coefficients_zyx.dtype)
    base = torch.floor(coords).to(dtype=torch.long)
    frac = coords - base.to(dtype=coords.dtype)
    weights = cubic_bspline_weights(frac)

    x_base = base[:, 0]
    y_base = base[:, 1]
    z_base = base[:, 2]
    wx = weights[:, 0, :]
    wy = weights[:, 1, :]
    wz = weights[:, 2, :]

    flat = coefficients_zyx.reshape(-1)
    out = coefficients_zyx.new_zeros((coords.shape[0],))
    for cz in range(4):
        iz = z_base + (cz - 1)
        valid_z = (iz >= 0) & (iz < depth)
        iz_safe = iz.clamp(0, max(depth - 1, 0))
        for cy in range(4):
            iy = y_base + (cy - 1)
            valid_y = (iy >= 0) & (iy < height)
            iy_safe = iy.clamp(0, max(height - 1, 0))
            valid_yz = valid_z & valid_y
            weight_yz = wz[:, cz] * wy[:, cy]
            for cx in range(4):
                ix = x_base + (cx - 1)
                valid = valid_yz & (ix >= 0) & (ix < width)
                ix_safe = ix.clamp(0, max(width - 1, 0))
                linear_index = (iz_safe * height + iy_safe) * width + ix_safe
                values = flat[linear_index]
                out = out + torch.where(valid, values * (weight_yz * wx[:, cx]), torch.zeros_like(values))
    return out


def raw_indices_zyx_to_bev_xyz(
    raw_indices_zyx: torch.Tensor,
    raw_index_to_bev_index_affine: torch.Tensor,
) -> torch.Tensor:
    """Transform raw zyx tensor indices into continuous BEV xyz indices."""
    if raw_indices_zyx.ndim != 2 or raw_indices_zyx.shape[1] != 3:
        raise ValueError(f"raw_indices_zyx must be [N,3], got {tuple(raw_indices_zyx.shape)}")
    affine = raw_index_to_bev_index_affine.to(device=raw_indices_zyx.device, dtype=torch.float32)
    idx_xyz = torch.stack(
        (raw_indices_zyx[:, 2], raw_indices_zyx[:, 1], raw_indices_zyx[:, 0]),
        dim=1,
    ).to(dtype=affine.dtype)
    ones = torch.ones((idx_xyz.shape[0], 1), dtype=idx_xyz.dtype, device=idx_xyz.device)
    hom = torch.cat((idx_xyz, ones), dim=1)
    return hom @ affine.T[:, :3]


def raw_chunk_bev_coords_from_affine(
    z0: int,
    z1: int,
    raw_y: int,
    raw_x: int,
    affine: torch.Tensor,
    yy: torch.Tensor,
    xx: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    z = torch.arange(z0, z1, dtype=dtype, device=device)[:, None, None]
    a = affine.to(device=device, dtype=dtype)
    x_bev = xx[None, :, :] * a[0, 0] + yy[None, :, :] * a[0, 1] + z * a[0, 2] + a[0, 3]
    y_bev = xx[None, :, :] * a[1, 0] + yy[None, :, :] * a[1, 1] + z * a[1, 2] + a[1, 3]
    z_bev = xx[None, :, :] * a[2, 0] + yy[None, :, :] * a[2, 1] + z * a[2, 2] + a[2, 3]
    x_bev = x_bev.expand(z1 - z0, raw_y, raw_x)
    y_bev = y_bev.expand(z1 - z0, raw_y, raw_x)
    z_bev = z_bev.expand(z1 - z0, raw_y, raw_x)
    return torch.stack((x_bev, y_bev, z_bev), dim=-1).reshape(-1, 3)


def sample_cubic_bspline_volume(
    coefficients: torch.Tensor,
    raw_index_to_bev_index_affine: torch.Tensor,
    raw_shape_zyx: Iterable[int],
    chunk_z: int = 8,
) -> torch.Tensor:
    """Chunked full raw-volume B-spline evaluation.

    Supports BATCH_SIZE=1 and returns [Z,Y,X] on the same device as coefficients.
    """
    if coefficients.ndim != 4 or coefficients.shape[0] != 1:
        raise ValueError("sample_cubic_bspline_volume currently supports coefficients shaped [1,Z,Y,X].")
    if chunk_z <= 0:
        raise ValueError("chunk_z must be positive.")
    raw_z, raw_y, raw_x = (int(v) for v in raw_shape_zyx)
    affine = raw_index_to_bev_index_affine.to(device=coefficients.device, dtype=coefficients.dtype)
    if affine.ndim == 3:
        if affine.shape[0] != 1:
            raise ValueError("Batched affine is only supported for BATCH_SIZE=1.")
        affine = affine[0]
    if affine.shape != (4, 4):
        raise ValueError(f"raw_index_to_bev_index_affine must be [4,4], got {tuple(affine.shape)}")

    y = torch.arange(raw_y, dtype=coefficients.dtype, device=coefficients.device)
    x = torch.arange(raw_x, dtype=coefficients.dtype, device=coefficients.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    chunks: list[torch.Tensor] = []
    for z0 in range(0, raw_z, chunk_z):
        z1 = min(z0 + chunk_z, raw_z)
        coords = raw_chunk_bev_coords_from_affine(
            z0,
            z1,
            raw_y,
            raw_x,
            affine,
            yy,
            xx,
            coefficients.dtype,
            coefficients.device,
        )
        sampled = sample_cubic_bspline_points_unbatched(coefficients[0], coords).reshape(z1 - z0, raw_y, raw_x)
        chunks.append(sampled)
    return torch.cat(chunks, dim=0)


def coefficients_to_bev_lattice_samples(coefficients: torch.Tensor, chunk_z: int = 8) -> torch.Tensor:
    """Evaluate coefficients at their own integer BEV lattice centers."""
    if coefficients.ndim != 4:
        raise ValueError(f"coefficients must be [B,Z,Y,X], got {tuple(coefficients.shape)}")
    outputs: list[torch.Tensor] = []
    for batch_idx in range(coefficients.shape[0]):
        raw_shape_zyx = coefficients.shape[1:]
        affine = torch.eye(4, dtype=coefficients.dtype, device=coefficients.device)
        sampled = sample_cubic_bspline_volume(
            coefficients[batch_idx : batch_idx + 1],
            affine,
            raw_shape_zyx,
            chunk_z=chunk_z,
        )
        outputs.append(sampled)
    return torch.stack(outputs, dim=0)
