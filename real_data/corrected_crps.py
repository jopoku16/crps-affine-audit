"""Stable scoring wrappers around the pinned public compressor."""
import torch
from utils.kernel_quadrature import kq_compress


CANONICAL_GRID = 1_000_000
HALF_GRID_GUARD = 1e-6


def canonicalize_standardized(values):
    """Round nonnegative standardized values with a guarded half-up rule.

    Positive-affine recodings can place the same mathematical value just above
    and just below a half-grid boundary in float64. The small guard acts in
    grid units and makes those machine-level perturbations choose the same
    side. It changes a standardized coordinate by at most about 1e-6.
    """
    values = torch.as_tensor(values, dtype=torch.float64)
    return (
        torch.floor(values * CANONICAL_GRID + 0.5 + HALF_GRID_GUARD)
        / CANONICAL_GRID
    )


def weighted_empirical_score(weights, nodes, observation):
    """CRPS of the normalized weighted empirical distribution."""
    w = weights.double()
    z = nodes.double()
    w = w / w.sum()
    distances = (z[:, None] - z[None, :]).abs()
    return w @ (z - observation).abs() - 0.5 * (w @ distances @ w)


def anchored_weighted_compression(samples, observation, n_low_rank, n_compressed, seed):
    """Anchor the ensemble before compression and score its weighted distribution.

    The anchor is computed in float64. Subtracting the minimum member keeps the
    compressor input nonnegative and unchanged under a common translation.
    """
    x = torch.as_tensor(samples, dtype=torch.float64)
    y = torch.as_tensor(observation, dtype=torch.float64)
    anchor = x.min()
    centered = (x - anchor).float()
    torch.manual_seed(int(seed))
    weights, nodes = kq_compress(centered, n_low_rank, n_compressed)
    return weighted_empirical_score(weights, nodes, y - anchor), weights, nodes


def affine_standardized_weighted_compression(
    samples, observation, n_low_rank, n_compressed, seed
):
    """Score compressed samples after location and range standardization.

    The wrapper uses the sample minimum as its location and the sample range as
    its scale. A positive affine recoding therefore gives the compressor the
    same standardized input. The weighted empirical CRPS is mapped back to the
    original units after compression.

    A constant ensemble is handled exactly without calling the compressor.
    """
    x = torch.as_tensor(samples, dtype=torch.float64).reshape(-1)
    y = torch.as_tensor(observation, dtype=torch.float64)
    if x.numel() == 0:
        raise ValueError("samples must contain at least one value")
    if not torch.isfinite(x).all() or not torch.isfinite(y):
        raise ValueError("samples and observation must be finite")

    location = x.min()
    scale = x.max() - location
    if scale == 0:
        value = (x[0] - y).abs()
        weights = torch.ones(1, dtype=torch.float64)
        nodes = torch.zeros(1, dtype=torch.float64)
        return value, weights, nodes, location, scale

    standardized64 = canonicalize_standardized((x - location) / scale)
    # The pinned compressor uses float32. Canonicalization before conversion
    # makes small float64 differences from equivalent recodings choose the same
    # grid point, including values that lie on a half-grid boundary.
    standardized = standardized64.float()
    standardized_observation = (y - location) / scale
    torch.manual_seed(int(seed))
    weights, nodes = kq_compress(
        standardized, min(int(n_low_rank), x.numel()), min(int(n_compressed), x.numel())
    )
    value = scale * weighted_empirical_score(
        weights, nodes, standardized_observation
    )
    return value, weights, nodes, location, scale
