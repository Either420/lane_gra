import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicAnchorPredictor(nn.Module):
    """
    Predicts non-uniform sorted anchor positions per image from backbone features.

    Idea: predict a probability density over the anchor axis via softmax.
    Higher weight at position i → anchors cluster there (e.g. curved lane region).
    Final anchors are the CDF midpoints → always sorted, always in [lo, hi], differentiable.
    """

    def __init__(self, in_channels, num_row, num_col,
                 row_range=(0.42, 1.0), col_range=(0.0, 1.0)):
        super().__init__()
        self.num_row = num_row
        self.num_col = num_col
        self.row_range = row_range   # fraction of image height
        self.col_range = col_range   # fraction of image width

        # Compress channels first (cheap)
        self.compress = nn.Sequential(
            nn.Conv2d(in_channels, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Row anchors: attend to vertical structure
        # Pool over width → 1-D vertical signal → adaptive pool to 16 bins → MLP
        self.row_pool   = nn.AdaptiveAvgPool2d((16, 1))   # [B, 64, 16, 1]
        self.row_head   = nn.Sequential(
            nn.Flatten(),                                  # [B, 64*16]
            nn.Linear(64 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_row),
        )

        # Col anchors: attend to horizontal structure
        self.col_pool   = nn.AdaptiveAvgPool2d((1, 16))   # [B, 64, 1, 16]
        self.col_head   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_col),
        )

    def forward(self, x4):
        feat = self.compress(x4)                           # [B, 64, H, W]

        row_logits = self.row_head(self.row_pool(feat))    # [B, num_row]
        col_logits = self.col_head(self.col_pool(feat))    # [B, num_col]

        row_anchors = _logits_to_anchors(row_logits, self.row_range)
        col_anchors = _logits_to_anchors(col_logits, self.col_range)

        # also return logits so the anchor-entropy loss can be applied
        return row_anchors, col_anchors, row_logits, col_logits


def _logits_to_anchors(logits, value_range):
    """
    Convert raw logits to N sorted anchor positions in [lo, hi].

    Steps:
      1. softmax(logits)  →  probability mass p_i  (all positive, sum = 1)
      2. CDF midpoints:  cdf_i = sum_{j<i} p_j + p_i/2  ∈ (0, 1)
      3. scale to [lo, hi]

    Gradient path: logits → softmax → cumsum → anchors  (fully differentiable)
    Regions where p_i is large → anchors cluster there.
    """
    lo, hi = value_range
    weights = F.softmax(logits, dim=1)                     # [B, N]
    cdf     = torch.cumsum(weights, dim=1) - weights * 0.5 # [B, N] midpoints in (0,1)
    return lo + (hi - lo) * cdf                            # [B, N] in [lo, hi]


# ---------------------------------------------------------------------------
# Differentiable GT label interpolation at dynamic row/col anchors
# (replaces my_interp.run when anchors are per-image)
# ---------------------------------------------------------------------------

def interp_label_rows(points, query_rows_px):
    """
    For each image, for each lane, linearly interpolate x at dynamic row positions.

    points        : [B, num_lanes, N_pts, 2]  — (x_px, row_px) per stored point
                    invalid points have x_px == -99999 (from cache script)
    query_rows_px : [B, num_row]              — row pixel positions to query

    Returns       : [B, num_lanes, num_row]   — interpolated x in pixels,
                    -1.0 where the lane is absent at that row.
    """
    B, L, N, _ = points.shape
    num_row = query_rows_px.shape[1]

    x_src   = points[..., 0]   # [B, L, N]
    row_src = points[..., 1]   # [B, L, N]

    # Mark invalid source points
    valid_src = (x_src > -9000).float()   # [B, L, N]  1 = valid

    # ---- find lower and upper neighbours for each query ----
    # row_src : [B, L, 1,  N]
    # q       : [B, 1, num_row, 1]
    r_exp = row_src.unsqueeze(2)                       # [B, L, 1, N]
    q_exp = query_rows_px.unsqueeze(1).unsqueeze(-1)   # [B, 1, num_row, 1]

    # above[b, l, i, j] = True when source row j is > query i
    above = (r_exp > q_exp)                            # [B, L, num_row, N]

    # upper_idx: first source index above query
    # If nothing is above, clamp to N-1
    has_above    = above.any(-1)                       # [B, L, num_row]
    upper_idx    = above.long().argmax(-1)             # [B, L, num_row]
    upper_idx    = upper_idx.clamp(max=N - 1)
    lower_idx    = (upper_idx - 1).clamp(min=0)

    def gather(t, idx):
        # t: [B, L, N], idx: [B, L, num_row]
        Bi, Li, Ni = t.shape
        out = t.reshape(Bi * Li, Ni).gather(
            1, idx.reshape(Bi * Li, num_row)
        )
        return out.reshape(Bi, Li, num_row)

    x_lo  = gather(x_src,   lower_idx)   # [B, L, num_row]
    x_hi  = gather(x_src,   upper_idx)
    r_lo  = gather(row_src, lower_idx)
    r_hi  = gather(row_src, upper_idx)
    v_lo  = gather(valid_src, lower_idx)
    v_hi  = gather(valid_src, upper_idx)

    # Linear interpolation weight
    denom = (r_hi - r_lo).abs().clamp(min=1.0)
    q     = query_rows_px.unsqueeze(1).expand(B, L, num_row)
    t     = ((q - r_lo) / denom).clamp(0.0, 1.0)

    x_interp = x_lo + t * (x_hi - x_lo)   # [B, L, num_row]

    # Invalidate where both neighbours are invalid or query is outside lane range
    both_invalid = (v_lo + v_hi) == 0
    x_interp[both_invalid] = -1.0

    return x_interp   # [B, L, num_row]


def interp_label_cols(points, query_cols_px):
    """Same idea for column direction: interpolate row_px at dynamic col positions."""
    B, L, N, _ = points.shape
    num_col = query_cols_px.shape[1]

    row_src = points[..., 1]   # [B, L, N]
    x_src   = points[..., 0]   # [B, L, N]  ← now used as the "position axis"
    valid_src = (x_src > -9000).float()

    c_exp = x_src.unsqueeze(2)
    q_exp = query_cols_px.unsqueeze(1).unsqueeze(-1)

    above     = (c_exp > q_exp)
    upper_idx = above.long().argmax(-1).clamp(max=N - 1)
    lower_idx = (upper_idx - 1).clamp(min=0)

    def gather(t, idx):
        Bi, Li, Ni = t.shape
        out = t.reshape(Bi * Li, Ni).gather(
            1, idx.reshape(Bi * Li, num_col)
        )
        return out.reshape(Bi, Li, num_col)

    row_lo = gather(row_src,  lower_idx)
    row_hi = gather(row_src,  upper_idx)
    x_lo   = gather(x_src,   lower_idx)
    x_hi   = gather(x_src,   upper_idx)
    v_lo   = gather(valid_src, lower_idx)
    v_hi   = gather(valid_src, upper_idx)

    denom  = (x_hi - x_lo).abs().clamp(min=1.0)
    q      = query_cols_px.unsqueeze(1).expand(B, L, num_col)
    t      = ((q - x_lo) / denom).clamp(0.0, 1.0)
    r_interp = row_lo + t * (row_hi - row_lo)

    both_invalid = (v_lo + v_hi) == 0
    r_interp[both_invalid] = -1.0

    return r_interp   # [B, L, num_col]
