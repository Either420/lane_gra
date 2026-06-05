import torch
import torch.nn.functional as F
from model.backbone import resnet
import numpy as np
from utils.common import initialize_weights
from model.seg_model import SegHead
from model.layer import CoordConv
from model.anchor_predictor import DynamicAnchorPredictor

class parsingNet(torch.nn.Module):
    def __init__(self, pretrained=True, backbone='50', num_grid_row=None, num_cls_row=None,
                 num_grid_col=None, num_cls_col=None, num_lane_on_row=None, num_lane_on_col=None,
                 use_aux=False, input_height=None, input_width=None, fc_norm=False,
                 dynamic_anchor=False):
        super(parsingNet, self).__init__()
        self.num_grid_row = num_grid_row
        self.num_cls_row = num_cls_row
        self.num_grid_col = num_grid_col
        self.num_cls_col = num_cls_col
        self.num_lane_on_row = num_lane_on_row
        self.num_lane_on_col = num_lane_on_col
        self.use_aux = use_aux
        self.dynamic_anchor = dynamic_anchor
        self.dim1 = self.num_grid_row * self.num_cls_row * self.num_lane_on_row
        self.dim2 = self.num_grid_col * self.num_cls_col * self.num_lane_on_col
        self.dim3 = 2 * self.num_cls_row * self.num_lane_on_row
        self.dim4 = 2 * self.num_cls_col * self.num_lane_on_col
        self.total_dim = self.dim1 + self.dim2 + self.dim3 + self.dim4
        mlp_mid_dim = 2048
        self.input_dim = input_height // 32 * input_width // 32 * 8

        self.model = resnet(backbone, pretrained=pretrained)
        in_ch = 512 if backbone in ['34', '18', '34fca'] else 2048
        self.pool = torch.nn.Conv2d(in_ch, 8, 1)

        if dynamic_anchor:
            # Anchor predictor: reads from raw backbone features (before pool)
            self.anchor_predictor = DynamicAnchorPredictor(
                in_ch, num_cls_row, num_cls_col,
                row_range=(0.42, 1.0), col_range=(0.0, 1.0),
            )
            # Per-anchor shared MLP: feature = 8 channels × (W/32) width pixels
            feat_w = input_width // 32   # e.g. 50 for 1600-wide input
            feat_h = input_height // 32  # e.g. 10 for 320-tall input
            row_feat_dim = 8 * feat_w    # pool over width → row feature
            col_feat_dim = 8 * feat_h    # pool over height → col feature
            row_out_dim = num_grid_row * num_lane_on_row + 2 * num_lane_on_row
            col_out_dim = num_grid_col * num_lane_on_col + 2 * num_lane_on_col
            self.row_cls = torch.nn.Sequential(
                torch.nn.Linear(row_feat_dim, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, row_out_dim),
            )
            self.col_cls = torch.nn.Sequential(
                torch.nn.Linear(col_feat_dim, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, col_out_dim),
            )
            initialize_weights(self.row_cls, self.col_cls)
        else:
            self.cls = torch.nn.Sequential(
                torch.nn.LayerNorm(self.input_dim) if fc_norm else torch.nn.Identity(),
                torch.nn.Linear(self.input_dim, mlp_mid_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(mlp_mid_dim, self.total_dim),
            )
            initialize_weights(self.cls)

        if self.use_aux:
            self.seg_head = SegHead(backbone, num_lane_on_row + num_lane_on_col)

    def forward(self, x):
        x2, x3, fea = self.model(x)
        if self.use_aux:
            seg_out = self.seg_head(x2, x3, fea)

        pooled = self.pool(fea)   # [B, 8, H/32, W/32]

        if self.dynamic_anchor:
            pred_dict = self._forward_dynamic(fea, pooled)
        else:
            pred_dict = self._forward_fixed(pooled)

        if self.use_aux:
            pred_dict['seg_out'] = seg_out
        return pred_dict

    def _forward_fixed(self, pooled):
        flat = pooled.view(-1, self.input_dim)
        out  = self.cls(flat)
        return {
            'loc_row':   out[:, :self.dim1].view(-1, self.num_grid_row, self.num_cls_row, self.num_lane_on_row),
            'loc_col':   out[:, self.dim1:self.dim1+self.dim2].view(-1, self.num_grid_col, self.num_cls_col, self.num_lane_on_col),
            'exist_row': out[:, self.dim1+self.dim2:self.dim1+self.dim2+self.dim3].view(-1, 2, self.num_cls_row, self.num_lane_on_row),
            'exist_col': out[:, -self.dim4:].view(-1, 2, self.num_cls_col, self.num_lane_on_col),
        }

    def _forward_dynamic(self, fea, pooled):
        B = fea.shape[0]
        row_anchors, col_anchors, row_logits, col_logits = self.anchor_predictor(fea)
        # row_anchors: [B, num_cls_row]  fractions in [0.42, 1.0]
        # col_anchors: [B, num_cls_col]  fractions in [0.0,  1.0]

        # --- row path: sample pooled features at each row anchor ---
        # pooled: [B, 8, H, W]
        # For each anchor i, sample the entire row at y = row_anchors[:, i]
        row_feats = _sample_rows(pooled, row_anchors)   # [B, num_cls_row, 8*W]
        col_feats = _sample_cols(pooled, col_anchors)   # [B, num_cls_col, 8*H]

        # Shared per-anchor MLP
        nr, nc = self.num_cls_row, self.num_cls_col
        row_out = self.row_cls(row_feats.view(B * nr, -1)).view(B, nr, -1)
        col_out = self.col_cls(col_feats.view(B * nc, -1)).view(B, nc, -1)

        ngl, nll = self.num_grid_row, self.num_lane_on_row
        ngc, nlc = self.num_grid_col, self.num_lane_on_col

        # row_out: [B, num_cls_row, ngl*nll + 2*nll]
        loc_row   = row_out[:, :, :ngl * nll].view(B, nr, ngl, nll).permute(0, 2, 1, 3)
        exist_row = row_out[:, :, ngl * nll:].view(B, nr, 2, nll).permute(0, 2, 1, 3)

        loc_col   = col_out[:, :, :ngc * nlc].view(B, nc, ngc, nlc).permute(0, 2, 1, 3)
        exist_col = col_out[:, :, ngc * nlc:].view(B, nc, 2, nlc).permute(0, 2, 1, 3)

        return {
            'loc_row':    loc_row,     # [B, num_grid_row, num_cls_row, num_lane]
            'loc_col':    loc_col,
            'exist_row':  exist_row,   # [B, 2, num_cls_row, num_lane]
            'exist_col':  exist_col,
            'row_anchors': row_anchors,  # [B, num_cls_row]  — passed to inference for GT
            'col_anchors': col_anchors,
            'row_logits':  row_logits,   # [B, num_cls_row]  — for anchor entropy loss
            'col_logits':  col_logits,
        }

    def forward_tta(self, x):
        x2,x3,fea = self.model(x)

        pooled_fea = self.pool(fea)
        n,c,h,w = pooled_fea.shape

        left_pooled_fea = torch.zeros_like(pooled_fea)
        right_pooled_fea = torch.zeros_like(pooled_fea)
        up_pooled_fea = torch.zeros_like(pooled_fea)
        down_pooled_fea = torch.zeros_like(pooled_fea)

        left_pooled_fea[:,:,:,:w-1] = pooled_fea[:,:,:,1:]
        left_pooled_fea[:,:,:,-1] = pooled_fea.mean(-1)

        right_pooled_fea[:,:,:,1:] = pooled_fea[:,:,:,:w-1]
        right_pooled_fea[:,:,:,0] = pooled_fea.mean(-1)

        up_pooled_fea[:,:,:h-1,:] = pooled_fea[:,:,1:,:]
        up_pooled_fea[:,:,-1,:] = pooled_fea.mean(-2)

        down_pooled_fea[:,:,1:,:] = pooled_fea[:,:,:h-1,:]
        down_pooled_fea[:,:,0,:] = pooled_fea.mean(-2)

        fea = torch.cat([pooled_fea, left_pooled_fea, right_pooled_fea, up_pooled_fea, down_pooled_fea], dim = 0)
        fea = fea.view(-1, self.input_dim)

        out = self.cls(fea)

        return {'loc_row': out[:,:self.dim1].view(-1,self.num_grid_row, self.num_cls_row, self.num_lane_on_row),
                'loc_col': out[:,self.dim1:self.dim1+self.dim2].view(-1, self.num_grid_col, self.num_cls_col, self.num_lane_on_col),
                'exist_row': out[:,self.dim1+self.dim2:self.dim1+self.dim2+self.dim3].view(-1, 2, self.num_cls_row, self.num_lane_on_row),
                'exist_col': out[:,-self.dim4:].view(-1, 2, self.num_cls_col, self.num_lane_on_col)}

def _sample_rows(pooled, row_anchors):
    """
    Sample feature-map rows at dynamic anchor positions.

    pooled      : [B, C, H, W]
    row_anchors : [B, N] — fractions in [lo, hi]  (already in [0,1] range of image height)

    For each anchor i, bilinearly sample the full width of the feature map at that row.
    Returns     : [B, N, C*W]
    """
    B, C, H, W = pooled.shape
    N = row_anchors.shape[1]
    # Map row fractions to grid_sample y range [-1, 1]
    y = row_anchors * 2.0 - 1.0                              # [B, N]
    x = torch.linspace(-1, 1, W, device=pooled.device)       # [W]
    x = x.view(1, 1, W).expand(B, N, W)                      # [B, N, W]
    y = y.unsqueeze(2).expand(B, N, W)                        # [B, N, W]
    grid = torch.stack([x, y], dim=-1)                        # [B, N, W, 2]
    sampled = F.grid_sample(pooled, grid, mode='bilinear',
                            align_corners=True)               # [B, C, N, W]
    return sampled.permute(0, 2, 1, 3).flatten(2)            # [B, N, C*W]


def _sample_cols(pooled, col_anchors):
    """
    Sample feature-map columns at dynamic anchor positions.

    col_anchors : [B, N] — fractions in [0, 1]
    Returns     : [B, N, C*H]
    """
    B, C, H, W = pooled.shape
    N = col_anchors.shape[1]
    x = col_anchors * 2.0 - 1.0                              # [B, N]
    y = torch.linspace(-1, 1, H, device=pooled.device)       # [H]
    y = y.view(1, 1, H).expand(B, N, H)                      # [B, N, H]
    x = x.unsqueeze(2).expand(B, N, H)                       # [B, N, H]
    grid = torch.stack([x, y], dim=-1)                        # [B, N, H, 2]
    sampled = F.grid_sample(pooled, grid, mode='bilinear',
                            align_corners=True)               # [B, C, N, H]
    return sampled.permute(0, 2, 1, 3).flatten(2)            # [B, N, C*H]


def get_model(cfg):
    dynamic = getattr(cfg, 'dynamic_anchor', False)
    return parsingNet(
        pretrained=True, backbone=cfg.backbone,
        num_grid_row=cfg.num_cell_row, num_cls_row=cfg.num_row,
        num_grid_col=cfg.num_cell_col, num_cls_col=cfg.num_col,
        num_lane_on_row=cfg.num_lanes, num_lane_on_col=cfg.num_lanes,
        use_aux=cfg.use_aux,
        input_height=cfg.train_height, input_width=cfg.train_width,
        fc_norm=cfg.fc_norm,
        dynamic_anchor=dynamic,
    ).cuda()
