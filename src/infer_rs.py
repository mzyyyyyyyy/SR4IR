"""
Large-scale tile inference for SR4IR regression (base1_v2).

Pipeline per patch:
  Landsat LR  →  per-band MinMax [0,1]  →  SwinIR (×4 SR)  →  DAv2  →  CHM (metres)

The model predicts CHM directly in metres (target was NOT normalised during
training: normalize_target=False, minmax_target=False, unit_scale_ratio=1),
so no inverse transform is needed on the output.

Usage:
  cd SR4IR/src
  python infer_rs.py --config ../options/reg/base1_v2_pred.yaml
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))  # make 'archs', 'models', … importable

import argparse
import glob as _glob
import json
import logging
from itertools import product
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
from rasterio import windows
from rasterio.transform import from_bounds
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import yaml


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Normalisation — mirrors data/reg.py load_reg_data + get_multiscale_transforms
# ---------------------------------------------------------------------------

def _load_input_stats(stats_file: str, year: int, selected_bands: list) -> tuple:
    """
    Returns (p1_arr, p99_arr) both shape (C,) float32, already filtered to
    selected_bands — same as:
        gb_stats = all_stats[str(year)]
        gb_stats = {k: [v[i] for i in selected_bands] for k, v in gb_stats.items()}
        p1  = gb_stats["p1"]   # list of len(selected_bands)
        p99 = gb_stats["p99"]
    """
    with open(stats_file) as f:
        all_stats = json.load(f)
    stats = all_stats[str(year)]
    p1  = np.array([stats["p1"][b]  for b in selected_bands], dtype=np.float32)
    p99 = np.array([stats["p99"][b] for b in selected_bands], dtype=np.float32)
    return p1, p99


def _minmax_normalize(data: np.ndarray, p1: np.ndarray, p99: np.ndarray) -> np.ndarray:
    """
    Per-band MinMaxScale to [0, 1], matching data/data_transform.py MinMaxScale.

    data : (C, H, W)  float32
    p1   : (C,)
    p99  : (C,)
    """
    p1  = p1.reshape(-1, 1, 1)
    p99 = p99.reshape(-1, 1, 1)
    out = (data - p1) / (p99 - p1 + 1e-8)
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Blending mask & patch offsets
# ---------------------------------------------------------------------------

def _create_blending_mask(size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.float32)
    c = size // 2
    mask[c, c] = 1.0
    mask = gaussian_filter(mask, sigma=size / 4)
    mask /= mask.max()
    return mask


def _get_offsets(nrows: int, ncols: int, patch_size: int, stride: int) -> list:
    row_offs = list(range(0, nrows - patch_size + 1, stride))
    col_offs = list(range(0, ncols - patch_size + 1, stride))
    if nrows % stride != 0:
        row_offs.append(nrows - patch_size)
    if ncols % stride != 0:
        col_offs.append(ncols - patch_size)
    return list(product(col_offs, row_offs))


# ---------------------------------------------------------------------------
# Network builders — avoids pulling in the full training framework
# ---------------------------------------------------------------------------

def _build_swinir(opt_sr: dict, scale: int) -> nn.Module:
    from archs.common.swinir_arch import SwinIR
    opt = {k: v for k, v in opt_sr.items() if k not in ('name', 'trainable')}
    opt['upscale'] = scale
    return SwinIR(**opt)


def _build_dav2(opt_reg: dict) -> nn.Module:
    from archs.reg.dav2_arch import build_network
    opt = {k: v for k, v in opt_reg.items() if k not in ('name', 'trainable')}
    return build_network(**opt)


def _load_state(model: nn.Module, path: str, key: str = None):
    ckpt = torch.load(path, map_location='cpu')
    state = ckpt[key] if (key and key in ckpt) else ckpt
    model.load_state_dict(state, strict=True)
    logging.info(f"Loaded weights from {path}")


# ---------------------------------------------------------------------------
# Per-tile inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_tile(
    lr_path: Path,
    net_sr: nn.Module,
    net_reg: nn.Module,
    device: torch.device,
    selected_bands: list,
    p1: np.ndarray,
    p99: np.ndarray,
    out_scale: int,
    patch_size_lr: int,
    stride_lr: int,
    blending_mask: np.ndarray,
) -> tuple:
    """
    Returns (pred_metres [H_hr, W_hr], rasterio meta dict).

    The model predicts CHM in metres directly (no target normalisation was
    applied during training), so the output array is in metres.
    """
    n_bands = len(selected_bands)

    with rasterio.open(lr_path) as src:
        nols  = src.meta['width']
        nrows = src.meta['height']
        meta  = src.meta.copy()
        bounds = src.bounds
        crs    = src.crs

        H_hr = round(nrows * out_scale)
        W_hr = round(nols  * out_scale)

        output     = np.zeros((H_hr, W_hr), dtype=np.float32)
        weight_map = np.zeros((H_hr, W_hr), dtype=np.float32)

        big_win = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
        offsets = _get_offsets(nrows, nols, patch_size_lr, stride_lr)

        for col_off, row_off in tqdm(offsets, desc=lr_path.name, leave=False):
            win = windows.Window(
                col_off=col_off, row_off=row_off,
                width=patch_size_lr, height=patch_size_lr,
            ).intersection(big_win)

            # Read ALL bands at native LR resolution, then select by 0-index —
            # mirrors LazyMultiScaleDataset._load_multiband_patch.
            raw = src.read(window=win).astype(np.float32)  # (all_bands, H, W)
            max_idx = max(selected_bands)
            if max_idx >= raw.shape[0]:
                avail = list(range(min(n_bands, raw.shape[0])))
                data = raw[avail]
            else:
                data = raw[selected_bands]                  # (C, H, W)

            # Per-band MinMax to [0, 1] — matches MinMaxScale in get_multiscale_transforms
            data = _minmax_normalize(data, p1, p99)

            # (1, C, H, W) tensor
            img_ls = torch.from_numpy(data).unsqueeze(0).to(device)

            # SwinIR needs float32 (AMP causes NaN in attention — see train_one_epoch)
            img_sr = net_sr(img_ls.float())                     # (1, C, H_hr, W_hr)

            # DAv2 regression head
            pred = net_reg(img_sr).predicted_depth              # (1, H_hr, W_hr)

            pred_np = pred.squeeze().cpu().float().numpy()      # (H_hr, W_hr), metres

            # Place in output with Gaussian blending
            r_hr  = round(row_off * out_scale)
            c_hr  = round(col_off * out_scale)
            h_out = pred_np.shape[0]
            w_out = pred_np.shape[1]

            bld = blending_mask[:h_out, :w_out]
            output[r_hr:r_hr + h_out, c_hr:c_hr + w_out]     += pred_np * bld
            weight_map[r_hr:r_hr + h_out, c_hr:c_hr + w_out] += bld

    final = np.divide(output, weight_map, out=np.zeros_like(output), where=weight_map > 0)

    out_transform = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, W_hr, H_hr)
    meta.update({
        'count':     1,
        'width':     W_hr,
        'height':    H_hr,
        'dtype':     'float32',
        'crs':       crs,
        'transform': out_transform,
        'compress':  'lzw',
        'driver':    'GTiff',
        'nodata':    -9999.0,
    })
    return final, meta


def write_tif(arr: np.ndarray, meta: dict, out_path: Path):
    arr = np.clip(arr, 0.0, None)   # CHM cannot be negative
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, 'w', **meta) as dst:
        dst.write(arr.astype(np.float32), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='SR4IR tile inference')
    parser.add_argument('--config', type=str, default='./options/reg/base1_v2_pred.yaml',
                        help='Path to prediction YAML config')
    parser.add_argument('--rank', type=int, default=0)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    ds  = cfg['dataset']
    mdl = cfg['model']
    pred_cfg = cfg['prediction']

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s  %(levelname)s  %(message)s')

    device = torch.device(f"cuda:{args.rank}" if torch.cuda.is_available() else 'cpu')
    logging.info(f'device = {device}')

    # ---- Input normalisation stats (same as reg.py load_reg_data) ----
    p1, p99 = _load_input_stats(
        stats_file=ds['globalnorm_stats_file'],
        year=ds['year'],
        selected_bands=ds['selected_bands'],
    )

    # ---- Build networks ----
    scale = cfg.get('scale', 4)

    net_sr  = _build_swinir(cfg['network_sr'], scale=scale).to(device)
    net_reg = _build_dav2(cfg['network_reg']).to(device)

    _load_state(net_sr,  mdl['net_sr_path'])
    _load_state(net_reg, mdl['net_reg_path'])

    net_sr.eval()
    net_reg.eval()

    # ---- Blending mask at HR patch size ----
    patch_size_lr = pred_cfg['patch_size_lr']
    patch_size_hr = patch_size_lr * scale
    blending_mask = _create_blending_mask(patch_size_hr)

    # ---- Collect input tiles ----
    # Structure: input_dir/<tile_subdir>/*{p50}*.jp2  (one file per subdir)
    # Mirrors LazyMultiScaleDataset._gather_tile_ids glob pattern.
    input_dir = ds['input_dir']
    file_ext  = ds['file_type_input']
    percentile = ds['selected_percentile'][0]
    matched = _glob.glob(os.path.join(input_dir, '*', f'*{percentile}*.{file_ext}'))
    all_files = sorted(Path(p) for p in matched)
    logging.info(f'Found {len(all_files)} input tile(s).')

    # ---- Output directory ----
    out_dir = Path(ds['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    for lr_path in tqdm(all_files, desc='Tiles'):
        out_path = out_dir / f'{lr_path.stem}_sr4ir_pred.tif'
        if out_path.exists():
            logging.info(f'Skipping (exists): {out_path}')
            continue

        logging.info(f'Processing: {lr_path}')
        pred, meta = predict_tile(
            lr_path=lr_path,
            net_sr=net_sr,
            net_reg=net_reg,
            device=device,
            selected_bands=ds['selected_bands'],
            p1=p1,
            p99=p99,
            out_scale=scale,
            patch_size_lr=patch_size_lr,
            stride_lr=pred_cfg['stride_lr'],
            blending_mask=blending_mask,
        )
        write_tif(pred, meta, out_path)
        logging.info(f'Saved: {out_path}')

    logging.info('Done.')


if __name__ == '__main__':
    main()
