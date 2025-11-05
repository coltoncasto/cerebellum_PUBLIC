"""
mni_to_suit.py — Transform a full-brain MNI NIfTI to SUIT space using Functional Fusion.

Workflow:
1) Resample the full-brain input to the source cerebellar atlas grid (default: MNISymC2).
2) Zero out non-cerebellar voxels using the source atlas mask.
3) Deform to the target cerebellar atlas (default: SUIT2) and save.

Usage example:
  python3 mni_to_suit.py \
    --in ../data/input_in_MNI.nii.gz \
    --out ../data/output_in_SUIT2.nii.gz \
    --src MNISymC2 --trg SUIT2 \
    --interp linear \
    --atlas_dir /abs/path/to/Functional_Fusion/Functional_Fusion/Atlases \
    --save_masked_src ../data/input_MNISymC2masked.nii.gz
"""

import argparse
import os
import tempfile
import numpy as np
import nibabel as nb
from nilearn.image import resample_to_img
import Functional_Fusion.atlas_map as am

INTERP_MAP = {"nearest": 0, "linear": 1}


def _resolve_mask_path(info: dict, atlas_dir: str | None) -> str:
    """
    Robustly resolve the absolute path to an atlas mask, given the atlas info dict
    from am.get_atlas(...) and an optional atlas_dir (root of Atlases).
    """
    m = info["mask"]
    if isinstance(m, (list, tuple)):
        if len(m) != 1:
            raise ValueError("This script expects a single mask file for the atlas.")
        m = m[0]

    # Already absolute?
    if os.path.isabs(m):
        if os.path.exists(m):
            return m
        raise FileNotFoundError(f"Mask path listed as absolute but not found: {m}")

    candidates = []

    # Preferred: atlas_dir (root of Functional_Fusion/Functional_Fusion/Atlases)
    if atlas_dir:
        candidates.append(os.path.join(atlas_dir, m))

    # Fallbacks using 'dir' (may be relative subdir like 'tpl-<space>')
    d = info.get("dir")
    if d:
        if os.path.isabs(d):
            candidates.append(os.path.join(d, m))
        else:
            if atlas_dir:
                candidates.append(os.path.join(atlas_dir, d, m))

    for c in candidates:
        if c and os.path.exists(c):
            return c

    tried = ", ".join([c for c in candidates if c])
    raise FileNotFoundError(f"Could not resolve mask path. Tried: {tried or '<none>'}")


def _load_atlas_and_mask(atlas_id: str, atlas_dir: str | None):
    atlas, info = am.get_atlas(atlas_id, atlas_dir=atlas_dir)
    mask_path = _resolve_mask_path(info, atlas_dir)
    return atlas, mask_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in",  dest="infile",  required=True, help="Full-brain NIfTI in MNI space")
    p.add_argument("--out", dest="outfile", required=True, help="Output NIfTI in SUIT space")
    p.add_argument("--src", default="MNISymC2", help="Source cerebellar atlas ID (default: MNISymC2)")
    p.add_argument("--trg", default="SUIT2",   help="Target cerebellar atlas ID (default: SUIT2)")
    p.add_argument("--interp", choices=["nearest","linear"], default="linear",
                   help="Interpolation for sampling/deformation (default: linear)")
    p.add_argument("--atlas_dir", default=None,
                   help="Path to Functional_Fusion/Functional_Fusion/Atlases (RECOMMENDED)")
    p.add_argument("--save_masked_src", default=None,
                   help="Optional path to save the masked source-space NIfTI")
    args = p.parse_args()

    # Load atlases (+ mask for source)
    atlas_src, src_mask_path = _load_atlas_and_mask(args.src, atlas_dir=args.atlas_dir)
    atlas_trg, _ = am.get_atlas(args.trg, atlas_dir=args.atlas_dir)

    # Fail-fast: volumetric cerebellum -> cerebellum only
    if getattr(atlas_src, "structure", None) != "cerebellum":
        raise ValueError(f"Source atlas '{args.src}' is not cerebellum (got {atlas_src.structure}).")
    if getattr(atlas_trg, "structure", None) != "cerebellum":
        raise ValueError(f"Target atlas '{args.trg}' is not cerebellum (got {atlas_trg.structure}).")

    # Load full-brain image and source (MNISymC2) mask, then resample image to the mask grid
    in_img   = nb.load(args.infile)
    mask_img = nb.load(src_mask_path)
    in_rs    = resample_to_img(in_img, mask_img, interpolation="linear")  # align to src grid

    # Zero out non-cerebellar voxels
    data = in_rs.get_fdata().astype(np.float32, copy=False)
    mask = mask_img.get_fdata() > 0
    data[~mask] = 0.0
    masked_src_img = nb.Nifti1Image(data, mask_img.affine, mask_img.header)

    if args.save_masked_src:
        nb.save(masked_src_img, args.save_masked_src)

    # Read masked data via the source atlas (samples only cerebellar voxels)
    with tempfile.TemporaryDirectory() as td:
        tmp_path = os.path.join(td, "masked_src.nii.gz")
        nb.save(masked_src_img, tmp_path)
        X_src = atlas_src.read_data(tmp_path, interpolation=INTERP_MAP[args.interp])

    # Deform to target atlas and save
    X_trg = am.deform_data(X_src, atlas_src, atlas_trg, interpolation=INTERP_MAP[args.interp])
    out_img = atlas_trg.data_to_nifti(X_trg)
    nb.save(out_img, args.outfile)


if __name__ == "__main__":
    main()