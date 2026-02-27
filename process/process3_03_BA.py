# ============================================================================
# COLMAP binary writers (unchanged from original)
# ============================================================================

import numpy as np
import struct
from pathlib import Path
from scipy.spatial.transform import Rotation


def write_next_bytes(fid, data, format_str):
    if isinstance(data, (list, tuple, np.ndarray)):
        fid.write(struct.pack("<" + format_str, *data))
    else:
        fid.write(struct.pack("<" + format_str, data))


def matrix_to_quaternion_translation(matrix_3x4: np.ndarray):
    """3×4 [R | t] → COLMAP quaternion [w, x, y, z] + translation t."""
    R = matrix_3x4[:3, :3]
    t = matrix_3x4[:3, 3]
    rot  = Rotation.from_matrix(R)
    quat = rot.as_quat()            # [x, y, z, w]
    qvec = np.array([quat[3], quat[0], quat[1], quat[2]])  # COLMAP: [w, x, y, z]
    return qvec, t


def write_cameras_binary(cameras, path):
    with open(path, "wb") as fid:
        write_next_bytes(fid, len(cameras), "Q")
        for cam_id, cam in cameras.items():
            write_next_bytes(fid, cam_id, "I")
            write_next_bytes(fid, 1, "I")          # PINHOLE
            write_next_bytes(fid, cam['width'],  "Q")
            write_next_bytes(fid, cam['height'], "Q")
            for p in cam['params']:
                write_next_bytes(fid, float(p), "d")


def write_images_binary(images_data, path):
    with open(path, "wb") as fid:
        write_next_bytes(fid, len(images_data), "Q")
        for img_id, img in images_data.items():
            write_next_bytes(fid, img_id,         "I")
            write_next_bytes(fid, img['qvec'],    "dddd")
            write_next_bytes(fid, img['tvec'],    "ddd")
            write_next_bytes(fid, img['camera_id'], "I")
            for char in img['name']:
                write_next_bytes(fid, char.encode("utf-8"), "c")
            write_next_bytes(fid, b"\x00", "c")
            write_next_bytes(fid, len(img['xys']), "Q")
            for xy, pid in zip(img['xys'], img['point3D_ids']):
                write_next_bytes(fid, xy, "dd")
                write_next_bytes(fid, pid, "Q")


def write_points3d_binary(points3D, path):
    with open(path, "wb") as fid:
        write_next_bytes(fid, len(points3D), "Q")
        for pid, pt in enumerate(points3D):
            write_next_bytes(fid, pid,        "Q")
            write_next_bytes(fid, pt['xyz'],  "ddd")
            write_next_bytes(fid, pt['rgb'],  "BBB")
            write_next_bytes(fid, pt['error'],"d")
            write_next_bytes(fid, len(pt['image_ids']), "Q")
            for iid, p2d_idx in zip(pt['image_ids'], pt['point2D_idxs']):
                write_next_bytes(fid, int(iid),    "I")
                write_next_bytes(fid, int(p2d_idx),"I")


# ============================================================================
# VGGT → COLMAP conversion  (replaces extract_scene_data / convert_mast3r_to_colmap)
# ============================================================================

def extract_vggt_scene_data(predictions, image_paths, conf_threshold_pct=20.0, verbose=True):
    """
    Convert VGGT predictions to COLMAP cameras / images / points3D structures.

    Args:
        predictions      : dict returned by run_vggt()
        image_paths      : list of source image paths (for colors & names)
        conf_threshold_pct : drop the lowest N% confidence points
        verbose          : print progress

    Returns:
        cameras    : {camera_id: {...}}
        images_data: {image_id:  {...}}
        points3D   : [{xyz, rgb, error, image_ids, point2D_idxs}]
    """
    import numpy as np
    import torch

    cameras     = {}
    images_data = {}
    points3D    = []

    # ── Grab tensors ──────────────────────────────────────────────────────
    extrinsic = predictions["extrinsic"]          # (S, 3, 4)
    intrinsic  = predictions["intrinsic"]          # (S, 3, 3)

    # Prefer depth-unprojected points (more accurate)
    if "world_points_from_depth" in predictions:
        pts3d = predictions["world_points_from_depth"]   # (S, H, W, 3)
        conf  = predictions.get("depth_conf", None)      # (S, H, W)  or  (S, H, W, 1)
        if verbose:
            print("  Using depth-unprojected 3D points")
    else:
        pts3d = predictions["world_points"]               # (S, H, W, 3)
        conf  = predictions.get("world_points_conf", None)
        if verbose:
            print("  Using world_points 3D points")

    if isinstance(extrinsic, torch.Tensor):
        extrinsic = extrinsic.detach().cpu().numpy()
    if isinstance(intrinsic, torch.Tensor):
        intrinsic = intrinsic.detach().cpu().numpy()
    if isinstance(pts3d, torch.Tensor):
        pts3d = pts3d.detach().cpu().numpy()
    if conf is not None and isinstance(conf, torch.Tensor):
        conf = conf.detach().cpu().numpy()

    # Squeeze trailing dim if depth_conf has shape (S, H, W, 1)
    if conf is not None and conf.ndim == 4:
        conf = conf.squeeze(-1)

    S, H, W = pts3d.shape[:3]

    if verbose:
        print(f"  Views: {S}   Point map: {H}x{W}")

    # ── Build COLMAP image size from the original image ───────────────────
    # VGGT internally resizes; we store the preprocessed size for COLMAP
    img_h, img_w = H, W

    # ── Cameras & images ──────────────────────────────────────────────────
    for idx in range(S):
        K   = intrinsic[idx]    # (3, 3)
        ext = extrinsic[idx]    # (3, 4)

        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])

        cameras[idx] = {
            'model' : 'PINHOLE',
            'width' : img_w,
            'height': img_h,
            'params': [fx, fy, cx, cy]
        }

        # VGGT extrinsic is camera-from-world [R | t] — same as COLMAP convention
        # No inversion needed.
        qvec, tvec = matrix_to_quaternion_translation(ext)

        img_name = Path(image_paths[idx]).name if idx < len(image_paths) else f"image_{idx:04d}.jpg"

        images_data[idx + 1] = {
            'qvec'       : qvec,
            'tvec'       : tvec,
            'camera_id'  : idx,
            'name'       : img_name,
            'xys'        : np.empty((0, 2)),
            'point3D_ids': np.empty((0,), dtype=np.int64),
        }

    # ── 3D points ─────────────────────────────────────────────────────────
    if verbose:
        print("  Extracting 3D points with colors...")

    # Build global confidence mask
    if conf is not None:
        conf_flat = conf.reshape(-1)
        thr = np.percentile(conf_flat[np.isfinite(conf_flat)], conf_threshold_pct)
        conf_mask = (conf >= thr).reshape(S, H, W)
    else:
        conf_mask = np.ones((S, H, W), dtype=bool)

    for view_idx in range(S):
        # Load source image for colors
        src_path = image_paths[view_idx] if view_idx < len(image_paths) else None
        if src_path and os.path.exists(src_path):
            img = np.array(Image.open(src_path).convert('RGB').resize((W, H), Image.LANCZOS))
        else:
            img = np.full((H, W, 3), 128, dtype=np.uint8)

        pts_view  = pts3d[view_idx]       # (H, W, 3)
        mask_view = conf_mask[view_idx]   # (H, W)

        pts_flat  = pts_view[mask_view]   # (N, 3)
        col_flat  = img[mask_view]        # (N, 3)

        for pt, col in zip(pts_flat, col_flat):
            if np.all(np.isfinite(pt)):
                points3D.append({
                    'xyz'         : pt.astype(np.float64),
                    'rgb'         : col.astype(np.uint8),
                    'error'       : 0.0,
                    'image_ids'   : np.array([], dtype=np.int32),
                    'point2D_idxs': np.array([], dtype=np.int32),
                })

    if verbose:
        print(f"  Extracted {len(points3D):,} 3D points")
        print(f"  Built {len(cameras)} cameras, {len(images_data)} image entries")

    return cameras, images_data, points3D


def save_images_to_colmap_dir(image_paths, images_dir, verbose=True):
    """Copy preprocessed images to the COLMAP images/ directory."""
    import shutil
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    for idx, src in enumerate(image_paths):
        dst = images_dir / Path(src).name
        shutil.copy2(src, dst)

    if verbose:
        print(f"  Copied {len(image_paths)} images to {images_dir}")


def run_global_bundle_adjustment(sparse_dir):

    print("\n" + "=" * 50)
    print("RUNNING GLOBAL BUNDLE ADJUSTMENT")
    print("=" * 50)

    if not (sparse_dir / "cameras.bin").exists():
        print("Error: Model files not found for BA.")
        return

    reconstruction = pycolmap.Reconstruction(sparse_dir)

    options = pycolmap.BundleAdjustmentOptions()
    options.solver_options.num_threads = 8
    options.solver_options.max_num_iterations = 100

    print(f"Initial Mean Reprojection Error: "
          f"{reconstruction.compute_mean_reprojection_error():.4f} pixels")

    pycolmap.bundle_adjustment(reconstruction, options)

    print(f"Final Mean Reprojection Error: "
          f"{reconstruction.compute_mean_reprojection_error():.4f} pixels")

    reconstruction.write(sparse_dir)
    print("✓ Model successfully refined and saved.")
    print("=" * 50 + "\n")


def convert_vggt_to_colmap(predictions, image_paths, output_dir,
                            conf_threshold_pct=20.0, verbose=True,
                            run_ba=True):
    """
    Convert VGGT predictions to a COLMAP sparse reconstruction on disk.
    Optionally runs global bundle adjustment after writing the initial model.

    Directory structure created:
        output_dir/
        ├── images/          (copied source images)
        └── sparse/0/
            ├── cameras.bin
            ├── images.bin
            └── points3D.bin

    Args:
        predictions        : dict from run_vggt()
        image_paths        : list of image file paths used for inference
        output_dir         : root output directory
        conf_threshold_pct : drop lowest N% confidence points
        verbose            : print progress
        run_ba             : if True, run global bundle adjustment after writing

    Returns:
        Path to sparse/0 directory
    """
    output_dir = Path(output_dir)
    sparse_dir = output_dir / "sparse" / "0"
    images_dir = output_dir / "images"

    sparse_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n" + "="*70)
        print("Converting VGGT predictions → COLMAP format")
        print("="*70)
        print(f"Output: {output_dir}")

    cameras, images_data, points3D = extract_vggt_scene_data(
        predictions, image_paths, conf_threshold_pct, verbose
    )

    save_images_to_colmap_dir(image_paths, images_dir, verbose)

    if verbose:
        print("\nWriting COLMAP binary files...")

    write_cameras_binary(cameras, sparse_dir / "cameras.bin")
    if verbose:
        print(f"  ✓ cameras.bin  ({len(cameras)} cameras)")

    write_images_binary(images_data, sparse_dir / "images.bin")
    if verbose:
        print(f"  ✓ images.bin   ({len(images_data)} images)")

    write_points3d_binary(points3D, sparse_dir / "points3D.bin")
    if verbose:
        print(f"  ✓ points3D.bin ({len(points3D):,} points)")

    if verbose:
        print("\n" + "="*70)
        print("✓ COLMAP conversion complete!")
        print("="*70)

    # ── Bundle Adjustment ─────────────────────────────────────────────────
    if run_ba:
        run_global_bundle_adjustment(sparse_dir)

    return sparse_dir
