#!/usr/bin/env python3
"""
使用 Depth Anything 3 为当前数据集批量生成深度图。
支持两种输入：
  1) NeRF/自定义 transforms.json：读取 frames 或 rgb.images（单模态/多模态）。
  2) COLMAP 数据集：若 <source_path>/<transforms> 不存在，则自动读取 <source_path>/sparse/0/*.bin
     并在 <source_path>/images（或 --image_dir 指定）中按 COLMAP 的 image name 找到图片。

对 RGB 图像运行深度估计，输出到 <source_path>/depth/<img_name>.npy（可选保存可视化 PNG）。

运行示例：
    python tools/generate_depth_anything.py \\
        --source_path data/birdhouse_use_priors_new_copy \\
        --model depth-anything/DA3NESTED-GIANT-LARGE \\
        --device cuda \\
        --process_res 504 \\
        --batch_size 16 \\
        --save_png
"""

import argparse
import json
import os
import sys
import struct
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

# 将 depth-anything-3 模块加入路径
ROOT = Path(__file__).resolve().parents[1]
DA3_PATH = ROOT / "depth-anything-3" / "src"
if str(DA3_PATH) not in sys.path:
    sys.path.insert(0, str(DA3_PATH))

import torch
from depth_anything_3.api import DepthAnything3


def load_frames_and_images(transforms_path: Path):
    with open(transforms_path, "r") as f:
        meta = json.load(f)

    images = []
    frames = []
    base = transforms_path.parent

    if "rgb" in meta and "images" in meta["rgb"]:
        frames = meta["rgb"]["images"]
        for frame in frames:
            fp = frame.get("file_path")
            if fp:
                images.append(base / fp)
    elif "frames" in meta:
        frames = meta["frames"]
        for frame in frames:
            fp = frame.get("file_path")
            if fp:
                images.append(base / fp)
    else:
        raise RuntimeError("transforms.json 未找到 rgb.images 或 frames 字段")

    if len(frames) != len(images):
        # frames 里可能有缺失 file_path 的条目；这里按 images 过滤 frames 保持对齐
        filtered_frames = []
        for frame in frames:
            fp = frame.get("file_path")
            if fp:
                filtered_frames.append(frame)
        frames = filtered_frames

    return meta, frames, images


def _auto_pose_type(meta: dict) -> str:
    if meta.get("_schema") == "colmap":
        return "w2c_colmap"
    # 标准 NeRF transforms.json 通常是 top-level "frames" 且 transform_matrix 为 c2w(OpenGL)。
    if "frames" in meta:
        return "c2w_opengl"
    return "w2c_colmap"


def _find_colmap_images_txt(dataset_root: Path) -> Optional[Path]:
    # keep a small, explicit list (no expensive recursion)
    candidates = [
        dataset_root / "colmap_workspace" / "model_txt" / "images.txt",
        dataset_root / "sparse" / "0" / "images.txt",
        dataset_root / "sparse" / "images.txt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = [float(x) for x in qvec]
    return np.array(
        [
            [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qw * qz, 2 * qz * qx + 2 * qw * qy],
            [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qw * qx],
            [2 * qz * qx - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx**2 - 2 * qy**2],
        ],
        dtype=np.float32,
    )


def _read_next_bytes(fid, num_bytes: int, fmt: str, endian: str = "<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian + fmt, data)


def _read_colmap_cameras_bin(cameras_bin: Path) -> dict[int, dict]:
    """
    Read COLMAP cameras.bin.
    Return: {camera_id: {"model": str, "width": int, "height": int, "params": np.ndarray}}
    """
    camera_model_id_to_name_and_num_params = {
        0: ("SIMPLE_PINHOLE", 3),
        1: ("PINHOLE", 4),
        2: ("SIMPLE_RADIAL", 4),
        3: ("RADIAL", 5),
        4: ("OPENCV", 8),
        5: ("OPENCV_FISHEYE", 8),
        6: ("FULL_OPENCV", 12),
        7: ("FOV", 5),
        8: ("SIMPLE_RADIAL_FISHEYE", 4),
        9: ("RADIAL_FISHEYE", 5),
        10: ("THIN_PRISM_FISHEYE", 12),
    }

    out: dict[int, dict] = {}
    with open(cameras_bin, "rb") as f:
        num_cameras = _read_next_bytes(f, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = _read_next_bytes(f, 24, "iiQQ")
            if model_id not in camera_model_id_to_name_and_num_params:
                raise RuntimeError(f"Unsupported COLMAP camera model id: {model_id}")
            model_name, num_params = camera_model_id_to_name_and_num_params[int(model_id)]
            params = _read_next_bytes(f, 8 * num_params, "d" * num_params)
            out[int(camera_id)] = {
                "model": model_name,
                "width": int(width),
                "height": int(height),
                "params": np.asarray(params, dtype=np.float32),
            }
    return out


def _read_colmap_images_bin(images_bin: Path) -> list[dict]:
    """
    Read COLMAP images.bin.
    Return list of dicts: {"image_id": int, "name": str, "camera_id": int, "qvec": np.ndarray, "tvec": np.ndarray}
    """
    out: list[dict] = []
    with open(images_bin, "rb") as f:
        num_images = _read_next_bytes(f, 8, "Q")[0]
        for _ in range(num_images):
            image_id = _read_next_bytes(f, 4, "i")[0]
            qvec = np.array(_read_next_bytes(f, 8 * 4, "dddd"), dtype=np.float32)
            tvec = np.array(_read_next_bytes(f, 8 * 3, "ddd"), dtype=np.float32)
            camera_id = _read_next_bytes(f, 4, "i")[0]

            name_bytes = bytearray()
            while True:
                c = f.read(1)
                if c == b"\x00" or c == b"":
                    break
                name_bytes.extend(c)
            name = name_bytes.decode("utf-8")

            num_points2D = _read_next_bytes(f, 8, "Q")[0]
            # each point2D: x(double), y(double), point3D_id(int64) -> 24 bytes
            f.seek(24 * int(num_points2D), os.SEEK_CUR)

            out.append(
                {
                    "image_id": int(image_id),
                    "name": name,
                    "camera_id": int(camera_id),
                    "qvec": qvec,
                    "tvec": tvec,
                }
            )
    return out


def _intrinsics_from_colmap_camera(cam: dict) -> Optional[np.ndarray]:
    model = cam["model"]
    w = cam["width"]
    h = cam["height"]
    params = cam["params"]

    if model == "SIMPLE_PINHOLE":
        f, cx, cy = [float(x) for x in params[:3]]
        fx, fy = f, f
    elif model == "PINHOLE":
        fx, fy, cx, cy = [float(x) for x in params[:4]]
    else:
        # DepthAnything3 只需要 K 矩阵用于尺度对齐，这里只实现常见无畸变针孔模型。
        # 若你确实有畸变模型，请先用 COLMAP undistort 得到 PINHOLE/SIMPLE_PINHOLE。
        return None

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    # basic sanity check: principal point should be inside image
    if not (0.0 <= cx <= float(w) and 0.0 <= cy <= float(h)):
        # don't hard fail, but alignment may be unstable
        pass
    return K


def load_frames_and_images_from_colmap(
    dataset_root: Path,
    *,
    colmap_model_rel: str = "sparse/0",
    image_dir_rel: Optional[str] = None,
):
    """
    Construct a "frames" list compatible with this script using COLMAP model (*.bin)
    and images in <dataset_root>/<image_dir_rel>.
    """
    model_dir = dataset_root / colmap_model_rel
    cameras_bin = model_dir / "cameras.bin"
    images_bin = model_dir / "images.bin"
    if not cameras_bin.exists() or not images_bin.exists():
        raise FileNotFoundError(f"未找到 COLMAP 模型文件: {cameras_bin} / {images_bin}")

    if image_dir_rel is None:
        for cand in ["images", "rgb", "input"]:
            if (dataset_root / cand).is_dir():
                image_dir_rel = cand
                break
    if image_dir_rel is None:
        raise FileNotFoundError("未找到图片目录（尝试 images/、rgb/、input/ 均不存在），请用 --image_dir 指定")

    image_dir = dataset_root / image_dir_rel
    cams = _read_colmap_cameras_bin(cameras_bin)
    imgs = _read_colmap_images_bin(images_bin)
    imgs = sorted(imgs, key=lambda x: x["name"])

    frames: list[dict] = []
    images: list[Path] = []
    for im in imgs:
        name = im["name"]
        img_path = image_dir / name
        if not img_path.exists():
            raise FileNotFoundError(f"COLMAP 图片不存在: {img_path} (from {images_bin})")

        cam = cams.get(im["camera_id"])
        if cam is None:
            raise RuntimeError(f"images.bin references missing camera_id={im['camera_id']}")

        K = _intrinsics_from_colmap_camera(cam)
        if K is None:
            raise RuntimeError(
                f"不支持的相机模型用于尺度对齐: {cam['model']}，"
                "请先用 COLMAP image_undistorter 得到 PINHOLE/SIMPLE_PINHOLE。"
            )

        R = _qvec2rotmat(im["qvec"])
        t = np.asarray(im["tvec"], dtype=np.float32)
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R
        w2c[:3, 3] = t

        frames.append(
            {
                "file_path": str(Path(image_dir_rel) / name),
                "transform_matrix": w2c.tolist(),  # w2c_colmap
                "fx": float(K[0, 0]),
                "fy": float(K[1, 1]),
                "cx": float(K[0, 2]),
                "cy": float(K[1, 2]),
            }
        )
        images.append(img_path)

    meta = {
        "_schema": "colmap",
        "frames": frames,
        "image_dir": str(image_dir_rel),
        "colmap_model": str(colmap_model_rel),
    }
    return meta, frames, images


def _parse_colmap_images_txt(images_txt: Path, *, max_images: int = 200) -> dict[str, np.ndarray]:
    """
    Parse COLMAP text model images.txt and return {image_name: w2c_4x4}.
    """
    out: dict[str, np.ndarray] = {}
    n = 0
    for line in images_txt.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        name = parts[9]
        qvec = np.array(list(map(float, parts[1:5])), dtype=np.float32)
        tvec = np.array(list(map(float, parts[5:8])), dtype=np.float32)
        R = _qvec2rotmat(qvec)
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R
        w2c[:3, 3] = tvec
        out[name] = w2c
        n += 1
        if n >= max_images:
            break
    return out


def _rot_angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """
    Rotation angle between Ra and Rb in radians.
    """
    R = Ra @ Rb.T
    tr = float(np.trace(R))
    c = (tr - 1.0) * 0.5
    c = max(-1.0, min(1.0, c))
    return float(np.arccos(c))


def _auto_pose_type_rgb_images(dataset_root: Path, frames: list[dict]) -> str:
    """
    For multimodal schema (rgb.images), guess pose semantic.
    Preference order:
      1) If COLMAP model exists, compare against COLMAP w2c by rotation error.
      2) If meta_data.json exists (MMS), assume c2w_opengl.
      3) Fallback to w2c_colmap (as used by data_zju).
    """
    images_txt = _find_colmap_images_txt(dataset_root)
    if images_txt is not None:
        try:
            colmap_w2c_by_name = _parse_colmap_images_txt(images_txt)
            # Evaluate on a few matched frames
            scores = {"w2c_colmap": [], "c2w_opengl": [], "c2w_opencv": []}
            for f in frames:
                fp = f.get("file_path")
                if not fp:
                    continue
                name = Path(fp).name
                w2c_ref = colmap_w2c_by_name.get(name)
                if w2c_ref is None:
                    continue
                tm = _as_4x4(np.array(f["transform_matrix"], dtype=np.float32))

                # Candidate 1: already w2c
                w2c_1 = tm
                # Candidate 2: OpenGL c2w -> (flip y/z) -> invert
                c2w_gl = tm.copy()
                c2w_gl[:3, 1:3] *= -1
                w2c_2 = np.linalg.inv(c2w_gl).astype(np.float32)
                # Candidate 3: OpenCV c2w -> invert
                w2c_3 = np.linalg.inv(tm).astype(np.float32)

                for key, w2c_cand in [("w2c_colmap", w2c_1), ("c2w_opengl", w2c_2), ("c2w_opencv", w2c_3)]:
                    scores[key].append(_rot_angle(w2c_cand[:3, :3], w2c_ref[:3, :3]))
                if len(scores["w2c_colmap"]) >= 10:
                    break

            best = None
            best_mean = None
            for k, errs in scores.items():
                if not errs:
                    continue
                m = float(np.mean(errs))
                if best_mean is None or m < best_mean:
                    best_mean = m
                    best = k
            if best is not None:
                return best
        except Exception:
            pass

    # MMS datasets generated from meta_data.json usually store camtoworld (OpenGL)
    if (dataset_root / "meta_data.json").exists():
        return "c2w_opengl"
    return "w2c_colmap"


def _as_4x4(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.shape == (4, 4):
        return mat
    if mat.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :4] = mat
        return out
    raise ValueError(f"Unsupported pose shape: {mat.shape}")


def build_camera_params(meta: dict, frames: list[dict], pose_type: str):
    # 全局内参（若没有 per-frame 内参，则复用全局）
    global_intr = None
    if "rgb" in meta and isinstance(meta["rgb"], dict):
        rgb = meta["rgb"]
        if all(k in rgb for k in ["fx", "fy", "cx", "cy"]):
            global_intr = np.array(
                [[rgb["fx"], 0, rgb["cx"]], [0, rgb["fy"], rgb["cy"]], [0, 0, 1.0]],
                dtype=np.float32,
            )
        elif all(k in rgb for k in ["fl_x", "fl_y", "cx", "cy"]):
            global_intr = np.array(
                [[rgb["fl_x"], 0, rgb["cx"]], [0, rgb["fl_y"], rgb["cy"]], [0, 0, 1.0]],
                dtype=np.float32,
            )
    elif all(k in meta for k in ["fl_x", "fl_y", "cx", "cy"]):
        global_intr = np.array(
            [[meta["fl_x"], 0, meta["cx"]], [0, meta["fl_y"], meta["cy"]], [0, 0, 1.0]],
            dtype=np.float32,
        )

    extrinsics_list: list[np.ndarray] = []
    intrinsics_list: list[np.ndarray] = []

    for frame in frames:
        tm = _as_4x4(np.array(frame["transform_matrix"], dtype=np.float32))
        if pose_type == "w2c_colmap":
            w2c = tm
        elif pose_type == "c2w_opengl":
            c2w = tm.copy()
            c2w[:3, 1:3] *= -1  # OpenGL/Blender -> COLMAP(OpenCV)
            w2c = np.linalg.inv(c2w).astype(np.float32)
        elif pose_type == "c2w_opencv":
            w2c = np.linalg.inv(tm).astype(np.float32)
        else:
            raise ValueError(f"Unknown pose_type: {pose_type}")
        extrinsics_list.append(w2c)

        if all(k in frame for k in ["fx", "fy", "cx", "cy"]):
            intrinsics_list.append(
                np.array(
                    [[frame["fx"], 0, frame["cx"]], [0, frame["fy"], frame["cy"]], [0, 0, 1.0]],
                    dtype=np.float32,
                )
            )
        elif all(k in frame for k in ["fl_x", "fl_y", "cx", "cy"]):
            intrinsics_list.append(
                np.array(
                    [[frame["fl_x"], 0, frame["cx"]], [0, frame["fl_y"], frame["cy"]], [0, 0, 1.0]],
                    dtype=np.float32,
                )
            )
        elif global_intr is not None:
            intrinsics_list.append(global_intr)

    if len(intrinsics_list) != len(extrinsics_list):
        intrinsics_list = []

    return extrinsics_list, intrinsics_list


def save_depth(depth: np.ndarray, out_npy: Path, save_png: bool):
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, depth.astype(np.float32))

    if save_png:
        # 简单 1%–99% 截断后归一化，方便快速查看
        d = depth.copy()
        finite = np.isfinite(d)
        if finite.any():
            lo, hi = np.percentile(d[finite], [1, 99])
            d = np.clip((d - lo) / (hi - lo + 1e-6), 0, 1)
        else:
            d = np.zeros_like(d)
        img = (d * 255).astype(np.uint8)
        Image.fromarray(img).save(out_npy.with_suffix(".png"))


def main():
    parser = argparse.ArgumentParser(description="Batch depth prediction with Depth Anything 3")
    parser.add_argument("--source_path", required=True, help="数据集根目录（支持 transforms.json 或 COLMAP sparse/0/*.bin）")
    parser.add_argument("--transforms", default="transforms.json", help="transforms 文件名，相对 source_path")
    parser.add_argument("--output_dir", default=None, help="输出深度目录，默认 <source_path>/depth")
    parser.add_argument("--image_dir", default=None, help="图片目录（相对 source_path），COLMAP 模式默认自动选择 images/rgb/input")
    parser.add_argument("--colmap_model", default="sparse/0", help="COLMAP 模型目录（相对 source_path），默认 sparse/0")
    parser.add_argument("--model", default="depth-anything/DA3NESTED-GIANT-LARGE", help="模型名称或本地权重目录")
    parser.add_argument("--device", default="cuda", help="设备 cuda/cpu")
    parser.add_argument("--process_res", type=int, default=504, help="推理分辨率")
    parser.add_argument("--process_res_method", default="upper_bound_resize", help="分辨率调整方法")
    parser.add_argument("--batch_size", type=int, default=16, help="批处理大小（>=3 才能做外参尺度对齐）")
    parser.add_argument(
        "--pose_type",
        default="auto",
        choices=["auto", "w2c_colmap", "c2w_opengl", "c2w_opencv"],
        help="transforms.json 中 transform_matrix 的语义：COLMAP(w2c) 或 NeRF(OpenGL c2w)",
    )
    parser.add_argument("--save_png", action="store_true", help="同时保存归一化可视化 PNG")
    parser.add_argument("--skip_existing", action="store_true", help="若目标 .npy 已存在则跳过")
    args = parser.parse_args()

    src = Path(args.source_path).resolve()
    transforms_path = src / args.transforms

    output_dir = Path(args.output_dir) if args.output_dir else (src / "depth")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 使用模型: {args.model}")
    model = DepthAnything3.from_pretrained(args.model)
    model = model.to(args.device)

    loaded_from = "transforms"
    if transforms_path.exists():
        meta, frames, images = load_frames_and_images(transforms_path)
    else:
        loaded_from = "colmap"
        meta, frames, images = load_frames_and_images_from_colmap(
            src,
            colmap_model_rel=args.colmap_model,
            image_dir_rel=args.image_dir,
        )
    print(f"[INFO] 输入类型: {loaded_from}")
    print(f"[INFO] 待处理 RGB 图像数: {len(images)}")

    # 读取相机外参（如有），用于 Depth Anything 进行尺度对齐
    extrinsics_list = []
    intrinsics_list = []
    try:
        if loaded_from == "colmap":
            # Our generated frames store w2c matrices.
            pose_type = "w2c_colmap" if args.pose_type == "auto" else args.pose_type
        else:
            if args.pose_type == "auto":
                pose_type = _auto_pose_type(meta)
                if "rgb" in meta and isinstance(meta["rgb"], dict) and "images" in meta["rgb"]:
                    pose_type = _auto_pose_type_rgb_images(src, frames)
            else:
                pose_type = args.pose_type
        print(f"[INFO] pose_type={pose_type} (args.pose_type={args.pose_type})")
        extrinsics_list, intrinsics_list = build_camera_params(meta, frames, pose_type)
    except Exception as e:
        print(f"[Warn] 读取相机外参/内参失败，深度将无法做尺度对齐: {e}")
        extrinsics_list = []
        intrinsics_list = []

    have_poses = bool(extrinsics_list) and bool(intrinsics_list) and (len(extrinsics_list) == len(intrinsics_list))
    min_views_for_pose_align = 3
    if have_poses and len(images) < min_views_for_pose_align:
        print(f"[Warn] 视角数={len(images)} < {min_views_for_pose_align}，将禁用外参尺度对齐并输出相对深度")
        have_poses = False

    def outputs_exist(image_index: int) -> bool:
        out_npy = output_dir / f"{Path(images[image_index]).stem}.npy"
        if not out_npy.exists():
            return False
        if args.save_png:
            out_png = out_npy.with_suffix(".png")
            return out_png.exists()
        return True

    idx = 0
    while idx < len(images):
        save_start = idx
        save_end = min(len(images), save_start + max(1, args.batch_size))

        # 推理窗口：如果要做尺度对齐，至少需要多个视角，否则 Umeyama 会退化报错
        infer_start, infer_end = save_start, save_end
        if have_poses and (infer_end - infer_start) < min_views_for_pose_align:
            if infer_start == 0:
                infer_end = min(len(images), infer_start + min_views_for_pose_align)
            else:
                infer_start = max(0, infer_end - min_views_for_pose_align)

        # 如果全都已存在且 skip_existing，则跳过整个 save 段
        if args.skip_existing:
            all_exist = True
            for j in range(save_start, save_end):
                if not outputs_exist(j):
                    all_exist = False
                    break
            if all_exist:
                idx = save_end
                continue

        pil_imgs = [Image.open(images[j]).convert("RGB") for j in range(infer_start, infer_end)]

        ex_t = None
        in_t = None
        if have_poses:
            ex_t = np.stack(extrinsics_list[infer_start:infer_end], axis=0)
            in_t = np.stack(intrinsics_list[infer_start:infer_end], axis=0)

        try:
            prediction = model.inference(
                pil_imgs,
                process_res=args.process_res,
                process_res_method=args.process_res_method,
                align_to_input_ext_scale=have_poses,
                infer_gs=False,
                extrinsics=ex_t,
                intrinsics=in_t,
            )
        except Exception as e:
            print(f"[Warn] inference failed at idx={infer_start}:{infer_end}, fallback to no-pose: {e}")
            prediction = model.inference(
                pil_imgs,
                process_res=args.process_res,
                process_res_method=args.process_res_method,
                align_to_input_ext_scale=False,
                infer_gs=False,
                extrinsics=None,
                intrinsics=None,
            )

        # 保存 save 段（可能是推理窗口的子集）
        offset = save_start - infer_start
        for j in range(save_start, save_end):
            img_path = Path(images[j]).resolve()
            name = img_path.stem
            out_npy = output_dir / f"{name}.npy"
            if args.skip_existing and outputs_exist(j):
                continue
            depth = prediction.depth[offset + (j - save_start)]
            save_depth(depth, out_npy, args.save_png)
            print(f"[{j+1}/{len(images)}] 保存深度 {out_npy}")

        idx = save_end


if __name__ == "__main__":
    main()
