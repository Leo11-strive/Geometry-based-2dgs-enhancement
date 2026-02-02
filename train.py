#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import math
from random import randint
import torch.nn.functional as F
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from utils.sh_utils import SH2RGB
from utils.edge_aware_utils import rgb_grad_weight, specular_mask
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.smooth_utils import build_knn_graph, knn_smooth_losses, quat_to_normal
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def _depth_z_to_ndc(depth_z: torch.Tensor, znear: float, zfar: float) -> torch.Tensor:
    """
    Convert view-space z depth to NDC z in [-1, 1] using the same (0..1) mapping
    implied by utils/graphics_utils.getProjectionMatrix (then remapped to [-1,1]).

    For the projection matrix used in this repo:
      ndc01 = A + B / z, where
        A = zfar / (zfar - znear)
        B = -(zfar * znear) / (zfar - znear)
      z=znear -> 0, z=zfar -> 1
    """
    z = depth_z.clamp(min=1e-6)
    denom = (zfar - znear)
    if denom <= 0:
        raise ValueError(f"Invalid z range: near={znear}, far={zfar}")
    A = zfar / denom
    B = -(zfar * znear) / denom
    ndc01 = A + (B / z)  # do NOT clamp; mask invalid values instead
    return ndc01 * 2.0 - 1.0

def _linear_ramp(iteration: int, start: int, ramp: int) -> float:
    if iteration <= start:
        return 0.0
    if ramp <= 0:
        return 1.0
    t = (iteration - start) / float(ramp)
    return float(min(1.0, max(0.0, t)))

def _linear_decay(iteration: int, start: int, end: int, final_scale: float) -> float:
    """
    Returns a multiplier in [final_scale, 1] (assuming final_scale in [0,1]),
    linearly decaying from 1 at start to final_scale at end.
    Disabled if start < 0 or end <= start.
    """
    if start < 0 or end <= start:
        return 1.0
    if iteration <= start:
        return 1.0
    if iteration >= end:
        return float(final_scale)
    t = (iteration - start) / float(end - start)
    return float((1.0 - t) * 1.0 + t * float(final_scale))

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)

    smooth_enabled = (getattr(opt, "lambda_smooth", 0.0) > 0.0) or (getattr(opt, "lambda_smooth_opacity", 0.0) > 0.0)
    smooth_state = {
        "knn_idx_cpu": None,
        "n_points": -1,
        "last_build_iter": -1,
        "knn_d_med": None,
        "dist_sigma_eff": None,
        "dirty": False,  # set True when densify/prune modifies the point set (even if N stays the same)
    }
    if smooth_enabled:
        print(f"[Smooth] Smoothing enabled (lambda_smooth={getattr(opt, 'lambda_smooth', 0.0)}, lambda_smooth_opacity={getattr(opt, 'lambda_smooth_opacity', 0.0)}).")
        print(f"[Smooth] smooth_knn_k={getattr(opt, 'smooth_knn_k', 16)}, smooth_knn_interval={getattr(opt, 'smooth_knn_interval', 500)}.")
        print(f"[Smooth] smooth_dist_sigma={getattr(opt, 'smooth_dist_sigma', 0.05)}, smooth_normal_sigma={getattr(opt, 'smooth_normal_sigma', 0.3)}.")
        print(f"[Smooth] smooth_dist_sigma_sample={getattr(opt, 'smooth_dist_sigma_sample', 0)}, smooth_quat_order={getattr(opt, 'smooth_quat_order', 'wxyz')}.")
        print(f"[Smooth] smooth_chunk_size={getattr(opt, 'smooth_chunk_size', 65536)}.")

    if opt.lambda_depth > 0.0:
        train_cams = scene.getTrainCameras()
        test_cams = scene.getTestCameras()
        n_train_depth = sum(1 for c in train_cams if getattr(c, "gt_depth", None) is not None)
        n_test_depth = sum(1 for c in test_cams if getattr(c, "gt_depth", None) is not None)
        print(f"[Depth] Depth supervision enabled (lambda_depth={opt.lambda_depth}).")
        print(f"[Depth] depth_loss_space={getattr(opt, 'depth_loss_space', 'raw')}, depth_loss_type={getattr(opt, 'depth_loss_type', 'l1')}.")
        print(f"[Depth] depth_near={getattr(opt, 'depth_near', 0.2)}, depth_far={getattr(opt, 'depth_far', 1000.0)}.")
        print(f"[Depth] depth_warmup={getattr(opt, 'depth_warmup', 1000)}, depth_ramp={getattr(opt, 'depth_ramp', 2000)}.")
        print(f"[Depth] depth_weight_mode={getattr(opt, 'depth_weight_mode', 'none')} (alpha={getattr(opt, 'depth_grad_alpha', 10.0)}, norm={getattr(opt, 'depth_grad_norm', 'mean')}).")
        print(f"[Depth] Train depth maps: {n_train_depth}/{len(train_cams)}")
        print(f"[Depth] Test depth maps: {n_test_depth}/{len(test_cams)}")

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    base_group_lrs = {g.get("name", f"g{idx}"): float(g["lr"]) for idx, g in enumerate(gaussians.optimizer.param_groups)}

    def _stage_name(iteration: int) -> str:
        freeze_xyz_iter = int(getattr(opt, "freeze_xyz_iter", -1))
        appearance_only_iter = int(getattr(opt, "appearance_only_iter", -1))
        if freeze_xyz_iter < 0:
            return "A"
        if iteration < freeze_xyz_iter:
            return "A"
        if appearance_only_iter >= 0 and iteration >= appearance_only_iter:
            return "C"
        return "B"

    def _apply_stage_lrs(iteration: int) -> str:
        stage = _stage_name(iteration)
        if stage == "A":
            # Keep normal training rates (xyz is already scheduled by update_learning_rate).
            for g in gaussians.optimizer.param_groups:
                name = g.get("name", "")
                if name != "xyz" and name in base_group_lrs:
                    g["lr"] = base_group_lrs[name]
            return stage

        if stage == "B":
            # Freeze xyz; lightly fine-tune shape/orientation; keep features learning.
            s_rot = float(getattr(opt, "stageB_lr_scale_rotation", 0.1))
            s_scl = float(getattr(opt, "stageB_lr_scale_scaling", 0.1))
            s_op = float(getattr(opt, "stageB_lr_scale_opacity", 0.1))
            for g in gaussians.optimizer.param_groups:
                name = g.get("name", "")
                if name == "xyz":
                    g["lr"] = 0.0
                elif name == "rotation" and name in base_group_lrs:
                    g["lr"] = base_group_lrs[name] * s_rot
                elif name == "scaling" and name in base_group_lrs:
                    g["lr"] = base_group_lrs[name] * s_scl
                elif name == "opacity" and name in base_group_lrs:
                    g["lr"] = base_group_lrs[name] * s_op
                elif name in base_group_lrs:
                    g["lr"] = base_group_lrs[name]
            return stage

        # stage == "C"
        for g in gaussians.optimizer.param_groups:
            name = g.get("name", "")
            if name in ("xyz", "rotation", "scaling", "opacity"):
                g["lr"] = 0.0
            elif name in base_group_lrs:
                g["lr"] = base_group_lrs[name]
        return stage

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_depth_for_log = 0.0
    ema_depth_w_for_log = 0.0
    ema_novel_depth_smooth_for_log = 0.0
    ema_smooth_for_log = 0.0
    ema_smooth_op_for_log = 0.0
    ema_smooth_w_for_log = 0.0
    ema_smooth_op_w_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    depth_ndc_range_warned = False
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        stage = _apply_stage_lrs(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        # Photometric loss (optional specular downweight in Stage A/B)
        spec = None
        if bool(getattr(opt, "spec_enable", False)) and stage in ("A", "B"):
            spec = specular_mask(gt_image.detach(), tV=float(getattr(opt, "spec_tV", 0.92)), tS=float(getattr(opt, "spec_tS", 0.15)))

        rgb_gamma = float(getattr(opt, "rgb_spec_gamma", 0.0))
        g_decay_start = int(getattr(opt, "rgb_spec_gamma_decay_start", -1))
        g_decay_end = int(getattr(opt, "rgb_spec_gamma_decay_end", -1))
        g_final = float(getattr(opt, "rgb_spec_gamma_final_scale", 0.0))
        rgb_gamma *= _linear_decay(iteration, g_decay_start, g_decay_end, g_final)

        if spec is not None and rgb_gamma > 0.0:
            w_rgb = (1.0 - rgb_gamma * spec.float()).clamp(min=0.05, max=1.0)
            l1_pix = (image - gt_image).abs().mean(dim=0)  # (H,W)
            Ll1 = (w_rgb * l1_pix).sum() / (w_rgb.sum() + 1e-12)
        else:
            Ll1 = l1_loss(image, gt_image)

        # SSIM remains global (unweighted); specular downweight targets the L1 term.
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # regularization
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0
        # Normal constraint: optional warmup + optional decay (large -> small)
        n_warm = int(getattr(opt, "normal_warmup", 7000))
        n_ramp = int(getattr(opt, "normal_ramp", 0))
        n_decay_start = int(getattr(opt, "normal_decay_start", -1))
        n_decay_end = int(getattr(opt, "normal_decay_end", -1))
        n_final = float(getattr(opt, "normal_final_scale", 0.0))
        lambda_normal = float(opt.lambda_normal) * _linear_ramp(iteration, n_warm, n_ramp) * _linear_decay(iteration, n_decay_start, n_decay_end, n_final)

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        novel_depth_smooth_loss = torch.tensor(0.0, device="cuda")
        if float(getattr(opt, "lambda_novel_depth_smooth", 0.0)) > 0.0 and stage in ("B", "C"):
            interval = int(getattr(opt, "novel_interval", 200))
            if interval > 0 and (iteration % interval == 0):
                # Sample a nearby "novel" view by applying a small perturbation in camera local space.
                res = int(getattr(opt, "novel_resolution", 256))
                res = max(32, res)
                H0 = int(viewpoint_cam.image_height)
                W0 = int(viewpoint_cam.image_width)
                scale = float(res) / float(max(H0, W0))
                Hn = max(32, int(round(H0 * scale)))
                Wn = max(32, int(round(W0 * scale)))

                rot_deg = float(getattr(opt, "novel_rot_deg", 2.0))
                trans = float(getattr(opt, "novel_trans", 0.01))
                rx = (torch.rand((), device="cuda") * 2.0 - 1.0) * (rot_deg * math.pi / 180.0)
                ry = (torch.rand((), device="cuda") * 2.0 - 1.0) * (rot_deg * math.pi / 180.0)
                rz = (torch.rand((), device="cuda") * 2.0 - 1.0) * (rot_deg * math.pi / 180.0)
                t = (torch.rand((3,), device="cuda") * 2.0 - 1.0) * trans

                cx, sx = torch.cos(rx), torch.sin(rx)
                cy, sy = torch.cos(ry), torch.sin(ry)
                cz, sz = torch.cos(rz), torch.sin(rz)
                Rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], device="cuda")
                Ry = torch.tensor([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], device="cuda")
                Rz = torch.tensor([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], device="cuda")
                R = Rz @ Ry @ Rx

                # Build camera-local SE(3) in standard (row-major) form (translation in last column).
                L = torch.eye(4, device="cuda")
                L[:3, :3] = R
                L[:3, 3] = t

                # Matrices in this repo are stored transposed (M^T). Applying a camera-local transform L
                # corresponds to right-multiplying by L^{-T}.
                Delta = torch.inverse(L).transpose(0, 1)
                nv_world_view = viewpoint_cam.world_view_transform @ Delta
                nv_full_proj = nv_world_view @ viewpoint_cam.projection_matrix
                from scene.cameras import MiniCam
                nv_cam = MiniCam(Wn, Hn, viewpoint_cam.FoVy, viewpoint_cam.FoVx, viewpoint_cam.znear, viewpoint_cam.zfar, nv_world_view, nv_full_proj)

                nv_pkg = render(nv_cam, gaussians, pipe, background)
                nv_rgb = nv_pkg["render"].detach()
                nv_depth = nv_pkg["surf_depth"].squeeze()
                nv_alpha = nv_pkg["rend_alpha"].squeeze()

                # Use the novel camera's near/far to stay consistent with its projection.
                nv_d = _depth_z_to_ndc(nv_depth, float(nv_cam.znear), float(nv_cam.zfar))

                w = rgb_grad_weight(
                    nv_rgb,
                    alpha=float(getattr(opt, "novel_rgb_alpha", 10.0)),
                    gray=True,
                    norm="mean",
                    w_min=0.05,
                    w_max=1.0,
                )

                mask = nv_alpha > float(getattr(opt, "novel_alpha_thresh", 0.5))
                dx = (nv_d[:, 1:] - nv_d[:, :-1]).abs()
                dy = (nv_d[1:, :] - nv_d[:-1, :]).abs()
                wx = 0.5 * (w[:, 1:] + w[:, :-1])
                wy = 0.5 * (w[1:, :] + w[:-1, :])
                mx = mask[:, 1:] & mask[:, :-1]
                my = mask[1:, :] & mask[:-1, :]
                wx = wx * mx.float()
                wy = wy * my.float()
                denom = torch.clamp_min(wx.sum() + wy.sum(), 1e-12)
                novel_depth_smooth_loss = (wx * dx).sum() / denom + (wy * dy).sum() / denom

        smooth_loss = torch.tensor(0.0, device="cuda")
        smooth_op_loss = torch.tensor(0.0, device="cuda")
        if smooth_enabled and (getattr(opt, "smooth_knn_k", 0) > 0):
            n_points = int(gaussians.get_xyz.shape[0])
            interval = int(getattr(opt, "smooth_knn_interval", 500))
            need_build = (smooth_state["knn_idx_cpu"] is None) or (smooth_state["n_points"] != n_points) or bool(smooth_state.get("dirty", False))
            if (not need_build) and (interval > 0) and (iteration % interval == 0):
                need_build = True
            if need_build:
                k = int(getattr(opt, "smooth_knn_k", 16))
                knn_idx, d_med = build_knn_graph(
                    gaussians.get_xyz,
                    k=k,
                    dist_sigma_sample=int(getattr(opt, "smooth_dist_sigma_sample", 0)),
                )
                smooth_state["knn_idx_cpu"] = knn_idx
                smooth_state["n_points"] = n_points
                smooth_state["last_build_iter"] = iteration
                smooth_state["knn_d_med"] = d_med
                smooth_state["dirty"] = False

                # Auto-select dist sigma if requested (<=0)
                dist_sigma_opt = float(getattr(opt, "smooth_dist_sigma", 0.05))
                if dist_sigma_opt <= 0 and (d_med is not None) and d_med > 0:
                    smooth_state["dist_sigma_eff"] = 2.0 * float(d_med)
                else:
                    smooth_state["dist_sigma_eff"] = dist_sigma_opt

                quat_order = str(getattr(opt, "smooth_quat_order", "wxyz"))
                with torch.no_grad():
                    q = gaussians.get_rotation
                    q_norm = torch.linalg.norm(q, dim=-1)
                    q_norm_med = float(q_norm.median().item())
                    q_norm_mean = float(q_norm.mean().item())
                    nrm = quat_to_normal(q, order=quat_order)
                    nrm_finite = torch.isfinite(nrm).all(dim=-1)
                    nrm_finite_ratio = float(nrm_finite.float().mean().item())
                    nrm_norm_mean = float(torch.linalg.norm(nrm, dim=-1).mean().item())
                print(
                    f"[Smooth] Built kNN (N={n_points}, K={k}) "
                    f"d_med={d_med} dist_sigma={smooth_state['dist_sigma_eff']:.6g} "
                    f"quat_order={quat_order} q_norm_med={q_norm_med:.4f} q_norm_mean={q_norm_mean:.4f} "
                    f"normal_finite={nrm_finite_ratio:.3f} normal_norm_mean={nrm_norm_mean:.4f}"
                )

            # _features_dc is (N, 1, 3); squeeze the SH-coefficient axis to get (N, 3)
            rgb_dc = SH2RGB(gaussians._features_dc.squeeze(1))
            opacity = gaussians.get_opacity
            smooth_loss, smooth_op_loss = knn_smooth_losses(
                xyz=gaussians.get_xyz,
                quat_wxyz=gaussians.get_rotation,
                rgb_dc=rgb_dc,
                opacity=opacity,
                knn_idx_cpu=smooth_state["knn_idx_cpu"],
                dist_sigma=float(smooth_state["dist_sigma_eff"] if smooth_state["dist_sigma_eff"] is not None else getattr(opt, "smooth_dist_sigma", 0.05)),
                normal_sigma=float(getattr(opt, "smooth_normal_sigma", 0.3)),
                quat_order=str(getattr(opt, "smooth_quat_order", "wxyz")),
                chunk_size=int(getattr(opt, "smooth_chunk_size", 65536)),
            )

        depth_loss = torch.tensor(0.0, device="cuda")
        depth_w_mean = torch.tensor(0.0, device="cuda")
        lambda_depth_eff = 0.0
        if opt.lambda_depth > 0.0:
            warm = int(getattr(opt, "depth_warmup", 1000))
            ramp = int(getattr(opt, "depth_ramp", 2000))
            lambda_depth_eff = float(opt.lambda_depth) * _linear_ramp(iteration, warm, ramp)
            d_decay_start = int(getattr(opt, "depth_decay_start", -1))
            d_decay_end = int(getattr(opt, "depth_decay_end", -1))
            d_final = float(getattr(opt, "depth_final_scale", 0.0))
            lambda_depth_eff *= _linear_decay(iteration, d_decay_start, d_decay_end, d_final)

            gt_depth = getattr(viewpoint_cam, "gt_depth", None)
            if (gt_depth is not None) and (lambda_depth_eff > 0.0):
                gt_depth = gt_depth.cuda()
                pred_depth = render_pkg["surf_depth"]
                # Base validity mask
                valid = torch.isfinite(gt_depth) & torch.isfinite(pred_depth) & (gt_depth > 0) & (pred_depth > 0)
                # Optional foreground mask if provided (already on GPU)
                if viewpoint_cam.gt_alpha_mask is not None:
                    valid = valid & (viewpoint_cam.gt_alpha_mask > 0.5)

                # Near/far mask (more controllable than clamping, avoids zero gradients)
                near = float(getattr(opt, "depth_near", 0.2))
                far = float(getattr(opt, "depth_far", 1000.0))
                valid = valid & (pred_depth > near) & (pred_depth < far) & (gt_depth > near) & (gt_depth < far)

                if valid.any():
                    # Squeeze to (H,W) for stable broadcasting
                    pred_z = pred_depth.squeeze()
                    gt_z = gt_depth.squeeze()
                    valid2d = valid.squeeze()

                    if getattr(opt, "depth_loss_space", "raw") == "ndc":
                        if not depth_ndc_range_warned:
                            cam_near = float(getattr(viewpoint_cam, "znear", near))
                            cam_far = float(getattr(viewpoint_cam, "zfar", far))
                            if abs(cam_near - near) > 1e-6 or abs(cam_far - far) > 1e-6:
                                print(f"[Depth][Warn] depth_near/far ({near},{far}) != camera znear/zfar ({cam_near},{cam_far}). For NDC-consistency, consider matching them.")
                            depth_ndc_range_warned = True
                        pred_d = _depth_z_to_ndc(pred_z, near, far)
                        gt_d = _depth_z_to_ndc(gt_z, near, far)
                    else:
                        pred_d, gt_d = pred_z, gt_z

                    # Per-pixel depth loss (no reduction yet)
                    if getattr(opt, "depth_loss_type", "l1") == "huber":
                        loss_pix = F.huber_loss(
                            pred_d,
                            gt_d,
                            reduction="none",
                            delta=float(getattr(opt, "depth_huber_beta", 0.1)),
                        )
                    else:
                        loss_pix = (pred_d - gt_d).abs()

                    # Edge-aware weighting (stronger on low-texture regions)
                    if str(getattr(opt, "depth_weight_mode", "none")).lower() == "rgb_grad":
                        w = rgb_grad_weight(
                            gt_image.detach(),
                            alpha=float(getattr(opt, "depth_grad_alpha", 10.0)),
                            gray=bool(getattr(opt, "depth_grad_gray", True)),
                            norm=str(getattr(opt, "depth_grad_norm", "mean")),
                            w_min=float(getattr(opt, "depth_weight_min", 0.05)),
                            w_max=float(getattr(opt, "depth_weight_max", 1.0)),
                        )

                        # Specular-aware boost to counteract edge-aware underweighting near highlights.
                        if spec is not None:
                            mode = str(getattr(opt, "depth_spec_mode", "mul")).lower()
                            if mode == "mul":
                                beta = float(getattr(opt, "depth_spec_beta", 0.0))
                                w = w * (1.0 + beta * spec.float())
                            elif mode == "clamp":
                                w_spec_min = float(getattr(opt, "depth_spec_min", 0.5))
                                w = torch.where(spec, torch.maximum(w, w.new_tensor(w_spec_min)), w)
                            else:
                                raise ValueError(f"Unknown depth_spec_mode: {mode} (expected 'mul'|'clamp')")

                            # Safety valve: if depth disagreement is large, reduce the boost.
                            tau = float(getattr(opt, "depth_conf_tau", 0.0))
                            if tau > 0:
                                conf = (loss_pix < tau).float()
                                min_scale = float(getattr(opt, "depth_conf_min_scale", 0.2))
                                w = w * (min_scale + (1.0 - min_scale) * conf)

                        # Debug: Save weight map visualization occasionally
                        if iteration % 1000 == 1:
                            try:
                                from torchvision.utils import save_image
                                debug_dir = os.path.join(scene.model_path, "debug_weights")
                                os.makedirs(debug_dir, exist_ok=True)
                                w_vis = w.unsqueeze(0).repeat(3, 1, 1)  # (1,H,W) -> (3,H,W)
                                comp = torch.cat([gt_image.detach(), w_vis], dim=2) # Concat horizontally
                                save_path = os.path.join(debug_dir, f"iter_{iteration:05d}_{viewpoint_cam.image_name}_weight.png")
                                save_image(comp, save_path)
                                print(f"[Debug] Saved weight map to {save_path}")
                            except ImportError:
                                pass
                            except Exception as e:
                                print(f"[Debug] Failed to save weight map: {e}")

                        w = w * valid2d.float()
                        denom = torch.clamp_min(w.sum(), 1e-12)
                        depth_loss = (w * loss_pix)[valid2d].sum() / denom
                        depth_w_mean = w[valid2d].mean()
                    else:
                        depth_loss = loss_pix[valid2d].mean()

        lambda_smooth = float(getattr(opt, "lambda_smooth", 0.0))
        lambda_smooth_op = float(getattr(opt, "lambda_smooth_opacity", 0.0))
        smooth_weighted = lambda_smooth * smooth_loss
        smooth_op_weighted = lambda_smooth_op * smooth_op_loss

        total_loss = (
            loss
            + dist_loss
            + normal_loss
            + lambda_depth_eff * depth_loss
            + smooth_weighted
            + smooth_op_weighted
            + float(getattr(opt, "lambda_novel_depth_smooth", 0.0)) * novel_depth_smooth_loss
        )
        
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            ema_depth_for_log = 0.4 * depth_loss.item() + 0.6 * ema_depth_for_log
            ema_depth_w_for_log = 0.4 * depth_w_mean.item() + 0.6 * ema_depth_w_for_log
            ema_novel_depth_smooth_for_log = 0.4 * novel_depth_smooth_loss.item() + 0.6 * ema_novel_depth_smooth_for_log
            ema_smooth_for_log = 0.4 * smooth_loss.item() + 0.6 * ema_smooth_for_log
            ema_smooth_op_for_log = 0.4 * smooth_op_loss.item() + 0.6 * ema_smooth_op_for_log
            ema_smooth_w_for_log = 0.4 * smooth_weighted.item() + 0.6 * ema_smooth_w_for_log
            ema_smooth_op_w_for_log = 0.4 * smooth_op_weighted.item() + 0.6 * ema_smooth_op_w_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "depth": f"{ema_depth_for_log:.{5}f}",
                    "d_w": f"{ema_depth_w_for_log:.{3}f}",
                    "nv_dtv": f"{ema_novel_depth_smooth_for_log:.{4}f}",
                    "smooth": f"{ema_smooth_for_log:.{5}f}",
                    "s_op": f"{ema_smooth_op_for_log:.{5}f}",
                    "s*w": f"{ema_smooth_w_for_log:.2e}",
                    "op*w": f"{ema_smooth_op_w_for_log:.2e}",
                    "stage": stage,
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/depth_loss', ema_depth_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/depth_w_mean', ema_depth_w_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/novel_depth_smooth', ema_novel_depth_smooth_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/smooth_loss', ema_smooth_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/smooth_opacity_loss', ema_smooth_op_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/smooth_weighted', ema_smooth_w_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/smooth_opacity_weighted', ema_smooth_op_w_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # Densification (Stage A only; Stage B/C aim to lock geometry)
            if iteration < opt.densify_until_iter and stage == "A":
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                    if smooth_enabled:
                        # Point set topology may have changed even if the final N stays identical.
                        # Rebuild kNN next iteration to avoid stale/mismatched neighborhoods.
                        smooth_state["dirty"] = True
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--port_retries', type=int, default=100, help="If port is busy, try next ports up to this many times (then fall back to ephemeral).")
    parser.add_argument('--port_strict', action='store_true', default=False, help="Do not auto-select a free port when the requested port is in use.")
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    actual_port = network_gui.init(
        args.ip,
        args.port,
        allow_fallback=(not args.port_strict),
        port_retries=args.port_retries,
        fallback_to_ephemeral=True,
    )
    if actual_port != args.port:
        print(f"[GUI] Port {args.port} unavailable; switched to {args.ip}:{actual_port}")
    else:
        print(f"[GUI] Listening on {args.ip}:{actual_port}")
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")
