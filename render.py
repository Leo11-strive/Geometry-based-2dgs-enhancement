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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from typing import Optional
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from utils.render_utils import generate_path, create_videos

import open3d as o3d

@torch.no_grad()
def report_metrics(tag: str, cams, rgbmaps, depthmaps=None, out_dir: Optional[str] = None):
    if cams is None or len(cams) == 0:
        return
    if rgbmaps is None or len(rgbmaps) != len(cams):
        print(f"[Warn] Metrics skipped for {tag}: mismatched cams/rgbmaps ({len(cams)} vs {0 if rgbmaps is None else len(rgbmaps)})")
        return

    psnrs = []
    ssims = []
    lpipss = []
    depth_l1s = []
    depth_rmses = []

    for i, cam in enumerate(tqdm(cams, desc=f"Computing metrics ({tag})")):
        # RGB Metrics
        gt = torch.clamp(cam.original_image[0:3].cuda().float(), 0.0, 1.0)
        pred = torch.clamp(rgbmaps[i].cuda().float(), 0.0, 1.0)
        
        if gt.shape != pred.shape:
            print(f"[Warn] Skip view {i}: shape mismatch gt={tuple(gt.shape)} pred={tuple(pred.shape)}")
            continue
            
        psnrs.append(float(psnr(pred.unsqueeze(0), gt.unsqueeze(0)).item()))
        ssims.append(float(ssim(pred.unsqueeze(0), gt.unsqueeze(0)).item()))
        lpipss.append(float(lpips(pred.unsqueeze(0), gt.unsqueeze(0), net_type='vgg').item()))

        # Depth Metrics
        if depthmaps is not None and getattr(cam, "gt_depth", None) is not None:
            # depthmaps[i] is typically (1, H, W)
            pred_depth = depthmaps[i].cuda().squeeze()
            gt_depth = cam.gt_depth.cuda().squeeze()
            
            if pred_depth.shape == gt_depth.shape:
                mask = (gt_depth > 0) & (pred_depth > 0)
                if mask.any():
                    diff = torch.abs(pred_depth[mask] - gt_depth[mask])
                    depth_l1s.append(float(diff.mean().item()))
                    depth_rmses.append(float(torch.sqrt((diff**2).mean()).item()))

    if not psnrs:
        print(f"[Warn] Metrics skipped for {tag}: no valid views")
        return

    mean_psnr = sum(psnrs) / len(psnrs)
    mean_ssim = sum(ssims) / len(ssims)
    mean_lpips = sum(lpipss) / len(lpipss)
    
    print(f"[METRICS] {tag}: PSNR={mean_psnr:.3f} dB, SSIM={mean_ssim:.4f}, LPIPS={mean_lpips:.4f}")
    
    if depth_l1s:
        mean_depth_l1 = sum(depth_l1s) / len(depth_l1s)
        mean_depth_rmse = sum(depth_rmses) / len(depth_rmses)
        print(f"[METRICS] {tag}: Depth L1={mean_depth_l1:.4f}, Depth RMSE={mean_depth_rmse:.4f} (views={len(depth_l1s)})")
    else:
        mean_depth_l1 = -1.0
        mean_depth_rmse = -1.0

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
            f.write(f"mean_psnr {mean_psnr:.6f}\n")
            f.write(f"mean_ssim {mean_ssim:.6f}\n")
            f.write(f"mean_lpips {mean_lpips:.6f}\n")
            f.write(f"mean_depth_l1 {mean_depth_l1:.6f}\n")
            f.write(f"mean_depth_rmse {mean_depth_rmse:.6f}\n")
            f.write("\n# Per-view Metrics\n")
            f.write("View_Idx PSNR SSIM LPIPS DepthL1 DepthRMSE\n")
            for i in range(len(psnrs)):
                d_l1 = depth_l1s[i] if i < len(depth_l1s) else -1.0
                d_rmse = depth_rmses[i] if i < len(depth_rmses) else -1.0
                f.write(f"{i:05d} {psnrs[i]:.6f} {ssims[i]:.6f} {lpipss[i]:.6f} {d_l1:.6f} {d_rmse:.6f}\n")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)


    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    train_dir = os.path.join(args.model_path, 'train', "ours_{}".format(scene.loaded_iter))
    test_dir = os.path.join(args.model_path, 'test', "ours_{}".format(scene.loaded_iter))
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)    
    
    if not args.skip_train:
        print("export training images ...")
        os.makedirs(train_dir, exist_ok=True)
        gaussExtractor.reconstruction(scene.getTrainCameras())
        report_metrics("train", scene.getTrainCameras(), gaussExtractor.rgbmaps, gaussExtractor.depthmaps, out_dir=train_dir)
        gaussExtractor.export_image(train_dir)
        
    
    if (not args.skip_test) and (len(scene.getTestCameras()) > 0):
        print("export rendered testing images ...")
        os.makedirs(test_dir, exist_ok=True)
        gaussExtractor.reconstruction(scene.getTestCameras())
        report_metrics("test", scene.getTestCameras(), gaussExtractor.rgbmaps, gaussExtractor.depthmaps, out_dir=test_dir)
        gaussExtractor.export_image(test_dir)
    
    
    if args.render_path:
        print("render videos ...")
        traj_dir = os.path.join(args.model_path, 'traj', "ours_{}".format(scene.loaded_iter))
        os.makedirs(traj_dir, exist_ok=True)
        n_fames = 240
        cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_fames)
        gaussExtractor.reconstruction(cam_traj)
        gaussExtractor.export_image(traj_dir)
        create_videos(base_dir=traj_dir,
                    input_dir=traj_dir, 
                    out_name='render_traj', 
                    num_frames=n_fames)

    if not args.skip_mesh:
        print("export mesh ...")
        os.makedirs(train_dir, exist_ok=True)
        # set the active_sh to 0 to export only diffuse texture
        gaussExtractor.gaussians.active_sh_degree = 0
        gaussExtractor.reconstruction(scene.getTrainCameras())
        # extract the mesh and save
        if args.unbounded:
            name = 'fuse_unbounded.ply'
            mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
        else:
            name = 'fuse.ply'
            depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0  else args.depth_trunc
            voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
            sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
            mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
        
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
        print("mesh saved at {}".format(os.path.join(train_dir, name)))
        # post-process the mesh and save, saving the largest N clusters
        mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('.ply', '_post.ply')), mesh_post)
        print("mesh post processed saved at {}".format(os.path.join(train_dir, name.replace('.ply', '_post.ply'))))
