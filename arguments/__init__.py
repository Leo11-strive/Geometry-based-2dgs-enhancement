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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.depth_dir = "depth"
        self.data_device = "cuda"
        self.eval = False
        self.render_items = ['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature']
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 0.0
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_dist = 0.0
        self.lambda_normal = 0.05
        self.lambda_depth = 0.0
        # Appearance / opacity smoothing (kNN graph regularization)
        self.lambda_smooth = 0.0
        self.lambda_smooth_opacity = 0.0
        self.smooth_knn_k = 16
        self.smooth_knn_interval = 500
        self.smooth_dist_sigma = 0.05  # <=0 enables auto (2 * median kNN distance)
        self.smooth_dist_sigma_sample = 200000  # sample size for median kNN distance (0 => full)
        self.smooth_normal_sigma = 0.3
        self.smooth_quat_order = "wxyz"  # wxyz | xyzw
        self.smooth_chunk_size = 65536

        # Stage schedule: geometry -> lock -> appearance refine
        # Stage A: iter < freeze_xyz_iter : normal training
        # Stage B: freeze_xyz_iter <= iter < appearance_only_iter : freeze xyz, scale rot/scale/opacity LR
        # Stage C: iter >= appearance_only_iter : freeze xyz/rot/scale/opacity, train features only (and disable densify)
        self.freeze_xyz_iter = -1            # <0 disables stage scheduling
        self.appearance_only_iter = -1       # <0 => no Stage C
        self.stageB_lr_scale_rotation = 0.1
        self.stageB_lr_scale_scaling = 0.1
        self.stageB_lr_scale_opacity = 0.1
        # Depth supervision options
        self.depth_loss_space = "raw"   # raw | ndc
        self.depth_near = 0.2
        self.depth_far = 1000.0
        self.depth_loss_type = "l1"     # l1 | huber
        self.depth_huber_beta = 0.1
        self.depth_warmup = 1000
        self.depth_ramp = 2000
        # Optional: decay depth/normal constraints (from large -> small)
        # eff = base * ramp_up(iter) * decay(iter)
        # decay is disabled when *_decay_start < 0 or *_decay_end <= *_decay_start
        self.depth_decay_start = -1
        self.depth_decay_end = -1
        self.depth_final_scale = 0.0  # multiplier at depth_decay_end (e.g., 0 to turn off)
        self.normal_warmup = 7000     # matches previous hard-coded behavior
        self.normal_ramp = 0          # 0 => step-on at normal_warmup
        self.normal_decay_start = -1
        self.normal_decay_end = -1
        self.normal_final_scale = 0.0
        # Edge-aware depth weighting (DN-Splatter-style): stronger in low-texture regions
        self.depth_weight_mode = "none"   # none | rgb_grad
        self.depth_grad_alpha = 10.0
        self.depth_grad_gray = True
        self.depth_grad_norm = "mean"     # none | mean | max
        self.depth_weight_min = 0.05
        self.depth_weight_max = 1.0
        # Specular-aware weighting (mitigate edge-aware underweighting near highlights)
        # Spec mask from GT RGB:
        #   V = max(R,G,B)
        #   S = (V - min(R,G,B)) / (V + eps)   (small => white/bright)
        #   spec = (V > tV) & (S < tS)
        self.spec_enable = False
        self.spec_tV = 0.92
        self.spec_tS = 0.15
        # Depth weight adjustment in spec regions (only applied in Stage A/B)
        self.depth_spec_mode = "mul"  # mul | clamp
        self.depth_spec_beta = 3.0
        self.depth_spec_min = 0.5
        # Safety valve: reduce spec boost when depth conflicts strongly
        self.depth_conf_tau = 0.2
        self.depth_conf_min_scale = 0.2
        # Photometric downweight in spec regions (only applied in Stage A/B)
        self.rgb_spec_gamma = 0.9
        self.rgb_spec_gamma_decay_start = -1
        self.rgb_spec_gamma_decay_end = -1
        self.rgb_spec_gamma_final_scale = 0.0
        self.opacity_cull = 0.05

        # Optional novel-view regularizer (off by default): edge-aware smoothness on rendered depth
        self.lambda_novel_depth_smooth = 0.0
        self.novel_interval = 200
        self.novel_resolution = 256
        self.novel_rot_deg = 2.0
        self.novel_trans = 0.01
        self.novel_rgb_alpha = 10.0
        self.novel_alpha_thresh = 0.5

        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
