# 相对于原版 2DGS 的新增实现说明（含所有 Loss 的数学推导与最终表达式）
分工：我一个人完成
本文档阐述前仓库相对原版 2DGS 的新增/改写实现。  
## 更多更详细的内容请阅读`详细说明.md`。
---

## 0. 代码层面的差异清单

### 0.1 新增文件
- `utils/edge_aware_utils.py`：Sobel 梯度、DN-Splatter 风格的边缘感知权重、specular mask。
- `utils/smooth_utils.py`：kNN 图构建（sklearn/Open3D）+ 基于邻域的平滑正则（颜色/opacity）。
- `tools/generate_depth_anything.py`：用 Depth Anything 3 生成深度图（支持 transforms.json / COLMAP）。
- `scripts/colmap_birdhouse.sh`：封装 `convert.py` 的 COLMAP 预处理脚本（自动准备 `input/`）。
- `scripts/extract_suitcase_frames.sh`：从视频抽帧生成 `rgb/` 图像序列。

### 0.2 修改文件
- `train.py`：加入 **深度监督 loss**、**kNN 平滑 loss**、**specular 处理**、**stage 训练调度**、**novel-view 深度平滑正则**、更丰富的日志。
- `arguments/__init__.py`：加入大量训练超参（见 §6）。
- `utils/camera_utils.py` / `scene/cameras.py`：支持从 `<source_path>/<depth_dir>/*.npy` 加载并携带 `gt_depth`。
- `render.py`：在导出渲染结果时额外计算并写出 `PSNR/SSIM/LPIPS/Depth-L1/Depth-RMSE`。
- `gaussian_renderer/network_gui.py` / `view.py` / `train.py`：Viewer 端口绑定支持自动回退与重试（避免端口占用导致训练/查看失败）。
- `convert.py`：增强 COLMAP 兼容性（新旧 GPU flag 自动识别），并调小 BA tolerance。
- `.gitignore`：忽略 `/output`、`/data`、`/depth-anything-3` 等目录（工程习惯变化）。

---

## 1. 当前训练流程（相对原版的关键变化点）

原版 2DGS 训练主循环（`origin/main:train.py`）本质上是：
1) 随机采样训练视角 → render；  
2) `L_photo = (1-λ)·L1 + λ·(1-SSIM)`；  
3) 加上 `L_distortion`（>3000 iter 开启）与 `L_normal`（>7000 iter 开启）；  
4) densify/prune + reset opacity；  

当前版本在此基础上，主要新增/改写：

### 1.1 深度监督（Depth Supervision）
- 数据层：从 `ModelParams.depth_dir`（默认 `"depth"`）读取 `*.npy` 深度图并挂到 `Camera.gt_depth`（CPU）；训练时再搬到 GPU。  
  - 读取/缩放：`utils/camera_utils.py::_load_depth_map()`  
  - 存储：`scene/cameras.py::Camera.gt_depth`
- loss 层：`train.py` 增加 `L_depth`（raw/ndc 空间，L1/Huber，warmup+ramp+decay，edge-aware 权重，specular-aware 修正，near/far mask 等）。

### 1.2 kNN 图上的外观/不透明度平滑正则
- `utils/smooth_utils.py` 提供 kNN 构图与两个平滑 loss：  
  - `L_smooth_color`：邻域内 DC 颜色（RGB）一致性（加权 L2）。  
  - `L_smooth_opacity`：邻域内 opacity 一致性（加权 L1）。  
- `train.py` 在 densify 过程中标记 point-set 变动并触发重建 kNN（避免邻接关系过期）。

### 1.3 Specular/高光区域处理（用于更稳的监督）
- 从 GT RGB 估计 specular mask（阈值：亮度 `tV` + “近白”指标 `tS`），并用于：
  - **下调** photometric L1 在 specular 区域的权重（避免违反朗伯假设导致优化发散）。
  - 在开启 `depth_weight_mode=rgb_grad` 时，**补偿**高光区域的深度权重（防止边缘感知在高光附近“误判为边缘→权重过小”）。

### 1.4 Stage 化训练调度（几何→锁定→外观细化）
通过 `freeze_xyz_iter` / `appearance_only_iter` 将训练切成 A/B/C 三个阶段（详见 §5.6）：
- Stage A：正常训练 + densify（默认与原版一致）。
- Stage B：冻结 xyz（几何锁定），缩小 rot/scale/opacity 的 LR，继续学 features。
- Stage C：仅学 features（外观细化），并禁用 densify。

### 1.5 Novel-view 深度平滑正则（可选）
在 Stage B/C 期间周期性采样一个“邻近视角”，对渲染深度做 **edge-aware TV** 平滑，鼓励几何在未见视角下更平滑、更一致。

---

