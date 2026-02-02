# 相对于原版 2DGS 的新增实现说明（含所有 Loss 的数学推导与最终表达式）
分工：我一个人完成
本文档阐述前仓库相对原版 2DGS 的新增/改写实现。  

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

## 2. 记号与渲染输出约定（Notation）

一次迭代随机采样到某个训练相机视角 \(v\)，渲染得到：

- 预测 RGB：\(\hat{\mathbf I}\in[0,1]^{3\times H\times W}\)（`render_pkg["render"]`）
- GT RGB：\(\mathbf I\in[0,1]^{3\times H\times W}\)（`viewpoint_cam.original_image`）
- 预测 alpha：\(\hat{\alpha}\in[0,1]^{H\times W}\)（`render_pkg["rend_alpha"]`）
- 预测 surf 深度：\(\hat{D}\in\mathbb{R}^{H\times W}\)（`render_pkg["surf_depth"]`）
- GT 深度：\(D\in\mathbb{R}^{H\times W}\)（`viewpoint_cam.gt_depth`，若存在）
- 渲染法线（未归一/带权）：\(\hat{\mathbf n}^{\,rend}\in\mathbb{R}^{3\times H\times W}\)（`render_pkg["rend_normal"]`）
- 由深度生成的 pseudo-surface 法线（未归一/带权）：\(\hat{\mathbf n}^{\,surf}\in\mathbb{R}^{3\times H\times W}\)（`render_pkg["surf_normal"]`）
- 深度 distortion map：\(\hat{R}\in\mathbb{R}^{H\times W}\)（`render_pkg["rend_dist"]`）

> 说明：`rend_normal/surf_normal` 在渲染器中都乘了 \(\hat{\alpha}\)（`surf_normal` 还显式 `detach` alpha），因此它们的点积天然会对前景高置信像素赋更大权重。

此外引入两类调度函数（与 `train.py` 一致）：

### 2.1 线性 ramp-up
给定起始 \(s\) 与 ramp 长度 \(r\)，定义
\[
\mathrm{ramp}(t;s,r)=
\begin{cases}
0, & t\le s\\
\min\left(1,\max\left(0,\frac{t-s}{r}\right)\right), & r>0\\
1, & r\le 0
\end{cases}
\]

### 2.2 线性 decay
给定衰减区间 \([a,b]\) 与末端倍率 \(c\in[0,1]\)，定义
\[
\mathrm{decay}(t;a,b,c)=
\begin{cases}
1, & a<0 \ \text{或}\ b\le a\\
1, & t\le a\\
c, & t\ge b\\
(1-\tau)+\tau c,\ \tau=\frac{t-a}{b-a}, & a<t<b
\end{cases}
\]

---

## 3. Loss 逐项数学推导（包含原版已有项 + 当前新增项）

本节给出当前训练中所有 loss 的数学表达式，并说明其在代码中的实现方式/开关条件。

### 3.1 Photometric loss：L1 + DSSIM（原版已有，但本仓库加入 specular 权重）

#### 3.1.1 基础 L1（像素级）
对每个像素 \(p=(u,v)\)，定义通道平均绝对误差：
\[
e_{p}=\frac{1}{3}\sum_{c\in\{r,g,b\}}\left|\hat{I}_{c,p}-I_{c,p}\right|
\]

原版 L1 为简单平均：
\[
L_{L1}^{\text{plain}}=\frac{1}{HW}\sum_{p} e_p
\]

#### 3.1.2 Specular 下调（新增）
当 `spec_enable=True` 且处于 Stage A/B 时，先由 GT RGB 构造 specular mask（§4.3），记为 \(s_p\in\{0,1\}\)。

定义（代码中有 clamp）：
\[
w^{rgb}_p=\mathrm{clip}\left(1-\gamma(t)\,s_p,\ 0.05,\ 1.0\right)
\]
其中 \(\gamma(t)=\gamma_0\cdot \mathrm{decay}(t;g_s,g_e,g_f)\) 对应参数：
`rgb_spec_gamma`, `rgb_spec_gamma_decay_start/end/final_scale`。

则加权 L1 为：
\[
L_{L1}^{\text{spec}}=\frac{\sum_{p} w^{rgb}_p\,e_p}{\sum_{p} w^{rgb}_p+\varepsilon}
\]
当 `spec_enable=False` 或 Stage=C 时退化回 \(L_{L1}^{\text{plain}}\)。

#### 3.1.3 DSSIM 融合（原版已有）
最终 photometric loss（与代码一致，仅对 L1 做 spec 加权，SSIM 不加权）：
\[
L_{\text{photo}}=(1-\lambda_{\text{dssim}})\,L_{L1}+\lambda_{\text{dssim}}\left(1-\mathrm{SSIM}(\hat{\mathbf I},\mathbf I)\right)
\]

> SSIM 的经典定义（窗口内）：
> \[
> \mathrm{SSIM}(x,y)=\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}
> \]
> 本仓库使用的 `utils/loss_utils.py::ssim` 实现与原版保持一致（只是在 `train.py` 里改变了 L1 的构造方式）。

---

### 3.2 Distortion regularization（原版已有，给出严格推导）

#### 3.2.1 像素内的“深度分布”与权重
对一个像素 \(p\)，渲染器沿着该像素的“可见 surfel 列表”累积得到一组贡献权重：
\[
w_i = \alpha_i\,T_i
\]
其中 \(\alpha_i\) 是第 \(i\) 个 surfel 在像素上的 alpha，\(T_i\) 是它之前的透射率（transmittance）。

同时将 surfel 的深度 \(z_i\) 映射到 \([0,1]\) 的“归一化深度”：
\[
m_i = \frac{z_{far}}{z_{far}-z_{near}}\left(1-\frac{z_{near}}{z_i}\right)
\]
该式与渲染子模块 `diff-surfel-rasterization` 中实现一致（见 `submodules/diff-surfel-rasterization/cuda_rasterizer/forward.cu`）。

#### 3.2.2 Distortion 的 pairwise 形式
当前渲染器输出的 distortion map \(\hat{R}_p\) 等价于以下 pairwise 二重和（**平方差**）：
\[
\hat{R}_p
=\sum_{i}\sum_{j<i} w_iw_j\,(m_i-m_j)^2
\]
它也等价于 \(\frac12\sum_{i\ne j}w_iw_j(m_i-m_j)^2\)。

进一步可化为“加权方差”形式（令 \(W=\sum_i w_i\)，\(\mu=\frac{1}{W}\sum_i w_im_i\)）：
\[
\sum_{i}\sum_{j<i} w_iw_j\,(m_i-m_j)^2 = W^2\cdot \mathrm{Var}_w(m)
\]
其中 \(\mathrm{Var}_w(m)=\frac{1}{W}\sum_i w_i(m_i-\mu)^2\)。

#### 3.2.3 为什么 forward.cu 的 streaming 更新是正确的（推导）
令在处理第 \(i\) 个 surfel 之前维护三项累积量：
\[
A_{i-1}=\sum_{j<i}w_j,\quad
M1_{i-1}=\sum_{j<i}w_jm_j,\quad
M2_{i-1}=\sum_{j<i}w_jm_j^2
\]
则第 \(i\) 项的 pairwise 累积贡献为：
\[
\sum_{j<i} w_iw_j(m_i-m_j)^2
=w_i\sum_{j<i} w_j(m_i^2+m_j^2-2m_im_j)
\]
将求和拆开：
\[
=w_i\left(m_i^2\sum_{j<i}w_j + \sum_{j<i}w_jm_j^2 - 2m_i\sum_{j<i}w_jm_j\right)
=w_i\left(m_i^2A_{i-1}+M2_{i-1}-2m_iM1_{i-1}\right)
\]
这正对应 `forward.cu` 中：
`distortion += (m*m*A + M2 - 2*m*M1) * w;` 的形式。

#### 3.2.4 Distortion loss（训练中使用）
训练里取 map 的像素均值：
\[
L_{\text{dist}}(t)=\lambda_{\text{dist}}(t)\cdot \frac{1}{HW}\sum_{p}\hat{R}_p
\]
其中原版硬开关为：`iteration > 3000` 才启用（当前代码仍保持）：
\[
\lambda_{\text{dist}}(t)=
\begin{cases}
0,& t\le 3000\\
\lambda_{\text{dist}},& t>3000
\end{cases}
\]

---

### 3.3 Normal consistency loss（原版已有，但新增了可配置 warmup/decay）

#### 3.3.1 形式
对每个像素 \(p\)，代码使用未归一（且带 alpha 权重）的法线向量点积：
\[
\ell^{n}_p = 1 - \left\langle \hat{\mathbf n}^{\,rend}_p,\ \hat{\mathbf n}^{\,surf}_p \right\rangle
\]
并对全图取均值：
\[
L_{\text{normal}}(t)=\lambda_{\text{normal}}(t)\cdot \frac{1}{HW}\sum_p \ell^n_p
\]

> 注：如果将二者归一化为单位向量，则 \(\ell^n_p\) 近似等价于 \(1-\cos\theta_p\)。  
> 由于实现中两者都与 \(\hat{\alpha}_p\) 成比例，因此该 loss 也会对高 alpha 的前景像素赋更大权重（对背景/空洞区域影响小）。

#### 3.3.2 权重调度（新增可配置）
原版固定：`iteration > 7000` 才启用 normal loss。当前版本改为：
\[
\lambda_{\text{normal}}(t)=\lambda_{\text{normal}}^{0}\cdot
\mathrm{ramp}(t;n_w,n_r)\cdot
\mathrm{decay}(t;n_{ds},n_{de},n_f)
\]
其中对应参数：
`normal_warmup`, `normal_ramp`, `normal_decay_start/end/final_scale`。
默认值保持与原版一致（`normal_warmup=7000`, `normal_ramp=0`）。

---

### 3.4 Depth supervision loss（新增）

#### 3.4.1 数据与有效像素集合
GT 深度图来自：
\[
D_p \leftarrow \texttt{<source\_path>/<depth\_dir>/<image\_stem>.npy}
\]
在 `utils/camera_utils.py` 中被 resize 到训练分辨率后存入 `Camera.gt_depth`。

训练时有效像素集合 \( \mathcal{V} \)（代码中 `valid2d`）由以下条件交集构成：
1) `finite`：\(\hat{D}_p, D_p\) 都是有限值；  
2) `positive`：\(\hat{D}_p>0,\ D_p>0\)；  
3) `near/far`：\(\hat{D}_p,D_p\in(z_{near},z_{far})\)（可配 `depth_near/depth_far`）；  
4) 可选前景 mask：若 `gt_alpha_mask` 存在，则要求 `mask_p>0.5`。  

#### 3.4.2 raw 深度 vs NDC 深度
代码支持两种比较空间：
- `depth_loss_space="raw"`：直接比较 \(d_p=\hat{D}_p,\ d_p^\* = D_p\)。
- `depth_loss_space="ndc"`：先把 view-space 的 \(z\) 映射到 NDC \(z\in[-1,1]\)，再比较。

##### (1) 从投影矩阵推导 NDC 映射（严格推导）
本仓库投影矩阵（`utils/graphics_utils.py::getProjectionMatrix`）满足（只看 z/w）：
\[
\begin{aligned}
z_{clip} &= A z + B\\
w_{clip} &= z
\end{aligned}
\]
其中
\[
A=\frac{z_{far}}{z_{far}-z_{near}},\quad B=-\frac{z_{far}z_{near}}{z_{far}-z_{near}}
\]
因此
\[
z_{ndc}^{01}=\frac{z_{clip}}{w_{clip}}=A+\frac{B}{z}
\]
可验证：当 \(z=z_{near}\) 时 \(z_{ndc}^{01}=0\)；当 \(z=z_{far}\) 时 \(z_{ndc}^{01}=1\)。
再线性映射到 \([-1,1]\)：
\[
z_{ndc}=2z_{ndc}^{01}-1
\]
这与 `train.py::_depth_z_to_ndc()` 完全一致。

#### 3.4.3 像素级误差：L1 / Huber
令误差 \(e_p=d_p-d_p^\*\)。
- `depth_loss_type="l1"`：
\[
\ell^d_p=|e_p|
\]
- `depth_loss_type="huber"`（PyTorch `F.huber_loss` 的标准 Huber，参数 \(\delta\) 对应 `depth_huber_beta`）：
\[
\ell^d_p=
\begin{cases}
\frac12 e_p^2,& |e_p|\le \delta\\
\delta\left(|e_p|-\frac12\delta\right),& |e_p|>\delta
\end{cases}
\]

#### 3.4.4 Edge-aware 加权（DN-Splatter 风格，新增）
当 `depth_weight_mode="rgb_grad"` 时，使用 GT RGB 的 Sobel 梯度构造权重：

1) 计算灰度（可选）：
\[
Y = 0.2989R + 0.5870G + 0.1140B
\]
2) Sobel 卷积：
\[
G_x = Y * K_x,\quad G_y = Y * K_y
\]
其中（与 `utils/edge_aware_utils.py` 一致）：
\[
K_x=\frac18\begin{bmatrix}1&0&-1\\2&0&-2\\1&0&-1\end{bmatrix},
\quad
K_y=\frac18\begin{bmatrix}1&2&1\\0&0&0\\-1&-2&-1\end{bmatrix}
\]
3) 梯度幅值：
\[
g_p = \sqrt{G_x(p)^2 + G_y(p)^2 + \epsilon}
\]
可选归一化：`norm in {none, mean, max}`。
4) DN-Splatter 风格权重：
\[
w^{grad}_p=\mathrm{clip}\left(\exp(-\alpha g_p),\ w_{min},\ w_{max}\right)
\]
其中 \(\alpha\) 与 clamp 范围由 `depth_grad_alpha`, `depth_weight_min/max` 控制。

#### 3.4.5 Specular-aware 权重修正（新增）
若启用 `spec_enable`，可得到 \(s_p\in\{0,1\}\)（§4.3）。在 `depth_weight_mode="rgb_grad"` 下可进一步修正权重：

- `depth_spec_mode="mul"`：
\[
w_p \leftarrow w_p\cdot (1+\beta s_p)
\]
- `depth_spec_mode="clamp"`：
\[
w_p \leftarrow
\begin{cases}
\max(w_p, w_{spec\_min}),& s_p=1\\
w_p,& s_p=0
\end{cases}
\]

并加入安全阀（避免高光 boost 在深度明显冲突时放大噪声）：
\[
\mathrm{conf}_p=\mathbf 1[\ell^d_p<\tau],\quad
w_p \leftarrow w_p\cdot \left(m + (1-m)\mathrm{conf}_p\right)
\]
其中 \(\tau=\) `depth_conf_tau`，\(m=\) `depth_conf_min_scale`。

#### 3.4.6 最终 depth loss（加权均值）
若使用 edge-aware 权重，则对有效像素集合 \(\mathcal V\)：
\[
L_{\text{depth}}=\frac{\sum_{p\in\mathcal V} w_p\,\ell^d_p}{\sum_{p\in\mathcal V} w_p+\varepsilon}
\]
否则：
\[
L_{\text{depth}}=\frac{1}{|\mathcal V|}\sum_{p\in\mathcal V}\ell^d_p
\]

#### 3.4.7 深度项权重调度（warmup+ramp+decay，新增）
深度监督最终乘的系数为：
\[
\lambda_{\text{depth}}(t)=\lambda_{\text{depth}}^{0}\cdot
\mathrm{ramp}(t;d_w,d_r)\cdot
\mathrm{decay}(t;d_{ds},d_{de},d_f)
\]
对应参数：`lambda_depth`, `depth_warmup`, `depth_ramp`, `depth_decay_start/end/final_scale`。

---

### 3.5 kNN 平滑正则（新增：颜色/opacity 两个 loss）

该部分在 3D 空间的 surfel/gaussian 集合上定义。设当前有 \(N\) 个点：
- 位置：\(\mathbf x_i\in\mathbb R^3\)
- 旋转四元数：\(\mathbf q_i\in\mathbb R^4\)（代码默认 `wxyz`）
- DC 颜色（从 SH0 转成 RGB）：\(\mathbf c_i\in\mathbb R^3\)
- 不透明度：\(o_i\in(0,1)\)

#### 3.5.1 由四元数得到法向（实现对应 `quat_to_normal`）
假设 surfel 的局部法向为 \(+\mathbf z\)，则
\[
\mathbf n_i = \mathbf R(\mathbf q_i)\,[0,0,1]^\top,\qquad \|\mathbf n_i\|=1
\]
代码展开成显式公式并做 normalize（避免数值问题）。

#### 3.5.2 kNN 邻域
对每个点 \(i\)，取其 \(K\) 个最近邻（不含自身）构成邻域 \(\mathcal N(i)\)。
构图在 CPU 上进行（backend：`sklearn` 或 `open3d`），并可选估计 kNN 距离中位数 \(d_{med}\) 用于自动设置 \(\sigma_d\)。

#### 3.5.3 邻接权重（距离核 + 法向核）
对每条边 \((i,j)\) 定义：
\[
w_{ij}^{dist}=\exp\left(-\frac{\|\mathbf x_i-\mathbf x_j\|^2}{2\sigma_d^2}\right)
\]
法向相似度（用 \(1-\cos\theta\) 表示）：
\[
\cos\theta_{ij}=\langle \mathbf n_i,\mathbf n_j\rangle,\quad
a_{ij}=1-\cos\theta_{ij}
\]
\[
w_{ij}^{n}=\exp\left(-\frac{a_{ij}^2}{2\sigma_n^2}\right)
\]
最终权重：
\[
w_{ij}=w_{ij}^{dist}\cdot w_{ij}^{n}
\]
若 `smooth_normal_sigma<=0`，则实现中可退化为只用距离核。

#### 3.5.4 颜色平滑 loss（加权 L2）
\[
L_{\text{smooth\_color}}
=\frac{\sum_i\sum_{j\in\mathcal N(i)} w_{ij}\,\|\mathbf c_i-\mathbf c_j\|_2}{\sum_i\sum_{j\in\mathcal N(i)} w_{ij}+\varepsilon}
\]

#### 3.5.5 Opacity 平滑 loss（加权 L1）
\[
L_{\text{smooth\_opacity}}
=\frac{\sum_i\sum_{j\in\mathcal N(i)} w_{ij}\,|o_i-o_j|}{\sum_i\sum_{j\in\mathcal N(i)} w_{ij}+\varepsilon}
\]

#### 3.5.6 总的平滑项系数
训练中加到 `total_loss` 的是：
\[
\lambda_{\text{smooth}}\,L_{\text{smooth\_color}} + \lambda_{\text{smooth\_op}}\,L_{\text{smooth\_opacity}}
\]
对应参数：`lambda_smooth`, `lambda_smooth_opacity`。

---

### 3.6 Novel-view depth smoothness（新增，可选）

该项的目标是：在一个“微小扰动的邻近视角”上，对渲染深度做边缘感知的 TV 平滑，提升几何的一致性与抗噪性。

#### 3.6.1 视角扰动
在当前相机的局部坐标系中采样一个小 SE(3) 变换 \( \mathbf L \)：
- 旋转：`novel_rot_deg`（绕 x/y/z 小角度）
- 平移：`novel_trans`

然后构造一个 `MiniCam` 并 render 得到：
\(\hat{\mathbf I}^{nv}\)、\(\hat{D}^{nv}\)、\(\hat{\alpha}^{nv}\)。

#### 3.6.2 深度转到 NDC（保证尺度一致）
\[
\hat{d}^{nv} = \mathrm{NDC}(\hat{D}^{nv}; z_{near}, z_{far})
\]
使用 §3.4.2 的公式（但这里 near/far 来自 novel camera 自己的 `znear/zfar`）。

#### 3.6.3 Edge-aware TV（权重来自渲染 RGB）
先用 §3.4.4 同样的方法从 \(\hat{\mathbf I}^{nv}\) 得到权重图 \(w_p\)，并用
\[
M_p=\mathbf 1[\hat{\alpha}^{nv}_p>\tau_\alpha]
\]
过滤背景像素。

对水平/垂直差分：
\[
\Delta_x(p)=|\hat{d}(u+1,v)-\hat{d}(u,v)|,\quad
\Delta_y(p)=|\hat{d}(u,v+1)-\hat{d}(u,v)|
\]
边上权重取相邻像素平均并乘 mask：
\[
w_x(p)=\tfrac12(w_p+w_{p+\hat{x}})\,M_pM_{p+\hat{x}},\quad
w_y(p)=\tfrac12(w_p+w_{p+\hat{y}})\,M_pM_{p+\hat{y}}
\]
最终 TV loss：
\[
L_{\text{novel\_depth\_tv}}
=\frac{\sum w_x\Delta_x + \sum w_y\Delta_y}{\sum w_x + \sum w_y+\varepsilon}
\]
训练中加权：
\[
\lambda_{\text{novel}}\,L_{\text{novel\_depth\_tv}}
\]
对应参数：`lambda_novel_depth_smooth` 及 `novel_*` 一系列超参。

> 代码开关：仅当 `stage in ("B","C")` 且 `iteration % novel_interval == 0` 时计算，否则视为 0。

---

## 4. 辅助构件的数学定义（新增实现所依赖）

### 4.1 `surf_depth` 的定义（说明深度/法向相关 loss 的“预测量”是什么）
渲染器里同时输出 expected depth 与 median depth（来自光栅化的统计量），并以 `depth_ratio` 插值：
\[
\hat{D}=(1-\rho)\,\hat{D}_{exp} + \rho\,\hat{D}_{med},\qquad \rho=\texttt{depth\_ratio}
\]
通常：
- 有界/小场景用 \(\rho=1\)（median 更稳）
- 大场景/无界用 \(\rho=0\)（expected 更不易出现 disk-aliasing）

### 4.2 由深度生成 pseudo normal（影响 `L_normal`、也影响 `L_smooth` 的 normal kernel）
`utils/point_utils.py::depth_to_normal` 的核心是：
1) 由相机模型把深度图 \(D(u,v)\) 反投影为 3D 点 \( \mathbf P(u,v)\)；  
2) 用中心差分近似：
\[
\partial_u \mathbf P \approx \mathbf P(u+1,v)-\mathbf P(u-1,v),\quad
\partial_v \mathbf P \approx \mathbf P(u,v+1)-\mathbf P(u,v-1)
\]
3) 叉积并归一化：
\[
\mathbf n(u,v)=\frac{\partial_u \mathbf P\times \partial_v \mathbf P}{\|\partial_u \mathbf P\times \partial_v \mathbf P\|+\varepsilon}
\]

### 4.3 Specular mask 的定义（新增）
`utils/edge_aware_utils.py::specular_mask` 使用一个简单启发式：
\[
V_p=\max(R_p,G_p,B_p),\quad m_p=\min(R_p,G_p,B_p),\quad S_p=\frac{V_p-m_p}{V_p+\varepsilon}
\]
\[
s_p=\mathbf 1[V_p>t_V]\cdot \mathbf 1[S_p<t_S]
\]
直觉：亮且“接近白色”（饱和度很低）的区域更可能是高光。

---

## 5. 最终 total loss 的数学表达式

综合 §3 的所有项，当前训练每次迭代的总损失为：
```math
\boxed{
\begin{aligned}
L_{\text{total}}(t)
&= L_{\text{photo}} + L_{\text{dist}}(t) + L_{\text{normal}}(t) \\
&\quad + \lambda_{\text{depth}}(t)\,L_{\text{depth}}
 + \lambda_{\text{smooth}}\,L_{\text{smooth\_color}} \\
&\quad + \lambda_{\text{smooth\_op}}\,L_{\text{smooth\_opacity}}
 + \lambda_{\text{novel}}\,L_{\text{novel\_depth\_tv}}
\end{aligned}
}
```


其中每项的“是否生效/何时为 0”取决于代码中的条件：
- \(L_{\text{dist}}(t)\)：当 \(t\le 3000\) 时系数为 0；  
- \(L_{\text{normal}}(t)\)：由 `normal_warmup/ramp/decay` 决定（默认 \(t\le 7000\) 关闭）；  
- \(L_{\text{depth}}\)：仅当该相机视角存在 `gt_depth` 且 `lambda_depth(t)>0` 且有效像素非空；  
- 平滑项：`lambda_smooth` 或 `lambda_smooth_opacity` 为 0 时相应项恒为 0；  
- novel 深度 TV：仅在 Stage B/C 且满足 interval 触发时非零；  
- specular photometric 下调：仅在 `spec_enable=True` 且 Stage A/B 生效（Stage C 直接退化为原版 photometric）。  

---

## 6. 新增/扩展的超参总表（`arguments/__init__.py`）

下面按模块列出新增参数（带默认值与含义）。若未提及则沿用原版。

### 6.1 数据/加载侧
- `ModelParams.depth_dir="depth"`：GT 深度图目录名（相对 `source_path`）。

### 6.2 深度监督（Depth Supervision）
- `lambda_depth=0.0`：深度监督总权重（最终还会乘 warmup/ramp/decay）。
- `depth_loss_space="raw"`：`raw` 或 `ndc`（是否把深度映射到 NDC 再监督）。
- `depth_near=0.2`, `depth_far=1000.0`：深度有效范围（mask）。
- `depth_loss_type="l1"`：`l1` 或 `huber`。
- `depth_huber_beta=0.1`：Huber 的 \(\delta\)。
- `depth_warmup=1000`, `depth_ramp=2000`：深度项 ramp-up。
- `depth_decay_start=-1`, `depth_decay_end=-1`, `depth_final_scale=0.0`：深度项可选线性衰减（默认关闭）。

### 6.3 Edge-aware 深度权重（DN-Splatter 风格）
- `depth_weight_mode="none"`：`none` 或 `rgb_grad`。
- `depth_grad_alpha=10.0`：权重衰减强度。
- `depth_grad_gray=True`：是否先转灰度再做 Sobel。
- `depth_grad_norm="mean"`：梯度归一化模式。
- `depth_weight_min=0.05`, `depth_weight_max=1.0`：权重 clamp 范围。

### 6.4 Specular 处理
- `spec_enable=False`：是否启用 specular mask。
- `spec_tV=0.92`, `spec_tS=0.15`：specular mask 阈值。
- `rgb_spec_gamma=0.9`：specular 区域 photometric L1 下调强度。
- `rgb_spec_gamma_decay_start=-1`, `rgb_spec_gamma_decay_end=-1`, `rgb_spec_gamma_final_scale=0.0`：可选让该下调随训练衰减。
- `depth_spec_mode="mul"`：`mul` 或 `clamp`（深度边缘权重在高光区域的修正方式）。
- `depth_spec_beta=3.0`：`mul` 模式强度。
- `depth_spec_min=0.5`：`clamp` 模式的下限。
- `depth_conf_tau=0.2`, `depth_conf_min_scale=0.2`：深度权重安全阀。

### 6.5 kNN 平滑
- `lambda_smooth=0.0`：颜色平滑权重。
- `lambda_smooth_opacity=0.0`：opacity 平滑权重。
- `smooth_knn_k=16`：kNN 邻居数。
- `smooth_knn_interval=500`：重建 kNN 的间隔（或点集变动时也会触发）。
- `smooth_dist_sigma=0.05`：距离核 \(\sigma_d\)；`<=0` 则自动用 `2*d_med`。
- `smooth_dist_sigma_sample=200000`：估计中位距离时的采样数（0 表示全量）。
- `smooth_normal_sigma=0.3`：法向核 \(\sigma_n\)。
- `smooth_quat_order="wxyz"`：四元数顺序。
- `smooth_chunk_size=65536`：分块计算，避免显存爆炸。

### 6.6 Stage 调度
- `freeze_xyz_iter=-1`：开启后 Stage B 从该迭代开始（冻结 xyz）。
- `appearance_only_iter=-1`：开启后 Stage C 从该迭代开始（只训练 features）。
- `stageB_lr_scale_rotation=0.1`, `stageB_lr_scale_scaling=0.1`, `stageB_lr_scale_opacity=0.1`：Stage B 的 LR 缩放。

### 6.7 normal/dist 相关（扩展）
- `normal_warmup=7000`, `normal_ramp=0`：normal loss 的 ramp-up（默认等价原版硬阈值 7000）。
- `normal_decay_start=-1`, `normal_decay_end=-1`, `normal_final_scale=0.0`：可选 normal loss 衰减。
- `lambda_dist`：仍保留原版 `>3000` 的硬开关。

### 6.8 Novel-view 深度 TV
- `lambda_novel_depth_smooth=0.0`：novel-view 深度平滑权重。
- `novel_interval=200`：触发间隔。
- `novel_resolution=256`：渲染分辨率（短边对齐/缩放）。
- `novel_rot_deg=2.0`, `novel_trans=0.01`：扰动幅度。
- `novel_rgb_alpha=10.0`：edge-aware 权重里的 \(\alpha\)。
- `novel_alpha_thresh=0.5`：前景 mask 阈值。

---

## 7. 其它工程性增强

### 7.1 COLMAP 转换更健壮（`convert.py`）
- 自动识别 COLMAP 新旧版本的 GPU 参数名：  
  - `feature_extractor`: `--FeatureExtraction.use_gpu` vs `--SiftExtraction.use_gpu`  
  - `exhaustive_matcher`: `--FeatureMatching.use_gpu` vs `--SiftMatching.use_gpu`
- 将 `Mapper.ba_global_function_tolerance` 调小到 `1e-6`（原注释：加速 BA）。

### 7.2 Viewer 端口自动回退（`gaussian_renderer/network_gui.py` / `train.py` / `view.py`）
- 当端口被占用或无权限时，按 `port_retries` 递增尝试；必要时绑定到系统分配的 ephemeral 端口。
- `init()` 返回实际端口，训练/查看时会打印最终监听端口。

### 7.3 导出时的评测更丰富（`render.py`）
- 在导出 train/test 渲染结果的同时，计算：
  - RGB：PSNR / SSIM / LPIPS(VGG)
  - 深度：Depth L1 / Depth RMSE（若存在 `cam.gt_depth`）
- 写入 `metrics.txt`（含 per-view 列表），方便批量对比实验。

### 7.4 Depth Anything 3 深度生成工具（`tools/generate_depth_anything.py`）
- 支持两种输入：
  1) `transforms.json`（NeRF 风格或自定义 schema）
  2) COLMAP `sparse/0/*.bin`
- 输出：
  - `<source_path>/depth/<img_stem>.npy`
  - 可选 `<img_stem>.png`（1%–99% 截断可视化）
- 可选把相机外参/内参传给 DepthAnything3，启用其内部的尺度对齐（`align_to_input_ext_scale`）。

---


