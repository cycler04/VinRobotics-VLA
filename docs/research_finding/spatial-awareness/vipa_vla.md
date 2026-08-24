# VIPA-VLA: Spatial-Aware VLA Pretraining through Visual-Physical Alignment from Human Videos

> **Paper Identity:**
> - **Title:** VIPA-VLA: Spatial-Aware VLA Pretraining through Visual-Physical Alignment from Human Videos
> - **Authors:** Yicheng Feng (Peking University / BeingBeyond), Wanpeng Zhang, Ye Wang, Hao Luo, Haoqi Yuan, Sipeng Zheng, Zongqing Lu (Peking University, Renmin University of China, BeingBeyond)
> - **arXiv / Date:** [arXiv:2512.13080v1 [cs.RO]](https://arxiv.org/abs/2512.13080), 15 Dec 2025
> - **Project / Code:** [https://beingbeyond.github.io/VIPA-VLA](https://beingbeyond.github.io/VIPA-VLA)
> - **PDF Location:** [docs/papers/spatial-awareness/Spatial-Aware VLA Pretrain.pdf](../../papers/spatial-awareness/Spatial-Aware%20VLA%20Pretrain.pdf)

---

## 1. Why — Problem Statement & Motivation

**Problem:**
- Current Vision-Language-Action (VLA) models attempt to infer 3D physical actions directly from 2D pixel observations. This creates a severe **2D visual-to-3D physical grounding gap**.
- Standard 2D vision backbones (e.g., SigLIP, DINOv2, CLIP) excel at visual-semantic reasoning ("what object is where on the image plane") but lack explicit 3D spatial awareness (depth, 3D geometry, distance, orientation, and physical spatial bounds).
- **Limitations of existing spatial/3D VLA approaches:**
    - **RGB-D / Hardware Dependency:** Methods requiring depth sensors in real deployment suffer when sensors encounter transparent, reflective, or small objects, and cannot readily utilize massive RGB-only human video pretraining datasets.
    - **3D Point Cloud / Gaussian Splatting Heavy Pipelines:** Point-cloud or 3D token representations (e.g., 3D-VLA) introduce heavy computational complexity and memory overhead, while lacking large-scale pretraining datasets.
    - **2D Point Tracking & Heuristic Representations:** Methods such as TraceVLA or HistRISE rely on 2D point trajectories or 2D segmentation (CoTracker, SAM 2), which miss true 3D spatial depth and real-world physical scale calibration.
    - **Lack of 3D Action Supervision in Human Videos:** Large-scale human video collections (Ego4D, Something-Something, etc.) contain rich interaction knowledge, but lack explicit 3D action labels or 3D hand/camera trajectories scaled to physical coordinates.

**Solution:**
- Introduce **Spatial-Aware VLA Pretraining**, a paradigm that performs explicit **visual-physical alignment** between 2D visual observations and 3D physical space *before* downstream robot policy learning.
- Construct **Hand3D**, a spatial-aware pretraining dataset derived from 1.03M+ human manipulation video clips:
    - **Hand3D-visual:** Combines 3D point cloud estimation (Cut3R), object proposals (Gemini-2.5-flash + GroundingDINO), and MANO-based 3D hand pose annotations calibrated to absolute physical scales.
    - **Hand3D-action:** Converts continuous 3D hand/wrist trajectories into discretized **3D motion tokens** ($1\text{m}^3$ physical volume discretized into $1024^3$ spatial bins).
- Propose **VIPA-VLA**, a dual-encoder architecture that fuses semantic visual representations with 3D spatial representations from a pretrained 3D vision encoder (Cut3R) via a learnable **Cross-Attention Fusion Layer**.
- Implement a **progressive 2-stage pretraining strategy** on human videos, followed by post-training with a Flow Matching Action Head (DiT) on downstream robot manipulation tasks.

---

## 2. Architecture Overview

- **Evaluated Robot Models & Environments:**
    - **LIBERO Sim:** Franka Emika Panda 7-DoF arm in MuJoCo (single-view and two-view inputs, 500 trials per suite across Spatial, Object, Goal, Long).
    - **RoboCasa Sim:** 3-view inputs, evaluated across 24 complex kitchen tasks (Pick & Place, Doors/Drawers, Others) using only 50 demonstrations per task.
    - **Real-World Setup:** 7-DoF Franka Research 3 arm with a 6-DoF Inspire dextrous hand and 2x RealSense L515 cameras (third-person & wrist views). Tasks: `Put-Three-Obj`, `Wipe-Board`, `Water-Plant` (in both seen and unseen environments).

- **Input Formats:**
    - **Visual:** 1 to 4 RGB frames (resized to 448×448). Stage 1 pretraining uses 1–4 consecutive frames for spatial VQA; Stage 2 & 3 use single-frame visual observations.
    - **Instruction Prompt:** Natural language instruction $l$.
    - **Conditioning Context:** $h_{\mathrm{cond}} = \text{VLM}_\phi(v, l, Q_a)$ extracted from fixed action queries $Q_a$.

- **Model Parameters & Module Specifications:**
    - **Semantic Vision Encoder:** ViT from InternVL3.5-2B ($V_{\mathrm{sem}} \in \mathbb{R}^{N_v \times d_v}$).
    - **3D Spatial Vision Encoder:** Cut3R ($V_{\mathrm{spa}} \in \mathbb{R}^{N_s \times d_s}$), providing explicit geometric understanding.
    - **VLM Backbone:** InternVL3.5-2B LLM backbone.
    - **Fusion Layer:** Cross-attention module ($F_{\mathrm{spa}} = \text{CrossAttn}(V_{\mathrm{sem}}, V_{\mathrm{spa}})$, residual connection $V_f = V_{\mathrm{sem}} + \alpha F_{\mathrm{spa}}$ with learnable scaling $\alpha$, initialized at 0.5).
    - **Motion Token Space:** $K = 1024$ bins per axis ($x, y \in [-0.5, 0.5]\text{m}, z \in [0, 1]\text{m}$ in front of camera), tokenizing 3D waypoints $(x_t, y_t, z_t)$ into discrete token triplets $(m_{xt}, m_{yt}, m_{zt})$.
    - **Action Head:** Diffusion Transformer (DiT) trained with a Flow Matching objective ($L_{\mathrm{FM}}$).

- **Output:** Executable continuous 3D action chunks $a_t = \{a_{t1}, \dots, a_{tH}\}$ (e.g., 7-DoF joint/pose control or dextrous hand movements).

---

## 3. Method

### 3.1 Hand3D Dataset Curation & Scale-Calibrated 3D Annotation Pipeline

- **Failure mode addressed:** Prevents spatial scale distortion and 2D-to-3D misalignment caused by uncalibrated 2D human video observations.
- **Input:** Raw human demonstration videos aggregated from MoCap datasets (Arctic, HOI4D, FPHA, H2O, OakInk2, TACO, DexYCB), VR datasets (EgoDex), and pseudo-annotated videos (Taste-Rob).
- **Mechanism:**
    1. **MANO Fitting & Alignment:** Fits human hand pose and shape to the standard MANO parametric representation $m = \{\theta, r, \tau, \beta\}$ across all datasets (using gradient optimization or HaWoR).
    2. **Point Cloud Estimation:** Uses Cut3R to estimate dense per-frame relative point cloud coordinates $P = \{(x_i, y_i, z_i)\}_{i=1}^N$.
    3. **Object Localization:** Uses Gemini-2.5-flash for object proposal generation and GroundingDINO for 2D bounding boxes $B_o$, combined with depth $P$ to localize objects in 3D space.
    4. **Scale Calibration:** Matches absolute MANO hand joint depths $J_{hz} = \{j_{kz}\}$ with relative point cloud depths $\tilde{J}_{hz} = \{\tilde{j}_{kz}\}$ to compute the scale factor $s$:
       $$s = \text{median}_{k \in \Omega} \left( \frac{j_{kz}}{\tilde{j}_{kz}} \right)$$
       Applying $s$ yields calibrated point clouds $sP$, placing hands and objects into a unified, physical 3D coordinate system.
    5. **Instructional VQA Curation (Hand3D-visual):** Generates 4 categories of VQA pairs using Gemini-2.5-flash: Spatial Relationship, Task Completion, Hand Movement, Camera Movement.
    6. **Discretized Motion Tokenization (Hand3D-action):** Discretizes continuous wrist waypoints $(x_t, y_t, z_t)$ in a $1\text{m}^3$ physical bounding volume into $K=1024$ discrete motion tokens $(m_{xt}, m_{yt}, m_{zt})$ per waypoint, yielding 1.03M+ motion sequence training samples.
- **Output:** Hand3D-visual VQA dataset and Hand3D-action motion token dataset.

### 3.2 Dual-Encoder Architecture & Cross-Attention Fusion Layer

- **Failure mode addressed:** Solves the lack of explicit 3D geometric structure in standard 2D ViT visual features by injecting pretrained 3D spatial features directly into semantic visual tokens.
- **Input:** Visual image frame $v$.
- **Mechanism:**
    - Semantic encoder produces visual embeddings $V_{\mathrm{sem}} \in \mathbb{R}^{N_v \times d_v}$; Cut3R 3D encoder outputs spatial embeddings $V_{\mathrm{spa}} \in \mathbb{R}^{N_s \times d_s}$.
    - Both features are projected into a shared attention space where $V_{\mathrm{sem}}$ queries $V_{\mathrm{spa}}$ via cross-attention.
    - The cross-attention output $F_{\mathrm{spa}}$ is combined with semantic features via a residual connection with learnable parameter $\alpha$:
      $$V_f = V_{\mathrm{sem}} + \alpha F_{\mathrm{spa}}$$
    - Dropout and layer normalization are applied to stabilize optimization, producing spatial-semantic fused visual tokens $V_f$.
- **Output:** Fused 3D-aware visual token representation $V_f$.

### 3.3 Progressive 2-Stage Spatial-Aware VLA Pretraining

- **Failure mode addressed:** Prevents sub-optimal downstream policy learning by progressively building spatial visual grounding first, followed by fine-grained 3D action trajectory prediction.
- **Input:** Fused visual tokens $V_f$, text instructions, and motion tokens.
- **Mechanism:**
    - **Stage 1 (3D-Visual Pretraining):** Initializes from pretrained VLM backbone. Freezes semantic encoder, 3D encoder, and LLM backbone; trains *only* the fusion layer on Hand3D-visual VQA data. Aligns $V_{\mathrm{sem}}$ and $V_{\mathrm{spa}}$ to reason about 3D spatial relationships, directions, and distances.
    - **Stage 2 (3D-Action Pretraining):** Extends the LLM vocabulary with discrete motion tokens. Freezes visual encoders; trains the fusion layer and LLM backbone on Hand3D-action motion sequences to predict 3D motion tokens conditioned on visual-text inputs.
- **Output:** Pretrained VIPA-VLA model equipped with 2D-to-3D visual-physical alignment and 3D motion priors.

### 3.4 Robot Post-Training via Flow Matching Action Head

- **Failure mode addressed:** Eliminates execution errors during real-world and simulated robot control by generating continuous action trajectories conditioned on spatial VLM features.
- **Input:** Fused visual features $V_f$, language instruction $l$, action queries $Q_a$, robot state $s_t$, and noisy action trajectory $\tilde{a}_t^{(\tau)} = (1 - \tau)\epsilon + \tau a_t$ with $\tau \sim U(0, 1)$.
- **Mechanism:**
    - Extracts conditional context $h_{\mathrm{cond}} = \text{VLM}_\phi(v, l, Q_a)$ from VLM hidden states corresponding to $Q_a$.
    - Inputs $h_{\mathrm{DiT}} = \text{concat}[\tilde{a}_t^{(\tau)}, s_t]$ and $h_{\mathrm{cond}}$ into a Diffusion Transformer (DiT) action head.
    - Minimizes the Flow Matching loss objective:
      $$\mathcal{L}_{\mathrm{FM}} = \mathbb{E}_{a_t, \tau, \epsilon, v, l} \left[ \| v_\theta - (a_t - \epsilon) \|_2^2 \right]$$
    - Freezes visual encoders and trains the LLM backbone and action head $f_\theta$.
- **Output:** Clean continuous 3D robot action chunk $a_t$.

---

## 4. Training Pipeline & Configurations

The training process consists of 3 distinct, progressive stages:

| Attribute / Hyperparameter | Stage 1: 3D-Visual Pretraining | Stage 2: 3D-Action Pretraining | Stage 3: Robot Post-Training |
| :--- | :--- | :--- | :--- |
| **Dataset** | Hand3D-visual (VQA pairs) | Hand3D-action (1.03M motion sequences) | Robot datasets (LIBERO / RoboCasa / Real) |
| **Trainable Modules** | Fusion Layer only ($\alpha$ init 0.5) | Fusion Layer + LLM Backbone | LLM Backbone + DiT Action Head ($f_\theta$) |
| **Frozen Modules** | Semantic Encoder, 3D Encoder, LLM | Semantic Encoder, 3D Encoder | Semantic Encoder, 3D Encoder (Cut3R) |
| **Objective / Loss** | VQA Text Cross-Entropy Loss | Motion Token Prediction Loss | Flow Matching Loss $\mathcal{L}_{\mathrm{FM}}$ |
| **Input Frame Count** | 1–4 frames (sampled at 1 fps) | Single frame | Single frame |
| **Optimizer & LR** | AdamW, lr $1\times 10^{-5}$ (warmup 0.03, weight decay 0.01) | AdamW, lr $1\times 10^{-5}$ (warmup 0.03, weight decay 0.01) | AdamW, lr $5\times 10^{-5}$ (warmup 0.05, weight decay $1\times 10^{-5}$) |
| **Compute & Duration** | 1 epoch (~6 hours on 8x NVIDIA A800) | 1 epoch (~20 hours on 8x NVIDIA A800) | 30K steps (LIBERO/Real: ~5h) / 60K (RoboCasa: ~40h) on 8x A800 |
| **Global Batch Size** | 32 | 32 | 128 (LIBERO / Real) / 256 (RoboCasa) |
| **Image Resolution** | 448 × 448 | 448 × 448 | 448 × 448 |

---

## 5. Claim → Evidence (Experimental Results)

### 5.1 Simulation Benchmarks: LIBERO & RoboCasa

**LIBERO Benchmark Results (Success Rate % across 500 trials per task suite):**

| Model | Input Setup | Spatial (%) | Object (%) | Goal (%) | Long (%) | Average (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| TraceVLA | Single-View | 84.6 | 85.2 | 75.1 | 54.1 | 74.8 |
| OpenVLA | Single-View | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| SpatialVLA | Single-View | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| DiT Policy | Single-View | 84.2 | 96.3 | 85.4 | 63.8 | 82.4 |
| CoT-VLA | Single-View | 87.5 | 91.6 | 87.6 | 69.0 | 83.9 |
| GR00T N1.5* | Single-View | 91.4 | 97.6 | 94.0 | 85.6 | 92.1 |
| **VIPA-VLA (Ours)** | **Single-View** | **92.6** | **97.2** | **94.2** | **85.6** | **92.4** |
| $\pi_0$ | Two-View | 98.0 | 96.8 | 94.4 | 88.4 | 94.4 |
| UniVLA | Two-View | 95.4 | 98.8 | 93.6 | 94.0 | 95.5 |
| $\pi_{0.5}$ | Two-View | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| **VIPA-VLA (Ours)** | **Two-View** | **96.6** | **98.6** | **97.0** | **95.0** | **96.8** |

**RoboCasa Benchmark Results (3-view input, 50 demonstrations/task, 24 tasks):**

| Model | Pick & Place (%) | Doors / Drawers (%) | Others (%) | Average (%) |
| :--- | :---: | :---: | :---: | :---: |
| GR00T N1 | 18.6 | 50.2 | 39.1 | 36.0 |
| $\pi_{0.5}$ | 21.5 | 57.8 | 44.9 | 41.4 |
| **VIPA-VLA (Ours)** | **20.8** | **67.7 (+9.9%)** | **52.8 (+7.9%)** | **45.8 (+4.4%)** |

### 5.2 Real Robot Experiments (7-DoF Franka + Inspire Dextrous Hand)

Results presented as **Sub-task Success Rate / Full Task Success Rate (%):**

| Environment | Model | Put-Three-Obj | Wipe-Board | Water-Plant |
| :--- | :--- | :---: | :---: | :---: |
| **Seen Environments** | GR00T N1.5 | 48% / 40% | 57% / 30% | 53% / 30% |
| | Being-H0 | 38% / 20% | 40% / 10% | 37% / 20% |
| | InternVL3.5 | 34% / 10% | 43% / 10% | 37% / 20% |
| | **VIPA-VLA (Ours)** | **52% / 10%** | **83% / 60%** | **57% / 50%** |
| **Unseen Environments** | GR00T N1.5 | 28% / 0% | 43% / 10% | N/A |
| | Being-H0 | 16% / 0% | 33% / 10% | N/A |
| | InternVL3.5 | 42% / 10% | 40% / 10% | N/A |
| | **VIPA-VLA (Ours)** | **44% / 20%** | **83% / 50%** | N/A |

### 5.3 3D Spatial Understanding & Architecture Ablations

**3D Spatial Reasoning Evaluation on Hand3D-test (2K unseen VQA pairs):**

| Model | Distance Error (m) $\downarrow$ | Direction Score (out of 3) $\uparrow$ |
| :--- | :---: | :---: |
| InternVL3.5 (Baseline VLM) | 0.18 m | 1.22 / 3 |
| InternVL3.5 + Hand3D (Data only) | 0.14 m | 1.75 / 3 |
| **VIPA-VLA-PT (Data + Dual Encoder)** | **0.12 m** | **1.82 / 3** |

**Architecture & Pretraining Ablation on LIBERO (Single-View Avg %):**

| Variant | Avg Success (%) | Performance Impact |
| :--- | :---: | :---: |
| **VIPA-VLA (Full)** | **92.4%** | Baseline |
| - W/o Pretraining | 91.2% | -1.2% |
| - W/o Dual Encoder | 90.4% | -2.0% |
| - W/o Both (Baseline VLM + DiT) | 88.7% | -3.7% |

---

## 6. Workspace Insight & Connection

- **Alignment with VLA-Core Architecture:**
  VIPA-VLA combines a frozen/fine-tuned VLM backbone with a Flow Matching Action Head (DiT). This matches the action chunking and flow matching policy direction in our workspace (`vla-core`).
- **Data Pipeline & Canonical Format Integration:**
  - Hand3D demonstrates that large-scale human demonstration videos can be converted into effective 3D spatial pretraining signals without relying on physical robot datasets during pretraining.
  - The scale calibration procedure (matching MANO joint positions with point clouds) provides a concrete methodology for calibrating multi-camera / third-person visual inputs within `vla-data-tools` canonical episode pipelines.
- **Key Policy Design Insight:**
  - Fusing 3D geometric features (from Cut3R or similar depth backbones) into the semantic visual encoder via cross-attention yields significant performance gains on spatially demanding tasks (e.g., Doors/Drawers in RoboCasa, Wipe-Board/Water-Plant in real-world setups) without needing major changes to the action head architecture.

---

## 7. Limitations & Residual Findings

1. **Single-Frame Temporal Limit:**
   Stage 2 pretraining and Stage 3 post-training rely primarily on single-frame visual inputs. While this minimizes token counts and GPU memory, it lacks explicit long-horizon temporal memory (unlike ReMem-VLA or MemoryVLA).
2. **Upstream Annotation Dependency:**
   Hand3D dataset generation depends on upstream models: Cut3R (point cloud depth), Gemini-2.5-flash (object proposals), GroundingDINO (bounding boxes), and MANO joint fitting. Errors in upstream depth estimation or hand fitting propagate to pretraining supervision.
3. **Bounded Motion Volume Constraint:**
   The continuous-to-discrete motion tokenization restricts wrist waypoints to a fixed $1\text{m}^3$ volume ($x,y \in [-0.5, 0.5]\text{m}, z \in [0, 1]\text{m}$) directly in front of the camera. Trajectories extending beyond this volume cannot be represented without re-scaling or re-binning.
4. **Fine-Grained Manipulation Inaccuracies:**
   Failure analysis shows that while VIPA-VLA achieves accurate global 3D spatial target localization, errors still occur during sub-millimeter grasping alignment or subtle force adjustments, highlighting the need for tactile or proprioceptive feedback for micro-precision tasks.

---

## 8. Proposed Next Experiments

1. **Combine 3D Spatial Fusion with Recurrent Memory:**
   Test integrating VIPA-VLA's 3D-aware dual encoder with a dual-level recurrent memory query scheme (like ReMem-VLA) to address both 3D spatial grounding AND long-horizon temporal dependencies simultaneously.
2. **Evaluate Cut3R vs. Lightweight Depth Backbones:**
   Ablate substituting Cut3R with lighter depth/3D feature extractors (e.g., Depth-Anything-V2 or DINOv2 depth heads) to measure inference latency vs. spatial grounding accuracy trade-offs.
3. **Implement Scale Calibration Filter in Data Tools:**
   Add a scale-calibration utility in `vla-data-tools` to assist in canonical episode inspection and multi-camera spatial alignment.
