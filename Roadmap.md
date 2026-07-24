Kết luận định hướng

Với bài toán này, bạn không nên bắt đầu bằng việc “tìm model mạnh nhất rồi train”. Pipeline đúng phải là:

Data audit→Camera validation→Baseline 3DGS→Local validation→Error analysis→Targeted improvements→Ensemble/config selection→Submission validation
	​


Mục tiêu thực tế không phải xây một model duy nhất, mà là xây một hệ thống tự động chọn cấu hình tốt nhất cho từng scene.

Khuyến nghị ban đầu của tôi:

Baseline bắt buộc: 3D Gaussian Splatting chính thức.
Ứng viên chính để nâng điểm: Mip-Splatting.
Ứng viên thứ hai: Scaffold-GS.
Ứng viên thiên về hình học: 2D Gaussian Splatting.
Không nên bắt đầu bằng NeRF thuần vì tốc độ thử nghiệm chậm hơn và khó chạy nhiều ablation trong thời gian cuộc thi.
Không nên dùng diffusion để “làm đẹp” ảnh ở giai đoạn đầu vì có thể tăng cảm quan nhưng làm sai thiết bị, dây cáp hoặc kết cấu, khiến PSNR và SSIM giảm.

3DGS khởi tạo từ sparse point cloud, tối ưu vị trí, covariance, opacity và màu của các Gaussian, đồng thời thực hiện densification/pruning trong quá trình train. Đây là một baseline rất phù hợp vì dữ liệu của bạn đã có reconstruction từ COLMAP.

1. Trước hết: hiểu đúng bản chất bài toán
1.1 Đây không phải bài toán “sinh ảnh” thông thường

Bạn không được tự do tạo một ảnh “trông giống trạm BTS”. Ảnh đầu ra phải là phép render của cùng một scene 3D, tại đúng camera:

I
pred
	​

=R
θ
	​

(K
test
	​

,R
test
	​

,t
test
	​

)

Trong đó:

R
θ
	​

: representation và renderer đã học.
K: camera intrinsics.
R,t: extrinsics.
I
pred
	​

: ảnh tại test pose.

Model phải học hai nhóm thông tin:

Geometry
Vật thể nằm ở đâu trong không gian.
Bề mặt hướng theo hướng nào.
Quan hệ che khuất giữa các vật thể.
Cấu trúc nhỏ như dây, thanh kim loại, anten và giá đỡ.
Appearance
Màu sắc.
Texture.
Ánh sáng.
Reflection và specular.
Thay đổi diện mạo theo hướng nhìn.

Điểm khó của trạm BTS là nó chứa nhiều:

Cấu trúc mỏng.
Thanh kim loại dài.
Dây cáp nhỏ.
Vật thể lặp lại.
Bề mặt phản xạ.
Bầu trời và nền xa.
Texture nghèo hoặc gần như đồng nhất.

Đây đều là trường hợp dễ gây:

Floating artifacts.
Gaussian phình quá lớn.
Mất dây hoặc anten nhỏ.
Sai che khuất.
Ghosting.
Blur tại góc nhìn mới.
1.2 Metrics quyết định cách tối ưu

Điểm:

Score=0.4(1−LPIPS)+0.3SSIM+0.3PSNR
norm
	​


Điều này có nghĩa:

LPIPS có trọng số lớn nhất.
Nhưng PSNR và SSIM chiếm tổng cộng 60%.
Một ảnh “đẹp hơn” nhưng lệch vài pixel vẫn có thể bị trừ mạnh.
Hallucination gần như luôn nguy hiểm.
Geometry và camera alignment phải được ưu tiên trước enhancement.

Thứ tự ưu tiên:

Camera correctness>Geometry>Exposure/color consistency>Fine texture>Visual enhancement

Sai camera 1–2 pixel có thể gây giảm cả ba metrics trên cạnh vật thể.

2. Lộ trình tổng thể từ hiểu sâu đến điểm cao

Tôi chia công việc thành tám giai đoạn.

Giai đoạn 0 — Xây dựng môi trường có thể tái lập

Mục tiêu của giai đoạn này không phải đạt điểm cao, mà là bảo đảm:

Một command có thể lấy dữ liệu, train một scene, render test poses và tạo submission đúng định dạng.

Các thành phần cần có
GitHub repository
        │
        ├── Source code
        ├── Configs
        ├── Scripts
        ├── Environment specification
        └── Kaggle entry point
                 │
                 ▼
        Kaggle Notebook / Script
                 │
                 ├── Clone repository
                 ├── Install CUDA extensions
                 ├── Locate dataset
                 ├── Train scenes
                 ├── Render test poses
                 ├── Validate outputs
                 └── Create submission.zip
Nguyên tắc

Không đặt logic quan trọng trực tiếp trong notebook. Notebook chỉ nên:

!git clone ...
!pip install ...
!python tools/train_all.py --config configs/competition.yaml
!python tools/render_submission.py ...

Toàn bộ logic phải nằm trong repository để:

Version control.
Chạy lại được.
Debug được.
Chuyển từ Kaggle sang máy local hoặc server.
Chứng minh khả năng tái lập khi ban tổ chức yêu cầu.
Definition of Done

Giai đoạn 0 hoàn thành khi:

python tools/run_pipeline.py \
    --data-root /kaggle/input/competition-data \
    --output-root /kaggle/working/run_001 \
    --config configs/baseline/3dgs.yaml

tự động tạo ra:

run_001/
├── checkpoints/
├── renders/
├── metrics/
├── logs/
├── resolved_config.yaml
└── submission.zip
Giai đoạn 1 — Hiểu và kiểm định dữ liệu

Đây có thể là phần mang lại nhiều điểm hơn việc đổi model.

1.1 Xây data inspector

Trước khi train, mỗi scene phải được phân tích tự động.

Thống kê ảnh
Số lượng ảnh.
Width, height.
Aspect ratio.
Mean/std RGB.
Độ sáng.
Blur score.
Saturation.
Overexposure và underexposure.
Ảnh trùng hoặc gần trùng.
Thống kê camera
Camera centers.
Hướng nhìn.
Khoảng cách camera tới scene.
Distribution của focal length.
Khoảng cách giữa các camera.
Coverage theo azimuth/elevation.
Test pose nằm trong hay ngoài train trajectory.
Thống kê sparse point cloud
Số points.
Bounding box.
Point density.
Reprojection error nếu lấy được.
Track length.
Outlier points.
Khoảng cách point cloud tới cameras.
1.2 Visualization bắt buộc

Mỗi scene nên sinh:

scene_report/
├── camera_trajectory_3d.png
├── camera_azimuth_elevation.png
├── focal_distribution.png
├── image_brightness.png
├── sparse_point_cloud.ply
├── train_test_pose_overlap.png
└── report.json
Biểu đồ quan trọng nhất

Vẽ train camera và test camera trong cùng hệ tọa độ:

                 test camera
                      ×
               ×             ×

        o   o   o   o   o   o   o
      train cameras surrounding BTS

Điều này giúp phân loại test poses:

Interpolation: nằm giữa các train cameras.
Mild extrapolation: hơi ra ngoài trajectory.
Strong extrapolation: xa distribution train.
Scale shift: khoảng cách hoặc focal khác đáng kể.
Elevation shift: camera cao/thấp hơn vùng train.

Phân loại này quyết định thuật toán nào phù hợp.

1.3 Xác minh convention của camera

Đây là lỗi có thể khiến toàn bộ submission sai dù model train tốt.

COLMAP thường biểu diễn:

x
cam
	​

=Rx
world
	​

+t

Camera center:

C=−R
⊤
t

Quaternion trong CSV phải được chuyển thành rotation matrix theo đúng:

Thứ tự q
w
	​

,q
x
	​

,q
y
	​

,q
z
	​

.
World-to-camera hay camera-to-world.
Hệ trục của COLMAP.
Convention của renderer.
Pixel center convention.
Principal point convention.

Phải viết unit test:

Đọc camera từ images.bin.
Render hoặc project sparse points vào một ảnh train.
Kiểm tra points có nằm đúng vị trí ảnh hay không.
Chuyển pose sang renderer.
Chuyển ngược lại và kiểm tra sai số.

Không nên tin conversion chỉ vì “ảnh render nhìn gần đúng”.

Giai đoạn 2 — Xây local validation đúng cách

Bạn không có ground-truth test, vì vậy phải tạo validation từ train images.

2.1 Không random split ảnh một cách ngây thơ

Drone thường chụp theo trajectory liên tục. Hai frame liền nhau có pose rất gần nhau.

Nếu random 80/20:

train: frame 001, 003, 004
val:   frame 002

Val quá dễ vì frame 002 gần như trùng góc nhìn với train.

Kết quả local cao nhưng leaderboard thấp.

2.2 Các validation split cần có
Split A — Interpolation

Giữ lại đều đặn các frame trong trajectory:

train train val train train val ...

Đo khả năng interpolation.

Split B — Angular block

Giữ lại một dải góc camera liên tiếp:

0°–60°: validation
60°–360°: training

Đo mild extrapolation và khả năng suy luận phần bị thiếu.

Split C — Distance/focal split

Giữ lại ảnh:

Xa nhất.
Gần nhất.
Focal đặc biệt.
Resolution khác.

Đo robustness khi sampling rate thay đổi.

Split D — Elevation split

Giữ lại camera cao hoặc thấp hơn phần còn lại.

2.3 Chỉ số cần lưu

Không chỉ lưu trung bình:

{
  "mean_psnr": 0,
  "mean_ssim": 0,
  "mean_lpips": 0,
  "final_score": 0,
  "per_view": [],
  "worst_10_views": [],
  "by_pose_type": {
    "interpolation": {},
    "extrapolation": {},
    "scale_shift": {}
  }
}

Bạn cần biết model thua ở loại camera nào, không chỉ biết điểm trung bình.

Giai đoạn 3 — Baseline 3DGS sạch và đáng tin cậy
3.1 Vì sao dùng 3DGS trước

3DGS phù hợp trực tiếp với dữ liệu:

COLMAP sparse points
        │
        ▼
Initialize 3D Gaussians
        │
        ├── position μ
        ├── covariance Σ
        ├── opacity α
        ├── color / spherical harmonics
        └── scale + rotation
        │
        ▼
Differentiable rasterization
        │
        ▼
Rendered RGB
        │
        ▼
Photometric loss

Representation của 3DGS sử dụng các Gaussian anisotropic và tối ưu xen kẽ với density control; renderer visibility-aware giúp quá trình train và render nhanh.

3.2 Loss baseline

Baseline thường có dạng:

L
rgb
	​

=(1−λ)L
1
	​

+λL
DSSIM
	​


Nhưng đối với cuộc thi, nên thử:

L=λ
1
	​

L
1
	​

+λ
s
	​

L
SSIM
	​

+λ
p
	​

L
perceptual
	​

+λ
r
	​

L
regularization
	​


Tuy nhiên:

Không thêm LPIPS loss ngay từ đầu.
LPIPS loss có thể cải thiện perceptual quality nhưng làm giảm pixel accuracy.
Cần đánh giá trực tiếp bằng score cuối cùng.
3.3 Baseline experiment đầu tiên

Cố định:

Resolution gốc hoặc một mức downscale.
30k iterations.
Default densification.
Default SH degree schedule.
Không pose refinement.
Không exposure correction.
Không mask.
Một seed cố định.

Kết quả cần lưu:

Training time.
Peak VRAM.
Số Gaussians cuối.
Validation metrics.
Render của best/worst views.
Checkpoint.
Config đầy đủ.

Baseline này là “control group” cho mọi cải tiến sau.

Giai đoạn 4 — Phân tích lỗi trước khi đổi thuật toán

Mỗi ảnh validation xấu cần được phân loại.

Taxonomy lỗi
A. Camera/pose error

Biểu hiện:

Toàn bộ cạnh bị double.
Cấu trúc dịch chuyển đồng đều.
PSNR, SSIM và LPIPS đều xấu.
Training views cũng không fit tốt.

Giải pháp:

Kiểm tra convention.
Pose refinement nhỏ.
Intrinsics refinement.
Loại camera outlier.
B. Exposure/color mismatch

Biểu hiện:

Geometry đúng.
Ảnh sáng/tối hoặc lệch màu.
LPIPS có thể tương đối ổn nhưng PSNR thấp.

Giải pháp:

Per-image exposure model.
Color affine transform.
White-balance normalization.
Camera appearance embeddings, nhưng phải regularize.
C. Geometry underfitting

Biểu hiện:

Dây và thanh mỏng biến mất.
Edge mềm.
Vật thể bị gộp.

Giải pháp:

Tăng resolution.
Tăng densification.
Điều chỉnh split threshold.
Tăng iteration.
Dùng 2DGS hoặc geometry regularization.
D. Floaters/overfitting

Biểu hiện:

Training view tốt.
Validation có các đám Gaussian lơ lửng.
Ghosting khi đổi góc.

Giải pháp:

Pruning mạnh hơn.
Opacity regularization.
Scale regularization.
Scaffold-GS.
2DGS.
Loại sparse point outlier.
E. Aliasing/scale shift

Biểu hiện:

Train ở một khoảng cách tốt.
Test gần hoặc xa bị răng cưa, phình hoặc mất chi tiết.

Giải pháp:

Mip-Splatting.
Multi-scale training.
Randomized resolution.
Scale-aware filtering.

Mip-Splatting được thiết kế đặc biệt để xử lý artifacts khi focal length, camera distance hoặc sampling rate thay đổi. Nó bổ sung 3D smoothing filter và 2D Mip filter để giảm aliasing và dilation artifacts. Đây là lý do nó là ứng viên rất mạnh cho ảnh drone có khoảng cách camera thay đổi.

Giai đoạn 5 — Chọn thuật toán theo bằng chứng

Không nên chọn thuật toán theo leaderboard của paper. Hãy chọn theo failure mode của dữ liệu.

5.1 Thuật toán 1: 3DGS
Dùng khi
Test poses chủ yếu interpolation.
Camera calibration tốt.
Scene coverage dày.
Cần baseline nhanh.
Muốn thực hiện nhiều thí nghiệm.
Ưu điểm
Train nhanh.
Render nhanh.
Repo ổn định.
Dễ debug.
Tận dụng trực tiếp sparse points.
Hạn chế
Có thể overfit train views.
Geometry không thực sự là bề mặt nhất quán.
Dễ floaters.
Scale/focal shift gây artifacts.
5.2 Thuật toán 2: Mip-Splatting
Dùng khi
Train/test có khoảng cách camera khác nhau.
Focal length hoặc resolution thay đổi.
BTS có nhiều pattern tần số cao.
Xuất hiện aliasing trên dây, lưới, thanh kim loại.
Vì sao ưu tiên

Cuộc thi cung cấp riêng fx, fy, width, height cho từng target pose. Điều này cho thấy sampling configuration có thể thay đổi giữa các view. Mip-Splatting giải quyết đúng failure mode đó.

Vai trò trong pipeline

Đây nên là main candidate, không chỉ là ablation phụ.

5.3 Thuật toán 3: Scaffold-GS

Scaffold-GS sử dụng anchor points và sinh thuộc tính Gaussian theo viewing direction và distance, đồng thời có cơ chế anchor growing/pruning. Phương pháp hướng tới giảm Gaussian dư thừa và cải thiện robustness ở góc nhìn khó, vùng ít texture và các hiệu ứng phụ thuộc hướng nhìn.

Dùng khi
Validation block-angle kém.
View-dependent appearance mạnh.
Metal/reflection.
Texture-less areas.
Model 3DGS có nhiều Gaussian nhưng vẫn extrapolate kém.
Rủi ro
Pipeline phức tạp hơn.
Hyperparameters khác baseline.
Có thể không thắng Mip-Splatting ở pixel metrics.
Cần nhiều thời gian tuning.
5.4 Thuật toán 4: 2D Gaussian Splatting

2DGS biểu diễn scene bằng các Gaussian disk phẳng thay vì ellipsoid 3D, đồng thời sử dụng ray-splat intersection, depth distortion và normal consistency để tăng tính nhất quán hình học.

Dùng khi
Dây, mặt phẳng và cạnh hình học bị sai.
Floaters nghiêm trọng.
Muốn surface geometry ổn định hơn.
Góc nhìn validation cho thấy 3DGS “vỡ hình”.
Không nên mặc định cho rằng 2DGS sẽ có score RGB cao nhất

“Geometry chính xác hơn” không đồng nghĩa tự động với:

PSNR cao hơn.
LPIPS thấp hơn.
Tốc độ train tốt hơn.

Đây là ứng viên cần được kiểm chứng bằng validation.

6. Architecture hệ thống nên xây

Đừng gộp toàn bộ vào một script 2.000 dòng.

                      ┌──────────────────────┐
                      │  Competition dataset │
                      └──────────┬───────────┘
                                 │
                    ┌────────────▼────────────┐
                    │ Dataset discovery layer │
                    └────────────┬────────────┘
                                 │
                 ┌───────────────▼────────────────┐
                 │ COLMAP + camera normalization  │
                 └───────────────┬────────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │ Scene inspector and quality report │
              └──────────────────┬──────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │ Scene classification    │
                    │ scale/view/data quality │
                    └────────────┬────────────┘
                                 │
             ┌───────────────────▼───────────────────┐
             │ Model/config policy                   │
             │ 3DGS / Mip / Scaffold / 2DGS          │
             └───────────────────┬───────────────────┘
                                 │
                  ┌──────────────▼───────────────┐
                  │ Training and checkpointing  │
                  └──────────────┬───────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │ Validation and ranking │
                    └────────────┬────────────┘
                                 │
                  ┌──────────────▼──────────────┐
                  │ Render official test poses │
                  └──────────────┬──────────────┘
                                 │
                   ┌─────────────▼─────────────┐
                   │ Submission validator     │
                   └─────────────┬─────────────┘
                                 │
                      submission.zip
7. Cấu trúc repository đề xuất
digital-twin-nvs/
│
├── README.md
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── Makefile
│
├── configs/
│   ├── data/
│   │   └── competition.yaml
│   ├── model/
│   │   ├── 3dgs.yaml
│   │   ├── mip_splatting.yaml
│   │   ├── scaffold_gs.yaml
│   │   └── 2dgs.yaml
│   ├── experiment/
│   │   ├── baseline.yaml
│   │   ├── high_resolution.yaml
│   │   └── pose_refinement.yaml
│   └── competition/
│       ├── phase1_fast.yaml
│       └── phase1_final.yaml
│
├── src/
│   └── nvs/
│       ├── data/
│       │   ├── discovery.py
│       │   ├── scene.py
│       │   ├── colmap_io.py
│       │   ├── cameras.py
│       │   ├── test_poses.py
│       │   ├── validation_split.py
│       │   └── preprocessing.py
│       │
│       ├── geometry/
│       │   ├── transforms.py
│       │   ├── quaternion.py
│       │   ├── projection.py
│       │   └── normalization.py
│       │
│       ├── models/
│       │   ├── base.py
│       │   ├── gaussian_splatting/
│       │   ├── mip_splatting/
│       │   ├── scaffold_gs/
│       │   └── two_dgs/
│       │
│       ├── training/
│       │   ├── trainer.py
│       │   ├── losses.py
│       │   ├── schedules.py
│       │   ├── checkpoint.py
│       │   └── callbacks.py
│       │
│       ├── evaluation/
│       │   ├── psnr.py
│       │   ├── ssim.py
│       │   ├── lpips.py
│       │   ├── competition_score.py
│       │   └── evaluator.py
│       │
│       ├── rendering/
│       │   ├── renderer.py
│       │   ├── test_renderer.py
│       │   └── image_writer.py
│       │
│       ├── analysis/
│       │   ├── scene_inspector.py
│       │   ├── camera_plots.py
│       │   ├── error_maps.py
│       │   └── report.py
│       │
│       ├── selection/
│       │   ├── scene_classifier.py
│       │   ├── config_policy.py
│       │   └── checkpoint_ranker.py
│       │
│       └── submission/
│           ├── builder.py
│           └── validator.py
│
├── tools/
│   ├── inspect_dataset.py
│   ├── create_val_split.py
│   ├── train_scene.py
│   ├── train_all.py
│   ├── evaluate_scene.py
│   ├── render_test.py
│   ├── build_submission.py
│   └── run_pipeline.py
│
├── notebooks/
│   ├── kaggle_train.ipynb
│   └── exploratory_analysis.ipynb
│
├── tests/
│   ├── test_camera_conversion.py
│   ├── test_colmap_io.py
│   ├── test_projection.py
│   ├── test_test_poses.py
│   ├── test_metrics.py
│   └── test_submission.py
│
└── docs/
    ├── problem_analysis.md
    ├── camera_conventions.md
    ├── experiment_protocol.md
    └── results.md

Không cần hoàn thiện toàn bộ ngay. Thứ tự triển khai:

data → geometry → baseline model → evaluation
→ rendering → submission → analysis → model variants
8. Interface chung cho nhiều thuật toán

Bạn không nên để code gọi trực tiếp từng repository theo cách khác nhau.

Tạo abstraction:

class NVSModel:
    def fit(self, scene, config) -> None:
        ...

    def render(self, camera):
        ...

    def save(self, path) -> None:
        ...

    @classmethod
    def load(cls, path):
        ...

Mỗi backend implement cùng interface:

class GaussianSplattingModel(NVSModel):
    ...

class MipSplattingModel(NVSModel):
    ...

class ScaffoldGSModel(NVSModel):
    ...

class TwoDGSModel(NVSModel):
    ...

Như vậy pipeline bên ngoài không đổi:

model = registry.create(config.model.name)
model.fit(scene, config)
predictions = renderer.render_poses(model, test_cameras)

Đây là yếu tố giúp bạn thử nhiều model mà không phá pipeline.

9. Thiết kế experiment có kỷ luật
9.1 Không thay nhiều thứ cùng lúc

Ví dụ sai:

Experiment 1:
3DGS, 15k steps, half resolution

Experiment 2:
Mip-Splatting, 40k steps, full resolution,
pose optimization, different split, different seed

Bạn không thể biết cải thiện đến từ đâu.

9.2 Ma trận thí nghiệm
Nhóm A — Xác minh pipeline
ID	Model	Resolution	Steps	Mục tiêu
A0	3DGS	1/4	3k	Pipeline chạy đúng
A1	3DGS	1/2	15k	Baseline nhanh
A2	3DGS	Full	30k	Baseline chính
Nhóm B — Data/camera
ID	Thay đổi
B1	Loại ảnh blur/outlier
B2	Exposure normalization
B3	Pose refinement
B4	Intrinsics refinement
B5	Scene normalization khác
Nhóm C — Optimization
ID	Thay đổi
C1	Densification schedule
C2	Opacity reset
C3	Scale threshold
C4	SH degree
C5	Learning rates
C6	Training steps
Nhóm D — Model family
ID	Model
D0	3DGS
D1	Mip-Splatting
D2	Scaffold-GS
D3	2DGS
Nhóm E — Final
ID	Mục tiêu
E1	Best single checkpoint
E2	Best checkpoint theo scene
E3	Multi-seed model selection
E4	Test-time crop/render check
E5	Submission integrity
10. Scene-adaptive strategy: yếu tố có thể tạo khác biệt lớn

Không nhất thiết mỗi scene dùng cùng một config.

Scene profile
@dataclass
class SceneProfile:
    num_images: int
    point_count: int
    camera_radius_mean: float
    camera_radius_std: float
    focal_variation: float
    elevation_coverage: float
    angular_coverage: float
    test_extrapolation_score: float
    brightness_variation: float
    sparse_density: float
Rule-based policy ban đầu
if focal_variation > threshold or radius_variation > threshold:
    model = "mip_splatting"

elif test_extrapolation_score > threshold:
    model = "scaffold_gs"

elif geometry_is_thin_and_floaters_are_high:
    model = "2dgs"

else:
    model = "3dgs"

Sau khi có đủ experiments, policy có thể dựa vào validation:

best_config = argmax(
    config.validation_score
    - uncertainty_penalty
    - failure_rate_penalty
)

Đây chưa phải meta-learning phức tạp. Nó chỉ là configuration selection có hệ thống.

11. Những cải tiến nên thử theo thứ tự
Tier 1 — Gần như bắt buộc
Camera convention chính xác.
Scene normalization ổn định.
Local validation tốt.
Exposure/color handling.
Mip-Splatting.
Full-resolution final training.
Submission validator.
Per-scene config selection.
Tier 2 — Có khả năng tăng điểm đáng kể
Pose refinement nhỏ.
Intrinsics refinement nhỏ.
Multi-scale training.
Better densification/pruning.
Scaffold-GS.
2DGS.
Mask vùng vô ích nếu luật cho phép và mask được sinh tự động.
Per-image appearance embeddings có regularization.
Tier 3 — Chỉ làm khi pipeline đã rất chắc
Model blending.
Multi-seed checkpoint selection.
Region-aware loss.
Depth/normal regularization.
Confidence-aware rendering.
Automatic hyperparameter policy.
Custom CUDA optimization.
12. Vấn đề về pretrained model và dữ liệu ngoài

Quy định “chỉ sử dụng dữ liệu do ban tổ chức cung cấp” có thể được hiểu theo hai cách:

Chỉ cấm ảnh/scene bên ngoài liên quan tới test data.
Cấm cả pretrained weights học từ dữ liệu ngoài.

Không nên tự suy diễn.

Trước khi dùng:

Pretrained LPIPS trong training loss.
Monocular depth model.
Segmentation model.
Foundation model.
Super-resolution model.
Diffusion prior.
DINO/CLIP features.

Bạn cần hỏi ban tổ chức rõ:

Có được phép sử dụng pretrained weights được huấn luyện trên dữ liệu công khai, nhưng không chứa scene của cuộc thi, hay không?

Metrics LPIPS do hệ thống chấm sử dụng không đồng nghĩa bạn được phép dùng pretrained LPIPS trong training.

Pipeline an toàn nhất là:

Representation được tối ưu riêng trên từng scene.
Không dùng ảnh ngoài.
Không dùng model sinh ảnh pretrained.
Chỉ sử dụng mã nguồn thuật toán công khai.
13. Những hướng có vẻ hấp dẫn nhưng chưa nên làm
Super-resolution

Có thể tạo texture đẹp hơn nhưng:

Sai pixel.
Hallucinate cạnh.
Thay đổi thiết bị nhỏ.
PSNR giảm.

Chỉ thử sau khi render geometry đã tốt.

Diffusion refinement

Rất nguy hiểm vì:

Có thể tạo dây hoặc anten không tồn tại.
Thay đổi logo, chi tiết kỹ thuật.
Không giữ multi-view consistency.
Có thể vi phạm quy định pretrained data.
NeRF từ đầu

Không phải vô dụng, nhưng chi phí opportunity cost cao:

Chậm iterate.
Khó train nhiều scene/config.
Khó sweep.
Submission deadline gần.
Tự viết renderer CUDA ngay

Không nên. Hãy chứng minh bottleneck trước.

14. Kế hoạch cụ thể từ ngày 23/07 đến 30/07/2026

Phase 1 kết thúc ngày 30/07/2026, nên hiện tại phải ưu tiên một pipeline thi đấu được trước, sau đó mới nghiên cứu sâu hơn.

23/07 — Pipeline và data audit

Hoàn thành:

Repo skeleton.
COLMAP reader.
Test CSV reader.
Camera conversion tests.
Scene discovery.
Dataset report.
Kaggle notebook clone và install.
Baseline train một scene ở resolution thấp.

Kết quả cuối ngày:

One command → one trained scene → rendered images
24/07 — Baseline chính xác

Hoàn thành:

Official 3DGS baseline.
Local interpolation split.
Metrics giống công thức cuộc thi.
Error maps.
Submission validator.
Full end-to-end dry run.

Kết quả:

Submission hợp lệ đầu tiên.
Điểm local baseline.
25/07 — Validation và Mip-Splatting

Hoàn thành:

Angular-block split.
Distance/focal split.
Mip-Splatting integration.
So sánh 3DGS và Mip-Splatting cùng điều kiện.

Kết quả:

Xác định scale shift có phải bottleneck hay không.
26/07 — Camera và appearance

Thử có kiểm soát:

Exposure affine.
Pose refinement nhỏ.
Intrinsics refinement nếu có bằng chứng.
Outlier image filtering.
Densification tuning.

Kết quả:

Một bảng ablation rõ cải tiến đến từ đâu.
27/07 — Model alternatives

Chọn một trong hai, không nên làm cả hai quá sâu:

Scaffold-GS nếu extrapolation/view dependence là lỗi chính.
2DGS nếu floaters/thin geometry là lỗi chính.

Kết quả:

Ít nhất hai model family được đánh giá công bằng.
28/07 — Scene-adaptive selection

Hoàn thành:

Scene profiling.
Chọn model/config theo scene.
Train full-resolution cho các scene.
Resume checkpoint.
Tự động xử lý lỗi scene.
29/07 — Final training
Train nhiều seed chỉ cho config tốt nhất.
Chọn checkpoint bằng local validation.
Render test poses.
Kiểm tra kích thước, tên file và số lượng.
Tạo submission cuối.
Lưu logs/config/commit hash.
30/07 — Không nghiên cứu lớn nữa

Chỉ:

Reproduce.
Validate.
Submit.
Kiểm tra submission.
Chuẩn bị fallback ZIP.
Không đổi camera conversion hoặc refactor lớn.
15. Định nghĩa “architecture và spec tốt nhất”

Bạn cần viết một system_spec.md trả lời được các câu hỏi sau.

Problem specification
Input chính xác là gì?
Output chính xác là gì?
Camera convention nào?
Test poses có nằm trong train distribution không?
Đâu là failure modes quan trọng?
Data specification
Scene discovery.
Image formats.
Intrinsic/extrinsic schema.
Coordinate transformations.
Normalization.
Invalid-data policy.
Outlier policy.
Model specification
Representation.
Initialization.
Renderer.
Losses.
Densification.
Pruning.
Optimization schedule.
Appearance model.
Pose/intrinsic refinement.
Checkpoint policy.
Evaluation specification
Validation split.
Metric implementation.
Aggregation.
Per-scene selection.
Seed policy.
Error analysis.
Runtime specification
GPU assumptions.
VRAM fallback.
Timeout handling.
Resume.
Logging.
Determinism.
Failure recovery.
Submission specification
Directory naming.
File naming.
Image mode.
Image dimensions.
Missing/extra file detection.
ZIP validation.
16. Mục tiêu milestone
Milestone 1 — Correctness

Pipeline tạo được submission đúng định dạng.

Milestone 2 — Reliable baseline

3DGS có local metrics và chạy lại được.

Milestone 3 — Meaningful validation

Local ranking giữa các config có khả năng tương quan với leaderboard.

Milestone 4 — Targeted optimization

Mỗi thay đổi đều giải quyết một failure mode cụ thể.

Milestone 5 — Scene adaptation

Không ép tất cả scene dùng cùng model/config.

Milestone 6 — Competition reliability

Pipeline có resume, logs, validation và fallback.

17. Thứ tự triển khai tôi khuyến nghị
1. Camera parser + unit tests
2. Dataset inspector
3. 3DGS baseline
4. Validation splits
5. Competition metrics
6. Submission validator
7. Mip-Splatting
8. Exposure and pose refinement
9. Densification tuning
10. Scaffold-GS hoặc 2DGS
11. Scene-adaptive config selection
12. Multi-seed final training

Điểm quan trọng nhất là:

Đừng xây “model tốt nhất” trước. Hãy xây một experimental system có khả năng chứng minh model nào tốt nhất cho chính dữ liệu cuộc thi.

Với thời gian Phase 1 còn lại, hướng có xác suất thành công cao nhất là:

Verified cameras+3DGS baseline+Mip-Splatting+good validation+per-scene selection
	​


thay vì cố tích hợp nhiều paper nhưng không biết cải tiến nào thực sự làm tăng score.+

---

# Execution order

## Setup một lần

1. Chuẩn hóa repository và cấu trúc thư mục: `src/`, `tools/`, `configs/`, `tests/`, `third_party/`.
2. Chốt environment: Python, PyTorch/CUDA, Graphdeco/3DGS, dependencies và script cài đặt.
3. Chốt data contract và submission contract; viết config loader, logging, seed và lưu resolved config.
4. Viết smoke-test dataset/config nhỏ để kiểm tra pipeline mà không cần chạy full dataset.

## Làm tuần tự, chỉ chuyển bước khi bước trước có bằng chứng

5. Data audit và camera validation; tạo report cho từng scene và unit tests cho pose conversion.
6. Hoàn thiện một scene end-to-end: discover data → train → render train/validation pose → tính metrics → validate output.
7. Tạo local validation splits A–D; lưu per-view metrics và phân loại pose.
8. Chạy baseline 3DGS cố định; lưu checkpoint, config, log, VRAM/time và kết quả.
9. Phân tích lỗi baseline trước khi sửa model; xác định failure mode chính bằng report và ảnh worst views.
10. Thử từng cải tiến độc lập: camera/exposure trước, sau đó resolution/densification, rồi Mip-Splatting/Scaffold-GS/2DGS tùy failure mode.
11. So sánh bằng cùng split, seed và metric; chọn config/model theo bằng chứng local.
12. Chạy nhiều scene, chọn config theo scene nếu dữ liệu chứng minh cần thiết; sau đó render test poses.
13. Validate toàn bộ submission: đủ scene, đủ ảnh, đúng tên, RGB, kích thước, không thừa/thiếu file; tạo `submission.zip`.
14. Chạy lại từ clean environment trên Kaggle; ghi commit, config, dependency và artifact để chứng minh tái lập.

## Cập nhật plan sau mỗi bước

Mỗi bước phải ghi: trạng thái, command đã chạy, artifact tạo ra, lỗi phát hiện, quyết định tiếp theo và tiêu chí pass/fail. Không chuyển sang tối ưu model khi baseline chưa render và validate được một scene. Không chạy full dataset trước khi smoke test và single-scene run thành công.

