# Long-horizon manipulation trong VLA — tổng hợp 8 paper còn lưu

## Ý tưởng chính

Corpus hiện tại gồm **tám PDF gốc** và tám báo cáo tương ứng. Chúng không trả lời
cùng một câu hỏi: hai paper xử lý phân rã task, ba paper duy trì trạng thái thời
gian, và ba paper dùng dự báo tương lai hoặc guidance trong action space để điều
khiển. Vì benchmark, robot và giao thức khác nhau, không xếp hạng trực tiếp các
con số success rate giữa chúng.

**Verified:** phạm vi này chỉ gồm các file đang có trong
[`docs/papers/05-long-horizon/`](../../papers/05-long-horizon/). Chỉ mục nguồn
chuẩn là [paper_link.txt](../../papers/05-long-horizon/paper_link.txt); các repo
công khai nằm ở [02_sota_co_code.md](02_sota_co_code.md).

## 1. Câu hỏi nghiên cứu

Khi manipulation kéo dài thất bại, thiếu phần nào của policy: phân rã mục tiêu,
trạng thái lịch sử, hay tín hiệu dự báo hành động/trạng thái tương lai?

## 2. Chỉ mục theo cơ chế

| Nhóm | Vấn đề chính | Paper |
|---|---|---|
| [hierarchical_agent/](hierarchical_agent/) | Suy ra subtask và tiếp nhận feedback ở tầng cao | [π0.5](hierarchical_agent/01_pi0_5.md), [Hi Robot](hierarchical_agent/02_hi_robot.md) |
| [memory_modules/](memory_modules/) | Giữ thông tin lịch sử khi quan sát hiện tại bị nhập nhằng | [MemoryVLA](memory_modules/01_memoryvla.md), [SeedPolicy](memory_modules/02_seedpolicy.md), [ReMem-VLA](memory_modules/03_remem_vla.md) |
| [future_prediction/](future_prediction/) | Dùng trạng thái, video hoặc action tương lai để điều kiện policy | [Seer](future_prediction/01_seer.md), [LingBot-VA](future_prediction/02_lingbot_va.md), [ACoT-VLA](future_prediction/03_acot_vla.md) |

ACoT-VLA dự báo chuỗi action thay vì ảnh/video. Nó vẫn ở nhóm `future_prediction/`
vì EAR tạo một reference action tương lai để điều kiện action head; đây là quyết
định phân loại, không phải taxonomy của tác giả.

## 3. So sánh tối thiểu

| Paper | Can thiệp | Tín hiệu/dữ liệu bổ sung | Bằng chứng chính trong báo cáo |
|---|---|---|---|
| π0.5 | Một VLA hai tầng, co-training không đồng nhất | Subtask semantic, action, dữ liệu web/robot | Task gia đình 10–15 phút trong nhà chưa thấy khi train |
| Hi Robot | Hai VLM tách rời, high-level gọi lại theo feedback | Prompt và utterance tổng hợp | Ba robot thật, lệnh mở và sửa giữa chừng |
| MemoryVLA | Perceptual-cognitive memory bank | Không cần nhãn mới ngoài demo có ảnh/lệnh | Tăng rõ ở các task temporal |
| SeedPolicy | Trạng thái ẩn đệ quy với cổng attention | Không cần nhãn mới ngoài demo | So sánh memory mechanism trên cùng Diffusion Policy |
| ReMem-VLA | Frame/chunk recurrent queries với fixed EMA + past-observation prediction | Demo tuần tự theo episode; auxiliary target là ảnh quá khứ | MemoryBench mở rộng và 4 task memory trên UR5 |
| Seer | Predictive inverse dynamics end-to-end | Play data có thể thiếu language label | LIBERO-LONG, CALVIN và robot thật |
| LingBot-VA | World model video-action tự hồi quy | Không cần nhãn phụ ngoài dữ liệu huấn luyện mô hình | RoboTwin, LIBERO và task temporal thật |
| ACoT-VLA | Explicit/implicit action reasoner trên π0.5 | Demo action là target cho EAR | LIBERO, LIBERO-Plus, VLABench và robot thật |

## 4. Điều có thể kết luận từ tám paper

1. **Phân rã task và memory là hai can thiệp khác nhau.** π0.5/Hi Robot tạo hoặc
   cập nhật bước cần làm; MemoryVLA/SeedPolicy giữ ngữ cảnh để cùng một quan sát
   không bị hiểu như cùng một trạng thái. **Inferred:** một hệ thực tế có thể cần
   cả hai, nhưng corpus này không có thí nghiệm ghép trực tiếp.
2. **Dự báo tương lai có nhiều modality.** Seer dự báo latent của ảnh tương lai,
   LingBot-VA mô hình hoá video và action, còn ACoT-VLA sinh reference action.
   Không nên gọi chúng là cùng một method chỉ vì đều có foresight.
3. **Code công khai không đồng nghĩa tái lập toàn bộ.** Năm paper có repo công
   khai, nhưng ACoT-VLA cần nền π0.5 và hạ tầng lớn; LingBot-VA có checkpoint nhưng
   pretraining quy mô lớn. Chi tiết capability/unknown nằm ở
   [02_sota_co_code.md](02_sota_co_code.md).
4. **Memory có thể là retrieval bank hoặc recurrent latent state.** MemoryVLA truy xuất
   bank hữu hạn; SeedPolicy và ReMem-VLA truyền state ẩn theo thời gian. ReMem-VLA thêm hai
   tốc độ update và auxiliary visual reconstruction. **Unknown:** chưa có so sánh ba cơ chế
   dưới cùng backbone, dataset và compute budget.

## 5. Giới hạn của tổng hợp

- **Unknown:** chưa có benchmark chung để so trực tiếp hierarchy, memory và future
  prediction trên cùng robot, dữ liệu và ngân sách compute.
- Các thành công được báo cáo không phải kết quả đã tái lập trong workspace.
- `vla-data-tools` hiện chỉ đọc/chuyển đổi dataset; chưa có training loop hay robot
  execution. Vì vậy mọi đề xuất model bên dưới là **Planned**, không phải capability
  hiện có.

## 6. Bước kiểm chứng tiếp theo

1. **Planned:** dùng một dataset nhỏ đã inspect để kiểm tra có đủ history,
   language và modality cho một baseline Seer hoặc SeedPolicy hay không.
2. **Planned:** nếu đo ACoT-VLA, tái đo latency theo cùng phần cứng/cấu hình và so
   với số paper báo cáo (baseline 91 ms, thêm EAR 110 ms, IAR thêm khoảng 2 ms).
3. **Planned:** tách benchmark theo failure mode trước khi chọn hướng: task cần
   feedback/subtask, state aliasing, hay dự báo chuyển động.

## Nguồn

- [Chỉ mục PDF và metadata](../../papers/05-long-horizon/paper_link.txt)
- Bảy báo cáo theo thư mục ở mục 2; mỗi báo cáo liên kết trực tiếp tới PDF nguồn.
