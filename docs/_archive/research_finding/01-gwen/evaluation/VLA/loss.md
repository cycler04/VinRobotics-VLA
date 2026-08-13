# Các hàm mất mát được dùng trong những mô hình Thị giác-Ngôn ngữ-Hành động Qwen

> **Câu hỏi:** Qwen-VLA và các mô hình Qwen chuyên biệt cho robot huấn luyện
> hành động, ngôn ngữ, chính sách và video tương lai như thế nào?
>
> **Phạm vi:** Các mục tiêu đã công bố trong Qwen-VLA, Qwen-RobotManip,
> Qwen-RobotNav và Qwen-RobotWorld. Nghiên cứu được kiểm tra ngày 2026-07-21.

## Câu trả lời ngắn

Các mô hình này không dùng chung một “hàm mất mát VLA” có thể thay thế lẫn nhau:

| Mô hình/giai đoạn | Đầu ra được học | Mục tiêu chính đã công bố |
|---|---|---|
| Tiền huấn luyện/SFT Qwen-VLA | các đoạn hành động + văn bản | MSE khớp dòng có mặt nạ + NLL từ tố kế tiếp |
| RL Qwen-VLA | chính sách hành động vòng kín + giá trị | mục tiêu tác tử cắt ngưỡng PPO + MSE giá trị cắt ngưỡng |
| Qwen-RobotManip | các đoạn hành động liên tục + văn bản | khớp dòng có mặt nạ + NLL từ tố kế tiếp; mặc định SFT chỉ dùng hành động |
| Qwen-RobotNav | tám điểm mốc 3D + văn bản | MSE quỹ đạo + NLL từ tố kế tiếp |
| Qwen-RobotWorld | các biến tiềm ẩn của video tương lai | khớp dòng có điều kiện với các mặt nạ điều kiện |

Thang đo mất mát phụ thuộc vào biểu diễn hành động, đường chân trời, phân phối bước thời
gian, mặt nạ và phép rút gọn. Nó phải được xem cùng với các định nghĩa điểm số trong
[metrics.md](metrics.md) và giao thức đánh giá trong
[benchmarks.md](benchmarks.md).

## Khớp dòng hành động của Qwen-VLA

Gọi đoạn hành động sạch là $Y_0\in\mathbb{R}^{H\times K}$ và lấy mẫu nhiễu Gauss
$Y_1\sim\mathcal{N}(0,I)$. Tại thời điểm nhiễu $\tau$, quá trình huấn luyện dựng

$$
Y_\tau=(1-\tau)Y_0+\tau Y_1,
$$

và yêu cầu chuyên gia hành động dự đoán vận tốc không đổi $Y_1-Y_0$.

Vì các tập dữ liệu có số chiều hành động khác nhau và phiên có thể kết thúc ở giữa một
đoạn, Qwen-VLA dùng mặt nạ $M_{h,k}$ và trước tiên lấy trung bình trên các bước thời
gian hợp lệ cho từng kênh hành động:

$$
\ell_k=
\frac{\sum_h M_{h,k}\left\|
v_\theta(Y_\tau,\tau,o,x)_{h,k}-(Y_1-Y_0)_{h,k}
\right\|_2^2}
{\sum_h M_{h,k}}.
$$

Sau đó, mô hình gán trọng số bằng nhau cho từng kênh trong số $c$ kênh đang hoạt động:

$$
\mathcal{L}_{\mathrm{act}}
=\mathbb{E}\left[\frac{1}{c}\sum_{k<c}\ell_k\right].
$$

Phép rút gọn hai cấp này rất quan trọng. Phần đệm không có gradien, các đoạn hợp lệ
dài hơn không tự động nhận trọng số lớn hơn, và một hiện thân có số bậc tự do cao không
chiếm ưu thế chỉ vì có nhiều kênh vô hướng đang hoạt động hơn. Phép rút gọn này **không**
làm cho đơn vị của các kênh tương đương nhau về ngữ nghĩa; phép chuẩn hóa và định nghĩa
hành động riêng cho từng tập dữ liệu vẫn có ý nghĩa. [Qwen-VLA, Mục 2.4–2.5][qwen-vla]

## Huấn luyện chung thị giác-ngôn ngữ và hành động

Mục tiêu ngôn ngữ phụ trợ là âm log hợp lý của từ tố kế tiếp:

$$
\mathcal{L}_{\mathrm{vl}}
=-\sum_i \log p_\theta(w_i\mid w_{<i},o_{1:t}),
$$

và mục tiêu chung là

$$
\mathcal{L}
=\lambda_{\mathrm{act}}\mathcal{L}_{\mathrm{act}}
+\lambda_{\mathrm{vl}}\mathcal{L}_{\mathrm{vl}}.
$$

Các giai đoạn sử dụng mục tiêu này theo những cách khác nhau:

| Giai đoạn | Thành phần có thể huấn luyện và tín hiệu | Quy tắc trọng số/nhiễu đã công bố |
|---|---|---|
| Căn chỉnh văn bản-hành động | Chuyên gia hành động; lời nhắc văn bản/hiện thân điều kiện hóa phép khớp dòng | VLM bị đóng băng; lấy mẫu thời gian Sigmoid-Normal |
| Tiền huấn luyện liên tục | VLM và chuyên gia hành động; dữ liệu VL và hành động được đồng huấn luyện | các trọng số được điều chỉnh để cân bằng độ lớn gradien; không công bố giá trị số; lấy mẫu thời gian Beta |
| Tinh chỉnh có giám sát | kết hợp các mục tiêu thao tác, điều hướng và VL | VL 0.1; hành động thao tác 1.0; hành động điều hướng 1.0; lấy mẫu thời gian Beta |

Trong SFT, thao tác dùng đường chân trời hành động 16 còn điều hướng dùng đường chân trời
8. Các thí nghiệm loại trừ của bài báo cho thấy phân phối bước thời gian không phải là một
chi tiết mang tính hình thức: Sigmoid-Normal hoạt động tốt nhất cho căn chỉnh văn bản-hành
động, còn Beta hoạt động tốt nhất cho SFT. [Qwen-VLA, Mục 3–5.2][qwen-vla]

## Học tăng cường Qwen-VLA

Qwen-VLA bắt đầu RL từ điểm kiểm tra SFT đa nhiệm và áp dụng PPO. Với tỷ số xác suất

$$
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)},
$$

mục tiêu tác tử cắt ngưỡng là

$$
\mathcal{L}_{\mathrm{actor}}
=-\mathbb{E}\left[
\min\left(r_t\hat A_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t\right)
\right].
$$

Tổng mục tiêu được công bố là

$$
\mathcal{L}_{\mathrm{RL}}
=\mathcal{L}_{\mathrm{actor}}+c_v\mathcal{L}_{\mathrm{value}},
$$

trong đó bộ phê bình sử dụng MSE cắt ngưỡng. Các thiết lập được báo cáo gồm
$\epsilon=0.2$, hệ số chiết khấu $\gamma=0.99$, GAE $\lambda=0.95$, $c_v=1$, bốn
vòng huấn luyện PPO, tốc độ học của tác tử $5\times10^{-6}$ và tốc độ học của đầu giá trị
$10^{-4}$. Các đặc trưng VLM là đầu vào ngắt gradien cho đầu giá trị.

Khớp dòng không trực tiếp cung cấp một log xác suất hành động có thể tính toán thuận lợi. Bài báo chuyển
ODE dòng xác suất thành một SDE với các chuyển tiếp Gauss và tính lại log xác suất tại
một bước khử nhiễu được lấy mẫu. Phần thưởng thưa và nhị phân—chỉ bằng 1 khi phiên
thành công, ngược lại bằng 0—và không sử dụng mô hình phần thưởng được học. Phần thưởng
và lợi thế hoạt động ở cấp đoạn hành động với $H=16$. Phương trình tổng đã công bố không
thể hiện hạng phạt entropy hoặc KL, vì vậy không nên thêm các hạng đó vào mục tiêu được
ghi nhận. [Qwen-VLA, Mục 4.2][qwen-vla]

SFT mang lại phần lớn mức cải thiện hạ nguồn được báo cáo; PPO bổ sung một mức tăng nhỏ
hơn trên miền triển khai SimplerEnv của nó và chủ yếu chỉ tạo ra những thay đổi nhỏ ở nơi
khác. Đây là lời nhắc rằng một hàm mất mát tinh vi tự nó không đồng nghĩa với mức cải
thiện hành vi lớn.

## Qwen-RobotManip

Qwen-RobotManip thay đổi hướng nội suy. Với hành động sạch $a$, nhiễu
$\epsilon\sim\mathcal{N}(0,I)$ và $t\sim\mathrm{Beta}(1,1.5)$:

$$
x_t=(1-t)\epsilon+ta,
\qquad
u=a-\epsilon.
$$

Mô hình hồi quy $u$ bằng sai số bình phương. Phép rút gọn có mặt nạ được công bố lấy
trung bình trên các phần tử hợp lệ của từng mẫu, rồi lấy trung bình trên lô. Mặt nạ
hợp lệ kết hợp tính hợp lệ của chiều hành động, tính hợp lệ của bước thời gian/ranh giới
phiên và khả năng quan sát của từng tay. Cách này khác với phép rút gọn gán trọng số
bằng nhau cho từng kênh của Qwen-VLA, do đó không thể so sánh các giá trị mất mát hành
động thô giữa hai bài báo.

Trong quá trình tiền huấn luyện, RobotManip kết hợp hàm mất mát này với hợp lý của từ tố
kế tiếp:

$$
\mathcal{L}=\mathcal{L}_{\mathrm{FM}}
+0.1\,\mathcal{L}_{\mathrm{VLM}}.
$$

Quá trình tiền huấn luyện của mô hình lấy mẫu dữ liệu VLA và VL theo tỷ lệ 9:1 và lặp
lại mỗi mẫu hành động với tám cặp nhiễu/bước thời gian được lấy mẫu độc lập. SFT dành
riêng cho từng tác vụ mặc định chỉ dùng khớp dòng, không có hạng VLM. Báo cáo không ghi
nhận giai đoạn RL, ưu tiên hoặc chưng cất. [Qwen-RobotManip,
Mục 3–4][robotmanip]

Một kết quả âm quan trọng xuất hiện trong thí nghiệm loại trừ ngữ cảnh: một cấu hình có
thể đạt mất mát huấn luyện thấp bằng cách sao chép các hành động gần đây nhưng lại đạt
kết quả kém về mức độ thành công của tác vụ. Vì vậy, mất mát tối ưu hóa và năng lực vòng
kín phải được báo cáo riêng.

## Qwen-RobotNav

Qwen-RobotNav dự đoán tám điểm mốc, mỗi điểm mốc có ba tọa độ. Hạng quỹ đạo của mô hình
là phép hồi quy trực tiếp:

$$
\mathcal{L}_{\mathrm{traj}}
=\left\|\hat W-W^*\right\|_2^2,
\qquad
\mathcal{L}=\mathcal{L}_{\mathrm{traj}}
+\lambda\mathcal{L}_{\mathrm{VL}},
\quad \lambda=1.0.
$$

Hạng quỹ đạo chỉ hoạt động đối với các mẫu điều hướng; hạng VL là mục tiêu từ tố kế
tiếp tiêu chuẩn. Quá trình huấn luyện trộn 85% dữ liệu quỹ đạo với 15% dữ liệu VL liên
quan đến điều hướng. Các tọa độ được chuẩn hóa riêng cho từng tập dữ liệu bằng phân vị thứ
99 và ánh xạ vào $[-1,1]$, vì vậy một giá trị MSE dạng số không có đơn vị khoảng cách
phổ quát. Báo cáo không mô tả các hàm mất mát khớp dòng, RL, ưu tiên hoặc chưng cất.
[Qwen-RobotNav, Mục 2.5–2.6][robotnav]

## Qwen-RobotWorld

Qwen-RobotWorld là mô hình video tương lai, không phải chính sách hành động. Mô hình mã
hóa video bằng VAE, làm nhiễu biến tiềm ẩn bằng nhiễu Gauss và học khớp dòng có điều kiện.
Thời điểm nhiễu tuân theo phân phối log-chuẩn với một độ dịch được điều chỉnh theo độ dài
chuỗi. Một bộ mã hóa Qwen2.5-VL bị đóng băng cung cấp chỉ dẫn ngôn ngữ/hành động.

Các khung hình điều kiện bị loại khỏi hàm mất mát khử nhiễu:

- trong tác vụ văn bản-hình ảnh-sang-video, biến tiềm ẩn của khung hình đầu tiên được cố
  định tại $t=0$;
- trong Scene2Robot, cả điều kiện cảnh và phân đoạn tham chiếu robot đều được cố định tại
  $t=0$;
- chỉ phân đoạn sinh tương lai nhận gradien khử nhiễu.

Báo cáo không in phương trình vận tốc/MSE chính xác, các trọng số mục tiêu, một hàm mất
mát tái dựng VAE riêng biệt, hoặc việc bản thân VAE có được huấn luyện hay không. Những
chi tiết này vẫn ở trạng thái `Unknown`. Không được so sánh bằng số hàm mất mát video
của mô hình với hàm mất mát hành động robot hoặc xem nó là bằng chứng về sự thành công
của một chính sách có thể thực thi. [Qwen-RobotWorld, Mục 3–4][robotworld]

## Danh sách kiểm tra khi so sánh

Trước khi diễn giải hai đường cong mất mát, hãy đối chiếu tất cả các yếu tố sau:

- biểu diễn đầu ra, đơn vị và thống kê chuẩn hóa;
- đường chân trời, các chiều đang hoạt động và mặt nạ chính xác;
- phép rút gọn theo từng từ tố, từng kênh, từng phần tử hoặc từng mẫu;
- hướng nội suy dòng, vận tốc mục tiêu và phân phối thời điểm nhiễu;
- tỷ lệ lấy mẫu VL/hành động và các hệ số mục tiêu;
- chính sách triển khai, phần thưởng, hệ số chiết khấu, lợi thế và phép cắt ngưỡng PPO cho RL;
- tập dữ liệu, tập phân chia, thành phần lô và giai đoạn điểm kiểm tra.

Hãy ghi `action_loss`, `vl_loss`, `actor_loss`, `value_loss`, phần thưởng và tỷ lệ thành
công thành các trường riêng biệt. Chỉ một tổng có trọng số không thể cho biết thành phần
nào đã thay đổi.

## Nguồn

- Wang và cộng sự. *Qwen-VLA*. [Bài báo][qwen-vla] · [PDF cục bộ][qwen-vla-local]
- Nhóm Qwen. *Qwen-RobotManip*. [Bài báo][robotmanip] ·
  [PDF cục bộ][robotmanip-local]
- Nhóm Qwen. *Qwen-RobotNav*. [Bài báo][robotnav] · [PDF cục bộ][robotnav-local]
- Nhóm Qwen. *Qwen-RobotWorld*. [Bài báo][robotworld] ·
  [PDF cục bộ][robotworld-local]

[qwen-vla]: https://arxiv.org/abs/2605.30280v2
[qwen-vla-local]: ../../../papers/05-gwen/vla-specific/qwen_vla_2605.30280.pdf
[robotmanip]: https://arxiv.org/abs/2606.17846
[robotmanip-local]: ../../../papers/05-gwen/vla-specific/qwen_robotmanip_2606.17846.pdf
[robotnav]: https://arxiv.org/abs/2606.18112
[robotnav-local]: ../../../papers/05-gwen/vla-specific/qwen_robotnav_2606.18112.pdf
[robotworld]: https://arxiv.org/abs/2606.17030
[robotworld-local]: ../../../papers/05-gwen/vla-specific/qwen_robotworld_2606.17030.pdf
