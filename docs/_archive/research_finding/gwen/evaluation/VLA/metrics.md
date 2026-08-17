# Các metric dùng để đánh giá mô hình Qwen VLA

> **Câu hỏi:** Mỗi điểm số về thao tác, điều hướng, lái xe và world model có ý nghĩa gì?
>
> **Phạm vi:** Chỉ có chức năng tính điểm. Nội dung và thang đo của tập dữ liệu/môi trường được
> trong [datasets.md](datasets.md); nhiệm vụ, phân chia, cảm biến, cài đặt bộ đánh giá và
> Bảng kết quả Qwen nằm trong [benchmarks.md](benchmarks.md); mục tiêu đào tạo
> nằm trong [loss.md](loss.md). Nghiên cứu được kiểm tra vào ngày 22-07-2026.

## Ranh giới phạm vi

- **tập dữ liệu** là các ví dụ, phương thức, chú thích, phần tách và giấy phép.
- **Giao thức điểm chuẩn** chỉ định phân tách, nhắc nhở, xử lý trước và
  người đánh giá được sử dụng.
- **số liệu** chuyển đổi dự đoán hoặc triển khai thành điểm số.

Để đánh giá được thể hiện, trước tiên người đánh giá chuyển đổi một quỹ đạo thành các sự kiện
chẳng hạn như thành công, va chạm hoặc khoảng cách đường đi. Số liệu tổng hợp các sự kiện đó.
Thay đổi robot, phân bổ cảnh, thời gian chờ, vị từ thành công hoặc số lượng
các thử nghiệm thay đổi giao thức điểm chuẩn ngay cả khi công thức vẫn được gọi là SR.

## Triển khai nhị phân thành công

Đối với các đợt triển khai $N$ và chỉ báo thiết bị đầu cuối nhị phân $S_i$,

$$
\mathrm{SR}=\frac{1}{N}\sum_{i=1}^{N}S_i,
\qquad S_i\in\{0,1\}.
$$

Cao hơn là tốt hơn. SR ước tính xác suất đáp ứng điểm chuẩn
vị từ thành công hoàn chỉnh dưới sự phân phối triển khai của nó. Nó không tiết lộ
tiến triển một phần, mức độ nghiêm trọng của va chạm, sự can thiệp, sự suôn sẻ hoặc sự hoàn thành
thời gian.

Việc tổng hợp phải rõ ràng:

$$
\mathrm{MacroSR}=\frac{1}{T}\sum_{t=1}^{T}\mathrm{SR}_t,
\qquad
\mathrm{PooledSR}=\frac{\sum_t n_t\mathrm{SR}_t}{\sum_t n_t}.
$$

Macro SR xử lý mọi tác vụ như nhau; SR tổng hợp các tác vụ theo số lần phát hành.
Chúng chỉ trùng nhau trong những trường hợp đặc biệt như mẫu số bằng nhau.

Bởi vì SR là tỷ lệ thực nghiệm, hãy báo cáo số lượng thử nghiệm và Wilson
hoặc khoảng nhị thức chính xác khi cách diễn giải nhị thức IID là hợp lý.
Nhiệm vụ và hạt giống tạo ra các cụm, do đó, giá trị mỗi nhiệm vụ/mỗi hạt giống hoặc nhận biết cụm
bootstrap thường mang lại nhiều thông tin hơn khoảng cách cấp độ mục hẹp.
[Khoảng nhị thức NIST][nist]

## Điểm thao tác DOMINO

DOMINO báo cáo SR và `Manipulation Score (MS)` ghi công một phần riêng biệt. Của nó
tỷ lệ tiến độ cơ bản là

$$
\rho=1-
\frac{\|p_{ee}^{T_{end}}-p_{obj}^{T_{end}}\|_2}
{\|p_{ee}^{0}-p_{obj}^{T_{end}}\|_2}.
$$

Đối với các nhiệm vụ hai nhánh, việc hoàn thành lộ trình là

$$
RC=100\max(\rho_{left},\rho_{right}),
$$

và các tập thành công được gán $RC=100$. MS bắt đầu từ RC, nhân nó lên
bằng 0,5 nếu mục tiêu thoát khỏi không gian làm việc hoặc trường quan sát an toàn và bằng 0,8 sau
một sự va chạm với sự lộn xộn môi trường. Các hình phạt có thể phức tạp.

Do đó MS kết hợp tiến trình tác động cuối cùng với các hình phạt về an toàn/khả năng hiển thị; nó là
không phải là xác suất và không được tính trung bình với SR. Bài báo nêu rõ
$RC\in[0,100]$ nhưng không đưa ra quy tắc cắt cho $\rho$ thô âm hoặc
trường hợp mẫu số 0 rõ ràng; những chi tiết đó yêu cầu người đánh giá được ghim.
[Giấy DOMINO, định nghĩa số liệu] [domino]

## Điều hướng tầm nhìn và ngôn ngữ

Cho phép đường dẫn thực thi $Q=(q_1,\ldots,q_m)$, đường dẫn tham chiếu
$R=(r_1,\ldots,r_n)$, khoảng cách trắc địa $d(\cdot,\cdot)$ và ngưỡng thành công
$d_{th}$.

- **NE (Lỗi điều hướng)** là khoảng cách mục tiêu trắc địa cuối cùng,
  $d(q_m,r_n)$; thấp hơn là tốt hơn.
- **OS/OSR (Tỷ lệ thành công của Oracle)** tính một tập nếu có bất kỳ điểm truy cập nào nằm
  trong phạm vi $d_{th}$ của mục tiêu. OS có thể vượt quá SR khi đại lý đến thăm mục tiêu
  khu vực nhưng không kết thúc ở đó.
- **SR** tính một tập nếu trạng thái cuối cùng của nó thỏa mãn vị từ thành công,
  thường là NE cuối cùng trong $d_{th}$ theo quy tắc dừng được chỉ định.

Với độ dài đường dẫn ngắn nhất $l_i$ và độ dài đường dẫn được thực hiện $p_i$,

$$
\mathrm{SPL}=\frac{1}{N}\sum_{i=1}^{N}
S_i\frac{l_i}{\max(p_i,l_i)}.
$$

SPL thưởng cho thành công và giảm giá cho những chuyến đi không hiệu quả. Một tập phim không thành công
luôn đóng góp bằng 0, ngay cả khi nó đi theo hầu hết lộ trình.
[Giấy SPL][spl]

Độ trung thực của đường dẫn được ghi lại bằng Độ cong thời gian động được chuẩn hóa:

$$
\mathrm{nDTW}(R,Q)=
\exp\left(-\frac{\mathrm{DTW}(R,Q)}{|R|d_{th}}\right).
$$

nDTW nằm trong $(0,1]$, duy trì thứ tự đường dẫn và đưa ra tín hiệu được phân loại ngay cả khi
đại lý không đạt được mục tiêu. Thay vào đó, `SDTW = SR × nDTW` kiểm soát độ trung thực của đường dẫn bằng cách
thành công cuối cùng. Loại khoảng cách và $d_{th}$ là các tham số giao thức.
[giấy nDTW][ndtw]

## Theo dõi mục tiêu hoạt động trong EVT-Bench

EVT-Bench định nghĩa **Tốc độ theo dõi** là

$$
\mathrm{TR}=\frac{S}{L},
$$

trong đó $S$ là số bước được theo dõi thành công và $L$ là tổng số bước
số bước. Cao hơn là tốt hơn. Bài viết không chính thức hóa đầy đủ các
vị từ theo dõi mỗi bước, vì vậy nó không nên được diễn giải đơn thuần là “mục tiêu
có thể nhìn thấy” mà không cần ghim mã.

**Tỷ lệ va chạm** của nó là phần nhỏ các tập phim bị kết thúc do va chạm với
hình người mục tiêu; thấp hơn là tốt hơn. SR của nó sử dụng quy tắc đầu cuối riêng biệt: tại
cuối cùng, đặc vụ phải luôn hướng về mục tiêu và ở một khoảng cách an toàn
từ 1–3m. Do đó, TR có thể cao trong khi SR thấp hơn.
[Định nghĩa chỉ số TrackVLA/EVT-Bench][trackvla]

## NAVSIM v1 PDMS

Điểm NAVSIM của v1 kết hợp hệ số an toàn với kế hoạch có trọng số
chất lượng:

$$
\mathrm{PDMS}
=NC\cdot DAC\cdot
\frac{5\,TTC+5\,EP+2\,C}{12}.
$$

Các thành phần là:

- `NC`: không có va chạm do lỗi;
- `DAC`: tuân thủ khu vực có thể lái xe;
- `TTC`: điểm thời gian xảy ra va chạm;
- `EP`: điểm tiến bộ cái tôi;
- `C`: điểm thoải mái.

`NC` **Không có va chạm do lỗi**, mặc dù bảng của Qwen-RobotNav đang mở rộng nó
là "Tuân thủ điều hướng". Sản phẩm tạo ra các hệ số cổng NC và DAC. PDMS là
được tính toán theo mô phỏng giả không phản ứng trong bốn giây trong đó nền
các tác nhân theo dõi tương lai đã được ghi lại; nó không phải là xác suất thành công về mặt vật chất hay
số liệu triển khai vòng kín phản ứng.

NAVSIM v2 hiện tại sử dụng `EPDMS`, với các thành phần và hệ số nhân khác nhau. PDMS
và EPDMS không thể thay thế cho nhau nên phiên bản này là một phần của tên chỉ số.
[Định nghĩa số liệu NAVSIM] [navsim]

## Điểm EBench

EBench báo cáo SR nhị phân và `Score` tiến trình một phần dành riêng cho nhiệm vụ. Điểm số
được tập hợp từ các mục tiêu phụ có trọng số thay vì một tỷ lệ hoàn thành chung
công thức. Ví dụ: phiếu tự đánh giá nhiệm vụ đã xuất bản có thể gán tín dụng Máy rửa chén
để mở, đặt từng bát và đóng lại, trong khi Peg-in-Hole chia tín dụng
giữa việc loại bỏ và chèn.

Điểm số trả lời “mức độ hài lòng của phần đánh giá của nhiệm vụ này” chứ không phải “cái gì
một phần của quỹ đạo chung đã được hoàn thành.” Việc tổng hợp nhiều nhiệm vụ phải
do đó hãy ghim bộ nhiệm vụ, trọng lượng và bản sửa đổi bộ. Các nguồn sẵn có làm
không chỉ định một phương trình tổng thể phổ quát trên tất cả
nhiệm vụ/trường hợp/hạt giống; nó vẫn là `Unknown`, không phải là giá trị trung bình giả định của các bước phụ.
[Tiêu chí nhiệm vụ EBench] [ebench]

## Số liệu mô hình thế giới

Qwen-RobotWorld tạo ra video trong tương lai chứ không phải hành động của robot. Họ số liệu của nó
phải giữ nguyên khoảng cách tên:

| Suite | Điểm thành phần được báo cáo là bao nhiêu | Giới hạn giải thích quan trọng |
|---|---|---|
| EWMBench | Tính nhất quán của cảnh; chuyển động HSD/Dyn/nDTW; đa dạng ngữ nghĩa, BLEU, CLIP và logic | Cân hỗn hợp; không phải chính sách SR |
| DreamGen | Căn chỉnh nhận thức (PA) và hướng dẫn tuân theo (IF) giữa các nhóm môi trường/đối tượng/hành vi | IF sử dụng Qwen2.5-VL làm người đánh giá |
| PBench | Miền QA cộng với chất lượng hình ảnh VBench-derived; Nhìn chung là ý nghĩa của họ | Tên miền QA sử dụng Qwen2.5-VL làm người đánh giá |
| WorldModelBench | Hướng dẫn tuân theo, lẽ thường về khung/thời gian và năm loại vi phạm vật lý | Sử dụng thẩm phán phù hợp với con người của riêng mình, không phải Qwen2.5-VL |

Những điểm số này đo lường chất lượng quan sát được tạo ra. Họ không xác lập rằng một
bộ điều khiển xuôi dòng có thể hoàn thành một nhiệm vụ. Thẩm phán nhận dạng, nhắc nhở và thành phần
thang đo là một phần của hợp đồng số liệu. Dòng dõi Qwen được chia sẻ trong DreamGen/PBench
là một yếu tố gây nhiễu tiềm ẩn, chưa được chứng minh là sai lệch của người đánh giá.
[Qwen-RobotWorld, Phần 5][thế giới robot]

## Số liệu hoạt động

Việc đánh giá robot phải báo cáo chất lượng nhiệm vụ bên cạnh, không trộn lẫn với:

- độ trễ suy luận trung bình và p95, thời hạn kiểm soát và thông lượng;
- tỷ lệ va chạm/suýt va chạm và sự can thiệp của con người;
- thời gian, khoảng cách và năng lượng để thành công;
- tốc độ phục hồi sau một xáo trộn xác định;
- bộ nhớ tăng tốc tối đa và phần cứng/cấu hình.

Mỗi mẫu số đều quan trọng: “va chạm trên mỗi tập”, “va chạm trên mét” và
“Phần của các tập phim bị kết thúc do va chạm” trả lời các câu hỏi khác nhau.

## Danh sách kiểm tra báo cáo số liệu

```text
tên số liệu, công thức, hướng và phạm vi
sửa đổi bộ/bộ đánh giá và trọng lượng thành phần
bộ nhiệm vụ, phân chia, robot/môi trường và vị từ thành công
đầu vào cảm biến/bộ điều khiển/lịch sử, thời gian chờ và quy tắc đặt lại/can thiệp
trục tổng hợp, số lần thử trên mỗi nhiệm vụ, hạt giống và khoảng tin cậy
Các số liệu về độ trễ, độ an toàn và lỗi được lưu giữ dưới dạng các trường riêng biệt
```

## Nguồn

- Fang và cộng sự. *DOMINO*. [Giấy][domino]
- Anderson và cộng sự. *Về việc đánh giá các tác nhân điều hướng được thể hiện*. [Giấy][sp]
- Ilharco và cộng sự. *Đánh giá chung về hướng dẫn điều hướng có điều kiện*.
  [Giấy][ndtw]
- Vương và cộng sự. *TrackVLA*. [Giấy][trackvla]
- Đội NAVSIM. [Định nghĩa số liệu chính thức][navsim]
- Thực tập sinh Robotics. [Giới thiệu nhiệm vụ EBench] [ebench]
- Đội Qwen. *Qwen-Thế giới robot*. [Giấy][thế giới robot] ·
  [PDF cục bộ][thế giới robot-cục bộ]
- NIST. [Khoảng tin cậy tỷ lệ nhị thức] [nist]

[domino]: https://arxiv.org/abs/2603.15620
[sp]: https://arxiv.org/abs/1807.06757
[ndtw]: https://arxiv.org/abs/1907.05446
[trackvla]: https://proceedings.mlr.press/v305/wang25f.html
[điều hướng]: https://github.com/autonomousvision/navsim/blob/main/docs/metrics.md
[bàn ghế]: https://internrobotics.github.io/EBench-doc/evaluation/task-showcase/
[thế giới robot]: https://arxiv.org/abs/2606.17030
[robotworld-local]: ../../../papers/05-gwen/vla-spec/qwen_robotworld_2606.17030.pdf
[nist]: https://itl.nist.gov/div898/handbook/prc/section2/prc241.htm
