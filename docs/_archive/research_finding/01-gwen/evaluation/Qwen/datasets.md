# Bộ dữ liệu đánh giá cho mô hình Qwen

> **Câu hỏi:** Những loại dữ liệu nào nằm sau bộ đánh giá Qwen chính,
> chúng lớn đến mức nào và mỗi ví dụ chứa gì?
>
> **Phạm vi:** Dữ liệu đánh giá công khai về ngôn ngữ và ngôn ngữ tầm nhìn Qwen
> các điểm kiểm tra. Đây không phải là bản dựng lại chương trình đào tạo độc quyền của Qwen
> tử thi. Thông tin thực tế về bộ dữ liệu đã được kiểm tra vào ngày 22-07-2026.

## Phạm vi ranh giới

Tệp này chỉ mô tả các ví dụ, phương thức, chú thích, phần tách, tỷ lệ,
quyền truy cập và giấy phép. Lời nhắc, tiền xử lý, người đánh giá và báo cáo Qwen
kết quả thuộc về [benchmarks.md](benchmarks.md); định nghĩa điểm thuộc về
[metrics.md](metrics.md); mục tiêu đào tạo thuộc về [loss.md](loss.md).

## Bản đồ tập dữ liệu

| Bộ dữ liệu | Loại và phương thức | Thang đo được xuất bản | Phân chia/cấu hình chính | Ví dụ nào chứa |
|---|---|---:|---|---|
| MMLU-Pro | Văn bản, kiến ​​thức/lý luận trắc nghiệm khó | 12.032 bài kiểm tra + 70 câu hỏi kiểm chứng | 14 lĩnh vực chủ đề; 10 lựa chọn trả lời | Câu hỏi, lựa chọn, câu trả lời và chủ đề |
| GPQA | Văn bản, trắc nghiệm khoa học trình độ sau đại học | Tổng cộng 448 câu hỏi; Kim cương có 198 | Sinh học, vật lý và hóa học; Diamond là tập con chặt chẽ nhất | Câu hỏi, lựa chọn và câu trả lời do chuyên gia viết |
| IFEval | Các ràng buộc tuân theo hướng dẫn văn bản | 541 lời nhắc, 25 loại hướng dẫn có thể kiểm chứng | Thu thập đánh giá công khai; mỗi lời nhắc có 1–3 ràng buộc | Lời nhắc cộng với hướng dẫn có thể kiểm tra bằng máy IDs |
| MMMU-Pro | Lý luận của chuyên gia về hình ảnh cộng với văn bản | 1.730 ví dụ trong mỗi cấu hình được phát hành | Tiêu chuẩn 10 lựa chọn, tiêu chuẩn 4 lựa chọn và chỉ tầm nhìn | Câu hỏi, một hoặc nhiều hình ảnh, tùy chọn, siêu dữ liệu chủ đề |
| MathVista | Lý luận toán học trực quan | 6.141 ví dụ từ 31 nguồn | `testmini` 1.000; đầy đủ `test` 5.141 | Hình ảnh, câu hỏi, loại câu trả lời và siêu dữ liệu nguồn/tác vụ |
| OCRBench | OCR và hình ảnh giàu văn bản QA | 1.000 cặp QA được xác minh thủ công | Năm nhóm nhiệm vụ; OCRBench gốc | Hình ảnh, câu hỏi liên quan đến văn bản và câu trả lời tham khảo |
| Tham chiếuCOCO | Nền tảng biểu thức giới thiệu trong hình ảnh COCO | 19.994 hình ảnh, 50.000 đối tượng, 142.210 biểu thức | Tàu 120.624; giá trị 10.834; thử nghiệmA 5.657; testB 5.095 biểu thức | Hộp/khu vực hình ảnh, cụm từ và đối tượng được giới thiệu |
| Video-MME | Video trả lời câu hỏi | 900 video, 254 giờ, 2.700 cặp QA | Ngắn, trung bình và dài; 6 miền/30 trường con | Video, ba câu hỏi trắc nghiệm, phụ đề tùy chọn |

Số lượng tùy thuộc vào phiên bản cụ thể. Đặc biệt, OCRBench nguyên bản không phải là OCRBench
v2 và Video-MME gốc không phải là Video-MME v2. Một bảng kê khai tập dữ liệu cục bộ phải
ghim bản phát hành và cấu hình được đặt tên.

## Bộ dữ liệu văn bản

### MMLU-Pro

MMLU-Pro là phiên bản sửa đổi khó hơn của đánh giá trắc nghiệm học thuật rộng rãi. Nó
mở rộng tập lựa chọn từ bốn lên mười, loại bỏ nhiều mục ồn ào hoặc tầm thường,
và nhấn mạnh những câu hỏi đòi hỏi phải suy luận hơn là nhớ lại trực tiếp. Của nó
14 lĩnh vực bao gồm toán học, vật lý, hóa học, luật, kỹ thuật, y tế,
lịch sử, tâm lý học và kinh doanh. Bản phát hành công khai cho thấy 12.032 hàng thử nghiệm
và một bộ xác nhận nhỏ gồm 70 hàng.

Nó hữu ích như một tập hợp hồi quy rộng, nhưng nó vẫn chỉ ở dạng văn bản và công khai. Nó
không kiểm tra xem tham chiếu trực quan có chính xác hay không hoặc liệu hành động của robot có
an toàn. [Giấy MMLU-Pro][mmlu-pro] [Kho lưu trữ chính thức][mmlu-pro-repo]

### GPQA và GPQA-Diamond

GPQA chứa 448 câu hỏi trắc nghiệm được viết và xác thực theo tên miền
chuyên gia về sinh học, vật lý và hóa học. Các tác giả đã thiết kế chúng để chống lại
câu trả lời thu được bằng cách tìm kiếm trên web thông thường. `GPQA-Diamond` là tập hợp con gồm 198 câu hỏi
được lựa chọn để có sự đồng thuận và chất lượng cao nhất; nó không phải là một cái lớn riêng biệt
tử thi.

Đây là một bộ sưu tập khoa học nhỏ, có độ khó cao hơn là một bộ sưu tập có phạm vi bao quát rộng.
tập dữ liệu kiến ​​thức.
[Giấy GPQA] [gpqa]

### IFEval

IFEval có 541 lời nhắc mang từ một đến ba ràng buộc được rút ra từ 25
các loại hướng dẫn có thể kiểm chứng bằng máy. Ví dụ yêu cầu các thuộc tính như chính xác
định dạng, sự xuất hiện của từ, cách viết hoa, cấu trúc danh sách hoặc sự hiện diện/vắng mặt của
nội dung quy định. Do đó, chú thích là một tập hợp các trình kiểm tra ràng buộc,
không phải là một nhãn ưu tiên hình thức tự do của con người.

Cấu trúc này làm cho các chú thích có thể được kiểm chứng bằng máy, nhưng nó
chỉ bao gồm các hướng dẫn có kiểm tra xác định. [Giấy IFEval] [ifeval]
[Triển khai chính thức][ifeval-repo]

## Bộ dữ liệu ngôn ngữ tầm nhìn

### MMMU-Pro

MMMU-Pro sửa đổi các câu hỏi đa phương thức của chuyên gia để giảm bớt các phím tắt. Việc phát hành
có ba cấu hình 1.730 ví dụ:

- `standard (10 options)`, dạng trắc nghiệm khó hơn chính;
- `standard (4 options)`, được giữ lại để so sánh với các giao thức cũ hơn;
- `vision`, nơi câu hỏi và các tùy chọn được hiển thị vào hình ảnh để
  đầu vào không thể được giải quyết thông qua đường dẫn chỉ có văn bản.

Nó bao gồm nhiều môn học đại học và chuyên nghiệp và có thể đính kèm một hoặc nhiều
đưa ra một câu hỏi. Ba cấu hình là kết xuất thay thế của
nhiệm vụ chứ không phải 5.190 câu hỏi ngữ nghĩa độc lập. [Giấy MMMU-Pro][mmmu-pro]
[Thẻ tập dữ liệu][mmmu-pro-data]

### MathVista

MathVista kết hợp 28 nguồn hiện có với ba nguồn mới được tạo:
IQTest, FunctionQA và PaperQA. 6.141 câu hỏi của nó bao gồm các số liệu, biểu đồ,
hình học, sơ đồ khoa học, tài liệu và hình ảnh giống như câu đố. nhỏ gọn
Phần chia `testmini` có 1.000 ví dụ; 5.141 còn lại tạo thành bộ kiểm tra đầy đủ.
Câu trả lời kiểm tra được giữ lại để đánh giá máy chủ, vì vậy các tệp cục bộ có thể không chứa
mọi thứ cần thiết để ghi điểm độc lập. [Bài báo MathVista] [Bài báo Mathvista]
[Dự án][mathvista]

### OCRBench

OCRBench gốc là bộ 1.000 câu hỏi hình ảnh nhỏ gọn, được xác minh thủ công
các cặp được tổ chức thành năm nhóm:

1. nhận dạng văn bản;
2. trả lời câu hỏi trực quan bằng văn bản cảnh;
3. tài liệu trả lời câu hỏi trực quan;
4. trích xuất thông tin quan trọng;
5. nhận dạng biểu thức toán học viết tay.

Nó kết hợp các cảnh thiên nhiên, tài liệu và chữ viết tay nên một tập hợp duy nhất có thể
ẩn lỗi miền máy ảnh. OCRBench v2 là một phiên bản song ngữ riêng biệt, lớn hơn nhiều
điểm chuẩn với 10.000 cặp QA và 31 kịch bản; kết quả của hai phiên bản
không được chia sẻ một cột. [Giấy OCRBench][ocrbench-giấy]
[Kho lưu trữ chính thức][ocrbench]

### giới thiệuCOCO

RefCOCO được xây dựng trên hình ảnh MS-COCO và các biểu thức giới thiệu được cộng đồng viết. Nó
chứa 142.210 biểu thức cho 50.000 trường hợp đối tượng trong 19.994 hình ảnh.
`testA` bị chi phối bởi con người, trong khi `testB` chứa các đối tượng khác.

Mục tiêu là một vùng đối tượng, do đó tập dữ liệu sẽ kiểm tra xem một cụm từ như
“Chiếc cốc bên trái đĩa” giải quyết được đối tượng đã định. Đó là một
tập dữ liệu nối đất 2D tĩnh, không phải tập dữ liệu hành động hoặc tư thế 3D.
[Giấy RefCOCO][refCOco]

### Video-MME

Video gốc-MME chứa 900 video với tổng thời lượng khoảng 254 giờ và 2.700
các cặp hỏi đáp. Thời lượng dao động từ khoảng 11 giây đến một giờ. các
bộ sưu tập bao gồm sáu tên miền cấp cao nhất và 30 trường con và được phân tầng thành
nhóm video ngắn, trung bình và dài. Các bản âm thanh và phụ đề được giữ lại để
các giao thức đó có thể đánh giá có hoặc không có thông tin phụ đề.

Hợp đồng tập dữ liệu thô phong phú hơn một tenxơ khung lấy mẫu. Một bản sao cục bộ
nên giữ nguyên video gốc, dấu thời gian, loại thời lượng, phụ đề và
đặt câu hỏi IDs trước khi áp dụng bộ lấy mẫu khung dành riêng cho từng mẫu máy. Video-MME v2 là
một tập dữ liệu sau này và phải được phiên bản riêng. [Giấy Video-MME][video-mme-giấy]
[Dự án][video-mme]

## Chi tiết truy cập và sao chép

Hầu hết các tập dữ liệu ở trên đều có mã công khai hoặc điểm nhập dữ liệu, nhưng “công khai” thì không.
có nghĩa là mọi đánh giá đều hoàn toàn cục bộ:

- MathVista giữ các nhãn kiểm tra đầy đủ đằng sau một dịch vụ đánh giá.
- Bản phân phối GPQA có các điều kiện cấp phép và quyền truy cập phải được kiểm tra tại
  thời gian tải xuống.
- bộ dữ liệu hình ảnh/video có thể kế thừa giấy phép hoặc điều khoản truy cập từ COCO, nguồn
  video hoặc bộ dữ liệu thành phần;

Ghi lại ít nhất những thông tin sau vào bảng kê khai tập dữ liệu cục bộ:

```yaml
name: Video-MME
version: original
source_revision: <commit-or-dataset-revision>
split: test
example_ids: <path-or-hash>
modalities: [video, audio, subtitles, text]
native_scale: {videos: 900, qa_pairs: 2700}
native_media_metadata: <timestamps-duration-audio-subtitle-tracks>
license_or_terms: <checked-url-and-date>
```

Đối với mọi tập dữ liệu, hãy giữ nguyên ví dụ IDs, siêu dữ liệu phương tiện gốc, phần tách, nguồn
bản sửa đổi và mọi hàng được lọc/loại trừ. Cài đặt nhắc nhở và số liệu thuộc về
tệp kê khai đang chạy, không có trong phần mô tả tập dữ liệu.

## Sự chồng chéo đào tạo và tình trạng địa phương

Qwen báo cáo các hỗn hợp đào tạo văn bản, hình ảnh, OCR, nối đất và video, nhưng
không phải là một bảng kê khai cấp độ mẫu đủ để loại trừ sự trùng lặp với mọi
bộ đánh giá. Do đó, sự nhiễm bẩn phải được ghi là **không xác định**, không phải
được cho là vắng mặt.

Không có tập dữ liệu nào trong báo cáo này được tải xuống hoặc nhập vào không gian làm việc này trong suốt thời gian
nghiên cứu này. Kho lưu trữ hiện tại có tính năng kiểm tra tập dữ liệu chuẩn và
công cụ chuyển đổi, nhưng không có Á hậu đánh giá Qwen.

## Nguồn

- Vương và cộng sự. *MMLU-Pro*. [Giấy][mmlu-pro] · [Kho lưu trữ][mmlu-pro-repo]
- Rein và cộng sự. *GPQA*. [Giấy][gpqa]
- Chu và cộng sự. *IFEval*. [Giấy][ifeval] · [Triển khai][ifeval-repo]
- Yue và cộng sự. *MMMU-Pro*. [Giấy][mmmu-pro] · [Bộ dữ liệu][mmmu-pro-data]
- Lu và cộng sự. *ToánVista*. [Giấy] [mathvista-paper] · [Dự án] [mathvista]
- Lưu và cộng sự. *OCRBench*. [Giấy][ocrbench-giấy] · [Kho lưu trữ][ocrbench]
- Yu và cộng sự. *Mô hình hóa bối cảnh trong các biểu thức giới thiệu*. [Giấy][refcoco]
- Fu và cộng sự. *Video-MME*. [Giấy][video-mme-paper] · [Dự án][video-mme]

[mmlu-pro]: https://arxiv.org/abs/2406.01574
[mmlu-pro-repo]: https://github.com/TIGER-AI-Lab/MMLU-Pro
[gpqa]: https://arxiv.org/abs/2311.12022
[nếu có]: https://arxiv.org/abs/2311.07911
[ifeval-repo]: https://github.com/google-research/google-research/tree/master/instruction_following_eval
[mmmu-pro]: https://arxiv.org/abs/2409.02813
[mmmu-pro-dữ liệu]: https://huggingface.co/datasets/MMMU/MMMU_Pro
[bài toán]: https://arxiv.org/abs/2310.02255
[toán học]: https://mathvista.github.io/
[ocrbench-giấy]: https://arxiv.org/abs/2305.07895
[điểm chuẩn]: https://github.com/qywh2023/OCRbench
[refcoco]: https://openaccess.thecvf.com/content_cvpr_2016/html/Yu_Modeling_Context_in_CVPR_2016_paper.html
[video-mme-giấy]: https://arxiv.org/abs/2405.21075
[video-mme]: https://video-mme.github.io/home_page.html
