# Qwen3.5 — Training datasets

## Kết luận ngắn

Qwen3.5 là **open-weight với partial technical disclosure**, không phải dự án open-training hoàn toàn. Qwen công bố các nhóm dữ liệu/năng lực được tối ưu, nhưng chưa công bố bảng corpus đầy đủ, quy mô từng nguồn và sampling ratio.

## 1. Các nhóm dữ liệu được công bố hoặc suy ra từ mô tả chính thức

| Nhóm                   | Nội dung                                                 | Trạng thái |
| ----------------------- | --------------------------------------------------------- | ------------ |
| Text                    | Web/text chất lượng cao, instruction và knowledge     | △           |
| Code                    | Code generation, debugging, repository và terminal tasks | ✓/△        |
| Mathematics/STEM        | Toán, khoa học, reasoning và verifier tasks            | ✓/△        |
| Multilingual            | 201 languages and dialects                                | ✓           |
| Image-text              | Caption, visual QA, reasoning                             | ✓           |
| Video-text              | Frame/temporal understanding, video QA                    | ✓           |
| Documents/charts/OCR/UI | Document, chart, screenshot và interface understanding   | ✓           |
| Reasoning traces        | Suy luận và câu trả lời có cấu trúc               | △           |
| Tool trajectories       | Tool call, observation, multi-step interaction            | ✓/△        |
| Agent environments      | Browser, coding, planning, real-world-like tasks          | ✓/△        |

## 2. Dạng dữ liệu trong unified sequence

```text
[text] [visual tokens] [text question] [visual tokens] [answer/tool call]
```

Điểm chính là visual token không chỉ dùng cho một encoder phụ ở inference; chúng tham gia vào multimodal foundation training cùng language tokens.

## 3. Mức độ public

| Trường cần biết                      | Tình trạng                                                   |
| ---------------------------------------- | -------------------------------------------------------------- |
| Tên đầy đủ từng dataset            | Chưa công bố đầy đủ                                     |
| Số mẫu / số token từng dataset       | Chưa công bố                                                |
| Sampling/mixing ratio                    | Chưa công bố                                                |
| Filtering pipeline                       | Chưa công bố hoàn chỉnh                                   |
| Deduplication recipe                     | Chưa công bố hoàn chỉnh                                   |
| Tỷ lệ text/code/vision/video           | Chưa công bố đầy đủ                                     |
| Synthetic data và recipe tạo dữ liệu | Chưa công bố đầy đủ                                     |
| Agent trajectory scale                   | Có mô tả quy mô lớn/million-scale, thiếu bảng chi tiết |

Vì vậy không nên nói “Qwen3.5 được huấn luyện trên Common Crawl/GitHub/OCR với tỷ lệ X%” nếu không có nguồn checkpoint chính thức xác nhận. Cách nói an toàn: “Qwen công bố các nhóm modality và domain, còn corpus, quy mô và tỷ lệ trộn chi tiết chưa public.”

## 4. Phân biệt training data và evaluation data

Các benchmark ở `evaluation.md` là dữ liệu đánh giá, không tự động là training corpus. Đặc biệt, tên benchmark được model card nhắc tới không chứng minh benchmark đó được dùng để pre-train hoặc SFT.

![TODO: taxonomy các nhóm dữ liệu](Image/qwen35_dataset_taxonomy.png)
![TODO: modality fusion của text/image/video/document](Image/qwen35_multimodal_data.png)
