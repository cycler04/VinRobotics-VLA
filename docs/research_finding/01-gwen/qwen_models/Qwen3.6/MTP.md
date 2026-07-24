# Multi-Token Prediction (MTP) trong Qwen

## Tóm tắt

MTP mở rộng mục tiêu **next-token prediction**: từ một hidden state, mô hình
không chỉ học dự đoán token kế tiếp mà còn học dự đoán một số token tương lai.
Trong inference, MTP đóng vai trò một **draft predictor** nhẹ. Mô hình chính
vẫn xác minh các token được đề xuất, vì vậy MTP không tự quyết định output cuối
­cùng.

> Lưu ý: phần mô tả dưới đây phân biệt cơ chế MTP nói chung với chi tiết triển
> khai của từng phiên bản Qwen. Qwen không công bố đầy đủ mọi siêu tham số như
> trọng số auxiliary loss hoặc lịch huấn luyện; không nên suy ra các giá trị đó.

## 1. Next-token prediction chuẩn

Giả sử chuỗi token là:

```text
x₀ = "The", x₁ = "cat", x₂ = "is", x₃ = "sleeping", x₄ = "."
```

Trong huấn luyện, toàn bộ chuỗi vẫn được xử lý song song bằng causal mask.
Tuy nhiên, tại mỗi vị trí `t`, hidden state `hₜ` chỉ được dùng để dự đoán một
token kế tiếp:

```text
Input:   The       cat       is        sleeping
Target:  cat       is        sleeping  .
```

$$
p(x_{t+1}\mid x_{\le t}) = \operatorname{softmax}(W_{\mathrm{LM}}h_t)
$$

$$
\mathcal{L}_{\mathrm{NTP}} = -\sum_t \log p(x_{t+1}\mid x_{\le t})
$$

```text
x₀ … xₜ → Main Decoder → hₜ → LM Head → distribution của xₜ₊₁
```

## 2. MTP thêm gì?

MTP không loại bỏ nhánh next-token gốc. Nó giữ:

```text
Main decoder → LM head → xₜ₊₁
```

và thêm các bước dự đoán token xa hơn:

```text
hₜ ──┬── Main LM head ──→ xₜ₊₁
     ├── MTP step 1 ────→ xₜ₊₂
     ├── MTP step 2 ────→ xₜ₊₃
     └── … ──────────────→ …
```

Nói ngắn gọn, mục tiêu chuyển từ **một vị trí → một future target** thành
**một vị trí → nhiều future targets**.

### Dịch label

Với chuỗi `[x₀, x₁, …, x₆]`, các label được tạo bằng cách dịch trái:

```text
next +1: [x₁, x₂, x₃, x₄, x₅, x₆]
next +2: [x₂, x₃, x₄, x₅, x₆]
next +3: [x₃, x₄, x₅, x₆]
```

Một dạng loss tổng quát là:

$$
\mathcal{L} = \mathcal{L}_{+1} + \lambda_1\mathcal{L}_{+2}
 + \lambda_2\mathcal{L}_{+3} + \cdots
$$

Trong đó `L₊₁` là loss next-token chuẩn; các thành phần còn lại là auxiliary
loss cho các token tương lai và `λ` là trọng số tương ứng.

## 3. MTP module dùng thông tin nào?

Theo các implementation phục vụ Qwen3-Next/Qwen3.x, MTP không chỉ nhận hidden
state của main decoder. Nó kết hợp:

1. hidden state từ main decoder (hoặc hidden state của bước MTP trước đó);
2. embedding của token hiện tại/token draft trước đó.

Một luồng khái quát:

```text
Norm(h) ───────────────┐
                      ├─ concatenate [h; e(x)] ∈ R²ᴰ
Norm(embedding(x)) ───┘
                              │
                         Linear 2D → D
                              │
                       MTP decoder layer
                              │
                          final norm
                              │
                         shared LM head
                              │
                    logits của token tương lai
```

Nếu hidden size là `D`, các shape điển hình là:

```text
H_main: [B, T, D]       E_token: [B, T, D]
Concat: [B, T, 2D]      Fusion:  [B, T, D]
Logits: [B, T, V]
```

Việc đưa embedding token trước đó vào giúp MTP vừa dùng context đã được main
model nén trong `h`, vừa biết token nào đang được nối tiếp. Vì vậy module có
thể hoạt động như một drafter nhỏ thay vì chạy lại toàn bộ backbone.

## 4. Một MTP head hay nhiều layer lặp?

Về mặt khái niệm có thể vẽ `MTP-1 → +2`, `MTP-2 → +3`, … nhưng điều này không
đồng nghĩa mỗi offset có một backbone độc lập. Trong serving, số bước thường
liên quan đến cấu hình như `num_nextn_predict_layers`; một module/layer có thể
được gọi lặp qua nhiều speculative steps.

Vì vậy, cách diễn đạt an toàn là:

```text
(hidden hiện tại, token trước đó) → token draft kế tiếp
```

và phép biến đổi này có thể được áp dụng đệ quy để tạo draft 1, draft 2, draft 3.

## 5. Training

### Main model

Main decoder vẫn tối ưu loss next-token:

```text
hₜ → LM head → xₜ₊₁
```

### MTP module

Trong teacher forcing, token đúng từ dataset được đưa vào bước kế tiếp:

```text
hₜ + Emb(xₜ₊₁) → MTP step 1 → dự đoán xₜ₊₂
hₜ¹ + Emb(xₜ₊₂) → MTP step 2 → dự đoán xₜ₊₃
```

Teacher forcing giúp tránh lỗi dây chuyền do draft sai ngay từ bước đầu. Tuy
nhiên, nếu không có tài liệu chính thức, không nên khẳng định Qwen dùng một
chiến lược regularization hoặc scheduled sampling cụ thể.

Mô tả tổng quát phù hợp nhất là:

```text
Main next-token loss + auxiliary multi-token losses
```

Chưa đủ thông tin công khai để kết luận chắc chắn MTP được train riêng sau khi
freeze backbone hay được train jointly trong toàn bộ quá trình.

## 6. Inference và speculative decoding

MTP tạo candidate tokens; **main model là target model và chịu trách nhiệm xác
minh**.

```text
Main model → token A
MTP        → draft [B, C, D]
Target     → verify [A, B, C, D] song song
           → accept longest valid prefix
```

Nếu draft sai ở token thứ ba, hệ thống chỉ giữ prefix đúng trước đó và lấy token
đúng từ target model (hoặc bắt đầu vòng draft tiếp theo). Nhờ việc xác minh nhiều
vị trí trong một forward, hệ thống có thể chốt nhiều token sau một lần chạy
backbone thay vì chạy backbone tuần tự cho từng token.

Speculative decoding được triển khai đúng có thể giữ phân phối đầu ra của target
model; MTP chỉ là bộ đề xuất nhanh. Mức tăng tốc phụ thuộc vào acceptance rate,
số token draft, kích thước module, batch size, phần cứng và framework serving.

## 7. Sơ đồ end-to-end

```text
Input x₀ … xₜ
      │
      ▼
Main Qwen decoder ──→ hₜ ──→ LM head ──→ x̂ₜ₊₁
                         │
                         ▼
              MTP: Norm(hₜ) + Norm(Emb(x̂ₜ₊₁))
                         │
                   Linear 2D → D
                         │
                 MTP layer / step 1 ──→ x̂ₜ₊₂
                         │
                 MTP layer / step 2 ──→ x̂ₜ₊₃
                         │
                         ▼
                  Draft sequence
                         │
                         ▼
             Main model verify in parallel
                         │
                Accept longest valid prefix
```

## Kết luận

MTP là một auxiliary objective và/hoặc draft mechanism bổ sung cho mô hình
autoregressive:

- training: main model học `+1`, MTP học thêm các token tương lai;
- inference: MTP sinh draft nhanh bằng hidden state và embedding token;
- verification: main model kiểm tra draft và quyết định token cuối cùng.

Do đó, MTP không biến mô hình thành một bộ dự đoán nhiều token hoàn toàn độc
lập; nó bổ sung khả năng dự đoán tương lai để phục vụ speculative decoding và
có thể cải thiện hiệu quả huấn luyện.
