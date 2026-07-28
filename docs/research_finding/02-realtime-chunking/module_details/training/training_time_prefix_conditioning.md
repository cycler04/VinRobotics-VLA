# Flow training và prefix conditioning

## Ý chính

Repository sử dụng **conditional flow matching** để huấn luyện action policy.

Ở chế độ bình thường, toàn bộ action chunk được làm nhiễu và toàn bộ chunk tham gia tính loss.

Khi bật `simulated_delay`, công thức flow matching cơ bản