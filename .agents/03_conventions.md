# Quy ước

## Knowledge và tài liệu

- `.agents/` là nguồn sự thật duy nhất cho knowledge, memory, plan, workflow và
  adapter skill của workspace.
- Chỉ file trong cùng một cấp được đánh số `01_`, `02_`, ...; thư mục không đánh
  số. Nếu chèn file giữa chuỗi, đánh số lại để thứ tự đọc liên tục.
- Một sự thật chỉ có một nơi sở hữu. File khác liên kết tới nơi đó thay vì chép.
- Tài liệu nghiên cứu viết tiếng Việt rõ ràng; giữ nguyên thuật ngữ tiếng Anh khi
  dịch làm mất nghĩa. Code, identifier và docstring dùng tiếng Anh.
- Mọi báo cáo phải tách rõ: nguồn/dữ kiện, diễn giải, giả thuyết, giới hạn và câu
  hỏi mở.
- Không ghi vào memory điều đã có thể đọc trực tiếp từ code hoặc Git.

## Code

- Python `>=3.11`, type hints, `from __future__ import annotations` và
  `pathlib.Path`.
- `snake_case` cho module/function/biến; `PascalCase` cho class/dataclass.
- Giữ raw state/action và provenance; không suy diễn semantics khi nguồn không
  khai báo.
- Thay đổi nhỏ, có test tương ứng. Không sửa vendor để ép theo style của project
  gốc.

## Thử nghiệm

- Ghi hypothesis và tiêu chí thành công trước khi chạy.
- Bắt đầu bằng smoke test nhỏ nhất có thể quan sát lỗi.
- Lưu command, config, input identity, environment, metric và artifact path.
- Không xem log đẹp hoặc một sample tốt là bằng chứng tổng quát.
- Dataset/workload lớn phải có ước lượng dung lượng, RAM/VRAM và thời gian trước.

## Output và an toàn

- Dataset, output, artifact, checkpoint và log sinh ra không được commit.
- Không commit, push, tạo/xóa repository hoặc thay đổi hệ thống bên ngoài nếu
  chưa được yêu cầu rõ.
- Trước khi xóa, resolve đường dẫn và xác nhận nằm dưới workspace.
- Bảo toàn thay đổi đang dở của người dùng; không dùng lệnh reset/checkout để
  dọn worktree.

## Ngoại lệ do tool

- `AGENTS.md`, `SKILL.md`, `MEMORY.md`, `agents/openai.yaml` và
  `.codex/config.toml` giữ tên cố định theo cơ chế của tool, nên không đánh số.
- Skill trong `.agents/skills/` chỉ là adapter mỏng: phần quy trình chi tiết phải
  nằm ở `.agents/workflows/`.

