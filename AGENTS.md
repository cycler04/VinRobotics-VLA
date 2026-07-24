# Hướng dẫn cho AI agent

`.agents/` là nguồn sự thật duy nhất cho knowledge, plan, memory, workflow và skill của
workspace này. Không tạo bản sao project-specific trong `~/.codex/`, `.codex/` hoặc
`.claude/`.

Đọc theo thứ tự trước khi làm việc:

1. [Tổng quan](.agents/01_overview.md)
2. [Kiến trúc](.agents/02_architecture.md)
3. [Quy ước](.agents/03_conventions.md)
4. [Lệnh](.agents/04_commands.md)

Sau đó đọc đúng workflow trong [`.agents/workflows/`](.agents/workflows/) và skill tương
ứng trong [`.agents/skills/`](.agents/skills/) nếu yêu cầu khớp. Bối cảnh dài hạn nằm ở
[`.agents/memory/MEMORY.md`](.agents/memory/MEMORY.md); kế hoạch nghiên cứu nằm ở
[`.agents/plans/`](.agents/plans/).

Các AI Agent (Antigravity, Codex, v.v.) tự discover skill trong `.agents/skills/`. Với việc đọc artifact hoặc tài liệu dài, delegate cho custom agent `dataset_artifact_reader` hoặc `research_reader`; wrapper của Codex nằm ở `.codex/agents/`, còn hướng dẫn thật cho agent nằm trong `.agents/agents/`. Sau khi sửa hạ tầng, chạy `python3 .agents/scripts/01_validate_workspace.py --full`.

## Ba bẫy nguy hiểm nhất

- Workspace có thư mục `.git/` rỗng nhưng chưa phải Git repository. Không hứa có lịch sử,
  rollback hay trạng thái clean; không chạy commit/push nếu người dùng chưa yêu cầu.
- Tên và roadmap rộng hơn code thật: hiện chỉ có `LeRobot/RLDS -> canonical -> inspect |
  HDF5 | Parquet`. Chưa có training loader, writer LeRobot hay round-trip hoàn chỉnh.
- Dataset có thể rất lớn. `--max-episodes` của LeRobot hiện không chặn việc đọc metadata
  Parquet ban đầu, còn Parquet writer tích frame trong RAM. Luôn smoke test nhỏ và đo dung
  lượng trước khi chạy full.

## Quyền tự chủ

Tự khảo sát, sửa code/docs, chạy test, tải dữ liệu và dọn artifact trong workspace khi cần
cho mục tiêu đã giao. Trước khi xóa, resolve path và xác nhận nó nằm dưới workspace; không
đụng đường dẫn ngoài workspace. Luôn hỏi trước commit, push, tạo/xóa Git repository, hoặc
thao tác với hệ thống bên ngoài có ảnh hưởng đến người khác.
