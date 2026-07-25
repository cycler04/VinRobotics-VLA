# Workflow: Validate Workspace Infrastructure

## Mục đích

Kiểm tra tính toàn vẹn của hạ tầng agent trong `.agents/`, liên kết trong tài liệu Markdown, định nghĩa skill, memory, và các smoke check của mã nguồn Python/Shell.

## Quy trình thực hiện

1. **Kiểm tra hạ tầng Agent & Markdown**
   ```bash
   python3 .agents/scripts/01_validate_workspace.py
   ```
   Lệnh này xác minh:
   - Tất cả các link Markdown trong `AGENTS.md` và `.agents/` không bị đứt.
   - Các file trong `.agents/memory/` chứa `name:` hợp lệ và không có wikilink hỏng.
   - Tất cả skills trong `.agents/skills/` có `SKILL.md` với `name:` và `description:` hợp lệ.
   - Các cấu hình TOML của Codex không bị lỗi cú pháp.

2. **Kiểm tra Toàn bộ Code & Tests (--full)**
   ```bash
   python3 .agents/scripts/01_validate_workspace.py --full
   ```
   Lệnh này thực hiện thêm:
   - Chạy `pytest` trong `.venv`.
   - Biên dịch kiểm tra cú pháp Python (`compileall`).
   - Kiểm tra cú pháp shell scripts (`bash -n`).
   - Kiểm tra lệnh CLI help (`python -m vla_data_tools --help`).

3. **Xử lý sự cố**
   - Nếu liên kết bị hỏng: sửa đường dẫn Markdown tương ứng.
   - Nếu skill bị lỗi metadata: kiểm tra frontmatter YAML của `SKILL.md`.
   - Nếu pytest fail: đọc log chi tiết và sửa bug trong `src/` hoặc `tests/`.
