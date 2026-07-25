---
name: corrections
---

# Agent corrections

- **Sai:** đặt plan/memory/skill của VinRobotics vào global hoặc `.claude/`.
  **Đúng:** toàn bộ source of truth nằm dưới `.agents/`; `AGENTS.md` chỉ làm entrypoint.
- **Sai:** gọi workspace là VLA training/inference pipeline hoàn chỉnh vì tên repo và tài liệu
  bao quát rộng.
  **Đúng:** kiểm tra code/runtime; hiện trọng tâm là data ingestion/conversion R&D.
- **Sai:** xem file HDF5/Parquet ghi thành công là training-ready.
  **Đúng:** cần loader test, semantics, validation và loss report.
- **Sai:** tự invent taxonomy cho paper khi người dùng đã đưa reading order.
  **Đúng:** ưu tiên cấu trúc người dùng chốt và dùng tên nhóm ngắn theo convention.

Preference nền nằm ở [[user-preferences]]; context dự án nằm ở [[project-context]].
