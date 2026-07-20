---
name: MEMORY
---

# Workspace memory

Đây là entrypoint memory của VinRobotics. Memory dự án chỉ sống trong `.agents/memory/`.
Không tạo hoặc cập nhật bản project-specific ở global.

Đọc theo nhu cầu:

- [[user-preferences]] — cách người dùng muốn agent làm việc.
- [[project-context]] — mục tiêu/giai đoạn dài hạn, không lặp runtime details.
- [[corrections]] — các lần agent đã hiểu sai và cách tránh lặp lại.

Runtime truth nằm ở `../01_overview.md`; kế hoạch nằm ở `../plans/`. Không chép chúng vào
memory vì sẽ trôi lệch.
