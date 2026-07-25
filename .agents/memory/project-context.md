---
name: project-context
---

# Project context

Giai đoạn hiện tại là nghiên cứu data layer cho VLA. Câu hỏi trung tâm không phải “convert có
chạy không” mà là “artifact có giữ đúng semantics và thực sự dùng được cho training không”.

Ưu tiên dài hạn:

1. hiểu format và semantics của từng dataset;
2. inspect/validate trên sample đại diện;
3. ingest qua canonical contract;
4. chứng minh bằng training loader và test;
5. ghi lại kết quả thành knowledge tái sử dụng.

Chi tiết code và trạng thái hiện tại luôn đọc ở `../01_overview.md`, không lấy từ memory này.
Kế hoạch thực hiện nằm ở `../plans/01_dataset_to_training_ready.md`.
