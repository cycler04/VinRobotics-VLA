# Kế hoạch: dataset tới training-ready

Trạng thái: **R&D plan**. File này mô tả hướng ưu tiên, không khẳng định các phần planned đã
được implement.

## Mục tiêu

Với ít nhất một LeRobot sample và hai RLDS sample, chứng minh đường đi:

```text
source -> canonical -> validated artifact -> training loader -> reproducible report
```

Trong đó episode boundary, timestamp, state/action semantics, language, image reference và
provenance không bị mất ngoài các loss đã ghi rõ.

## Các chặng

1. **Baseline đã verify một phần**
   - Giữ 4 unit test hiện tại xanh.
   - Lưu smoke commands cho LeRobot, DROID và OXE trong `.agents/04_commands.md`.
   - Cập nhật roadmap cũ để tách implemented và pending khi có yêu cầu sửa tài liệu dự án.

2. **Validation trên real fixture**
   - Thêm test RLDS nhỏ, deterministic và không phụ thuộc tải mạng.
   - Assert episode/step, flags, timestamp, state/action, language và image reference.
   - Ghi schema/loss report cho từng source.

3. **Training-ready contract**
   - Chọn training target cụ thể trước khi viết thêm writer.
   - Thêm manifest/version/provenance và loader đọc được một batch.
   - Test round-trip ở những field có thể bảo toàn; định nghĩa tolerance cho timestamp/pixel.

4. **Scale và reliability**
   - Streaming/chunked Parquet thay cho gom toàn bộ rows.
   - Push down episode selection hoặc metadata scan có giới hạn.
   - Atomic output/resume và benchmark RAM, throughput, disk.

5. **Knowledge loop**
   - Mỗi dataset/paper mới tạo hoặc cập nhật một báo cáo trong `docs/`.
   - Mỗi correction về cách agent làm việc cập nhật `.agents/memory/corrections.md`.

## Tiêu chí hoàn thành

- Một lệnh tái lập được từ source sample tới artifact.
- Một loader đích đọc được batch và kiểm tra shape/dtype/semantics.
- Validation và round-trip test pass trên fixture đại diện.
- Báo cáo nêu rõ verified, unknown, loss và giới hạn scale.
