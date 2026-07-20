# Quy ước làm việc

## Ưu tiên ra quyết định

1. Bảo toàn source data và semantics.
2. Tạo bằng chứng tái lập được trên sample nhỏ.
3. Phân biệt fact đo được, inference và proposal.
4. Sau khi đúng mới tối ưu scale và tốc độ.

Lý do: converter chạy xong nhưng làm mất action semantics hoặc episode boundary tạo dữ liệu
nguy hiểm hơn một pipeline chưa hoàn thành.

## Code

- Python >= 3.11; dùng type hint hiện đại và `pathlib.Path` như code hiện tại.
- Tên module/function/variable dùng `snake_case`, class dùng `PascalCase`, constant dùng
  `UPPER_SNAKE_CASE`.
- Giữ reader, canonical contract, validator và writer tách rời.
- Mọi thay đổi liên quan schema phải có test về episode boundary, shape, dtype, timestamp và
  metadata; không chỉ test “không crash”.
- Source input là immutable; derived output ghi sang `output/`.
- Không thêm dependency nếu standard library hoặc dependency hiện có giải quyết gọn được.

Repo chưa có formatter/linter chính thức. Không tuyên bố `ruff`, `black` hay `mypy` là gate
cho tới khi chúng được thêm và kiểm chứng.

## Báo cáo và ghi chú

- `docs/`: báo cáo đã tổng hợp, đủ bằng chứng để đọc lại sau nhiều tháng.
- `notes/`: source list, task, quan sát thô hoặc ghi chú đang phát triển.
- `.agents/memory/`: chỉ lưu preference, mục tiêu dài hạn và correction của agent; không
  chép lại thứ code hoặc tài liệu dự án đã nói.
- `.agents/plans/`: kế hoạch tương lai; luôn đánh dấu planned/verified để không bị đọc nhầm
  thành trạng thái đã chạy.

Khi sửa báo cáo cũ, giữ style và tên hiện hữu. File mới trong `.agents/` đánh số tuần tự
`01_name.md`, `02_name.md`; thư mục không đánh số. `SKILL.md`, `AGENTS.md` và file memory là
ngoại lệ vì có tên cố định hoặc cần khớp `name:`/wikilink.

Mỗi kết luận kỹ thuật phải truy được về ít nhất một trong các loại bằng chứng: code path,
command output, artifact metadata hoặc nguồn chính thức. Với tài liệu bên ngoài, ưu tiên
paper và documentation gốc; ghi rõ ngày truy cập nếu thông tin có thể thay đổi.

## Quyền tự chủ và an toàn

Agent được tự chạy code, tải dataset và xóa artifact trong workspace để hoàn thành nhiệm vụ.
Trước khi xóa, resolve đường dẫn và kiểm tra nó nằm dưới root workspace. Không hiển thị token,
password hoặc nội dung `.env`/`notes/wifi.txt`. Hỏi trước commit, push, Git init, thao tác ngoài
workspace hoặc thay đổi dịch vụ bên ngoài.

Không commit trừ khi người dùng yêu cầu rõ.
