# Module `src/render_levels.py`

## Vai trò

Đây là utility trực quan độc lập, không tham gia train/eval. Nó load 12 JSON level, reset
environment vào từng level, render một frame đầu tiên và ghi JPEG.

## Luồng

```text
worlds/l/*.json
-> saving.load_from_json_file
-> Kinetix-Symbolic-Continuous-v1
-> reset_to_level(seed=0)
-> make_render_pixels
-> uint8 + transpose + vertical flip
-> rendered_levels/<level>.jpg
```

Code đặt screen `512×512`, `downscale=2`, nên renderer nội bộ làm việc theo static params
đã thay đổi. Module không có dataclass/CLI flag; muốn đổi input/output phải sửa code
([`render_levels.py:10-20`](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b/src/render_levels.py#L10-L20),
[`render_levels.py:38-80`](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b/src/render_levels.py#L38-L80)).

## Khác với loader training

`render_levels.load_levels` không assert `StaticEnvParams`/`EnvParams` đọc từ JSON khớp
constant trong module; hai assert đã bị comment. `train_expert.load_levels` thì assert
nghiêm ngặt. Vì vậy một level có thể render được nhưng vẫn bị từ chối khi train
([`render_levels.py:23-35`](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b/src/render_levels.py#L23-L35)).

## Cách dùng

```bash
cd third_party/01_real-time-chunking-kinetix
uv run src/render_levels.py
```

Output dự kiến:

```text
rendered_levels/
├── car_launch.jpg
├── cartpole_thrust.jpg
├── ...
└── trampoline.jpg
```

Ảnh chưa được sinh trong workspace khi viết báo cáo này.
