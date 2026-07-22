# Qwen-RobotWorld — Dataset

## 1. Embodied World Knowledge (EWK)

EWK gồm khoảng **8.6M video-text pairs**, hơn **200M observation frames**:

| Thành phần | Quy mô/thông tin |
|---|---|
| General video/image | 30% tổng corpus |
| Embodied data | 70% tổng corpus |
| Manipulation | khoảng 5.9M samples; 20+ robot morphologies; 1,300+ skills |
| Driving | khoảng 200K samples trong summary; full curated mixture ~1,744,405 clips/2,405h |
| Indoor navigation | 6,064 episodes; 134 indoor scenes; ~49.8 km |
| Human-to-robot transfer | MANO-to-robot pipeline; 14 robot morphologies |
| Multi-view | khoảng 1.6M embodied samples; synchronized 2–4 views |

![Paper Figure 1 — EWK data overview](Image/figure_1_ewk_overview.png)

## 2. Action-language mapping

Action signals khác nhau giữa joint angles, waypoints, steering, heading và hand motion được chuyển thành natural-language action. Model học:

```text
Visual state s_t + language action a_t
                 ↓
             predict s_(t+1)
```

Coverage gồm 20+ embodiments và 500+ action categories: manipulation primitives, long-horizon compositions, locomotion/navigation và dynamic/deformable interactions.

## 3. Năm lớp annotation

1. **Task Goal:** mục tiêu và desired state transition.
2. **Action Detail:** trajectory, micro-action, speed, force, viewpoint.
3. **Physical Feedback:** displacement, deformation, contact-state change.
4. **Comprehensive caption:** 50–100 words.
5. **Concise caption:** 15–30 words.

Hai loại caption được sampling equal probability 50/50 trong training.

## 4. Data domains

| Domain | Nguồn/vai trò |
|---|---|
| Manipulation | EgoHOD, EPIC-Kitchens, Bridge V2, RH20T, DROID, RoboMIND, RoboCoin, Agibot-World, Galaxea, ActionNet, OpenLoong, Robotwin… |
| Autonomous driving | Waymo E2E, NVIDIA PhysicalAI-AD, Bench2Drive, Sekai |
| Indoor navigation | VLNVerse, Isaac Sim, 134 scenes |
| Human-to-robot | Human egocentric video → MANO reconstruction → robot retargeting/video editing |
| General data | Internet images/videos, multi-resolution, no AIGC according to paper |

## 5. Quality filtering

LLM judge kiểm tra factual accuracy, specificity, instruction clarity và viewpoint consistency. Caption gần ngưỡng hoặc domain thiếu đại diện được human review; prompt được refine theo scenario/task/embodiment rồi re-annotate.

![Paper Figure 2 — data processing pipeline](Image/figure_2_data_processing.png)
