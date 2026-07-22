# Qwen-RobotWorld — Evaluation

## 1. Các benchmark đo gì?

| Benchmark | Đo lường | Kết quả Qwen-RobotWorld |
|---|---|---:|
| EWMBench | Scene consistency, motion correctness, semantic alignment | Overall 4.60, hạng 1 |
| DreamGen Bench | Instruction following và physics alignment trên GR1 Env/Object/Behavior | Total 4.952, hạng 1 |
| PBench | Domain physical understanding + video quality | Overall 0.804, tốt nhất open-source |
| WorldModelBench | Instruction following, common sense, physics adherence | Total 8.99, hạng 3 overall |
| RoboTwin-IF | Zero-shot instruction following trên task manipulation mới | Qualitative/zero-shot evidence |

## 2. EWMBench

EWMBench có 21 samples, 7 tasks, action-ordering constraints. Nhóm metric:

- **SceneC:** scene consistency.
- **HSD, Dyn, nDTW:** motion fidelity/correctness.
- **Diversity, BLEU, CLIP, Logics:** semantic alignment và action-logic consistency.

Qwen-RobotWorld đạt 4.60, cao hơn LVP 4.05; HSD 0.566, SceneC 0.914 và Logics 1.00.

![Paper Figure 5 — fine-grained language grounding](Image/figure_5_language_grounding.png)

## 3. DreamGen Bench

Đánh giá ba subset GR1:

- GR1-Env: environment generalization;
- GR1-Object: object compositional generalization;
- GR1-Behavior: behavior/long-horizon generalization.

Mỗi subset đo physics alignment (PA) và instruction following (IF). Tổng Qwen-RobotWorld là 4.952. Điểm GR1-Object IF là 0.878; GR1-Behavior IF là 0.832, vẫn thấp hơn một số baseline.

![Paper Figure 6 — cross-embodiment/task/view](Image/figure_6_generalization.png)

## 4. PBench

PBench kết hợp:

```text
Domain Score: physical behavior QA
       +
Quality Score: VBench video metrics
       ↓
Overall Score
```

Domain gồm AV, Robot, Industry, Physics, Human và Common Sense. Qwen đạt Domain 0.857, Quality 0.751 và Overall 0.804. Motion smoothness 0.990; pixel/aesthetic metrics thấp hơn general video models do output resolution thấp hơn.

## 5. WorldModelBench

Đánh giá 350 instances, 7 domains, 56 subdomains:

- instruction following trên thang 0–3;
- common sense: frame và temporal quality;
- physics adherence: Newton, mass conservation, fluid, penetration và gravity.

Qwen đạt instruction following 2.33/3.0, physics adherence 1.00 ở các nhóm báo cáo, overall 8.99; đứng thứ 3 overall và tốt nhất trong open-source models.

## 6. Qualitative/generalization

- Fine-grained language grounding: đổi keyword target/action/destination làm output thay đổi tương ứng.
- Cross-embodiment: một instruction cho nhiều morphology.
- Cross-task/cross-environment: pick-place, bowl retrieval, cloth folding, handover.
- Multi-view consistency: main, wrist-left, wrist-right.
- Human-to-robot transfer.
- Autonomous driving và indoor navigation.
- RoboTwin-IF zero-shot trên task mới.

![Paper Figure 8 — RoboTwin-IF](Image/figure_8_robotwin_if.png)
![Paper Figure 9 — human-to-robot transfer](Image/figure_9_h2r.png)
![Paper Figure 10 — mobility generation](Image/figure_10_mobility.png)
