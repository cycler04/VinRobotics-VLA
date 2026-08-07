# Research draft: ViFailback

## 1. Mục tiêu

- **Paper:** *Diagnose, Correct, and Learn from Manipulation Failures via Visual Symbols*
  ([PDF](../../../papers/06-retry-handle/09_vifailback_visual_symbols.pdf), CVPR 2026).
- **Câu hỏi chính:** ViFailback giải quyết phần nào của manipulation recovery, và evidence
  có đủ để quy cải thiện cho visual symbols hay không?
- **Phạm vi:** dataset, benchmark, ViFailback-8B và recovery system.

## 2. Key points cần làm rõ

1. **Định vị đúng contribution:** core model học failure diagnosis/guidance bằng VQA;
   executor riêng chuyển guidance thành action. Cần tránh gọi đây là end-to-end recovery
   VLA nếu không có action-policy objective.
2. **Data và benchmark split:** paper công bố 95 training tasks, 22 benchmark tasks nhưng
   chỉ 100 tasks tổng; cần kiểm tra task/trajectory overlap.
3. **Visual-symbol contribution:** robot result là evidence của toàn supervisor–executor
   stack; chưa thấy text-only hoặc matched-executor ablation để cô lập symbols.
4. **Benchmark validity:** Hard tasks dùng GPT-4o judge; cần kiểm tra prompt, rubric và
   agreement trước khi xem score là khách quan.
5. **Deployment gap:** latency, failure trigger, retry budget và safety behavior chưa rõ.

## 3. Câu hỏi và evidence cần tìm

| Priority | Question | Evidence/source cần kiểm tra | Status |
|---|---|---|---|
| P0 | ViFailback có trực tiếp học recovery actions từ failed trajectories không? | PDF §3.3.2, §4.1, §4.3, §5; training code | Paper hiện chỉ hỗ trợ VQA supervisor |
| P0 | Train và benchmark có task/trajectory-disjoint không? | Fig.2, split manifest, dataset card | Unknown; cardinality gợi ý task overlap |
| P0 | Visual symbols có tốt hơn text-only trong cùng recovery stack không? | Table 4 và ablation/config public | Unknown; Table 4 là system-level result |
| P1 | Hard benchmark có tái lập được không? | Judge prompt, rubric, appendix, human agreement | Chưa đủ bằng chứng |
| P1 | Recovery loop có đáp ứng runtime/safety thực tế không? | Code, latency và trigger metrics | Unknown |

## 4. Outline báo cáo đầy đủ

1. **Nguồn và câu hỏi nghiên cứu** — định vị ViFailback trong retry/recovery.
2. **Why** — real-world failure data và grounded correction còn thiếu gì.
3. **ViFailback Dataset** — trajectory source, VQA/symbol annotation và split.
4. **ViFailback-Bench** — Lite/Hard protocol, evaluator và leakage risk.
5. **ViFailback-8B** — VQA fine-tuning, output contract và giới hạn claim.
6. **Recovery System** — supervisor→executor flow, robot evidence và ablation còn thiếu.
7. **Limitations và liên hệ workspace** — deployment gaps và thử nghiệm nên chạy.

## 5. Unknown và bước tiếp theo

- Kiểm tra dataset manifest để xác nhận split và accounting của 58,128 VQA.
- Tìm text-only/matched-executor ablation và Hard-evaluator protocol trong artifact public.
- Nếu hai điểm trên không có, báo cáo phải giữ kết luận ở mức: **failure VQA supervisor +
  downstream recovery integration**, chưa chứng minh visual symbols là nguyên nhân độc lập.
