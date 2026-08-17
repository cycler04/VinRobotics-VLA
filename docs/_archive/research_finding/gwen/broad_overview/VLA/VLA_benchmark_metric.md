# Các metric benchmark VLA: Công thức, phạm vi và diễn giải

## 1. Phạm vi

Không có một danh sách metric chuẩn hóa duy nhất được mọi bài báo về Vision-Language-Action sử dụng. Các nghiên cứu khác nhau đánh giá thao tác, điều hướng, học suốt đời, độ trung thực sim-to-real, triển khai thời gian thực hoặc dự đoán hành động offline.

Báo cáo này trình bày các nhóm metric xuất hiện thường xuyên trong nghiên cứu VLA và những metric chính dành riêng cho từng benchmark. Trọng tâm là thao tác robot, còn các metric điều hướng được trình bày riêng.

Một điểm phân biệt quan trọng là:

| Loại metric                       | Đánh giá                                                      | Ví dụ              |
| --------------------------------- | ------------------------------------------------------------- | ------------------ |
| **Training loss**                 | Quá trình tối ưu hóa có đang tiến triển hay không              | Flow-matching loss |
| **Offline prediction metric**     | Hành động dự đoán có giống hành động đã ghi lại hay không      | Action MAE         |
| **Rollout metric**                | Robot có thực sự hoàn thành tác vụ hay không                   | Success Rate       |
| **System metric**                 | Model có thể chạy kịp thời trên robot hay không                | End-to-end latency |

Training loss hoặc sai số hành động thấp không đảm bảo Rollout Success Rate cao. Rollout metric là bằng chứng mạnh nhất về năng lực thực tế của robot.

## 2. Ký hiệu

Giả sử:

- $N$: số episode được đánh giá;
- $T$: số tác vụ;
- $S_i \in \{0,1\}$: episode $i$ có thành công hay không;
- $p_i \in [0,1]$: mức tiến độ một phần trong episode $i$;
- $a_{t,d}$: hành động ground-truth tại timestep $t$, chiều $d$;
- $\hat{a}_{t,d}$: hành động được dự đoán;
- $H$: action horizon hoặc số timestep được dự đoán;
- $D$: số chiều hành động.

Khi bài báo báo cáo theo phần trăm, hãy nhân metric trong $[0,1]$ với 100.

## 3. Bảng tra cứu nhanh

| Metric                | Công thức hoặc đơn vị                      |      Phạm vi tự nhiên | Chiều hướng         | Ý nghĩa chính                              |
| --------------------- | ------------------------------------------ | --------------------: | ------------------- | ------------------------------------------ |
| Success Rate          | episode thành công / tổng episode          |               ([0,1]) | Cao hơn             | Hoàn thành toàn bộ tác vụ                   |
| Mean Success Rate     | SR trung bình trên các tác vụ              |               ([0,1]) | Cao hơn             | Hiệu năng đa tác vụ trung bình              |
| Progress Score        | các giai đoạn có trọng số đã hoàn thành    |               ([0,1]) | Cao hơn             | Hoàn thành một phần                         |
| CALVIN average length | số subtask hoàn thành trung bình           |               ([0,5]) | Cao hơn             | Kết nối chuỗi dài hạn                       |
| Episode Return        | tổng reward                                | Tùy theo environment | Cao hơn             | Reward mà policy RL tích lũy                |
| Action MAE            | sai số hành động tuyệt đối trung bình      |        ([0,infinity)) | Thấp hơn            | Độ gần của hành động offline                |
| Action MSE            | sai số hành động bình phương trung bình    |        ([0,infinity)) | Thấp hơn            | Phạt mạnh sai số hành động lớn              |
| Token Accuracy        | action token đúng / tổng token             |               ([0,1]) | Cao hơn             | Dự đoán hành động rời rạc                    |
| Cross-Entropy / NLL   | xác suất log âm                            |        ([0,infinity)) | Thấp hơn            | Chất lượng phân phối dự đoán                 |
| Position Error        | mét hoặc centimét                          |        ([0,infinity)) | Thấp hơn            | Độ chính xác điểm cuối                       |
| Rotation Error        | góc geodesic                               |     ([0,\pi]) radian | Thấp hơn            | Độ chính xác hướng                           |
| ADE                   | khoảng cách trung bình giữa các điểm quỹ đạo |      ([0,infinity)) | Thấp hơn            | Độ chính xác của toàn bộ quỹ đạo             |
| FDE                   | khoảng cách giữa các điểm cuối quỹ đạo     |        ([0,infinity)) | Thấp hơn            | Độ chính xác mục tiêu cuối                   |
| Collision Rate        | episode có va chạm / tổng episode          |               ([0,1]) | Thấp hơn            | An toàn                                     |
| Recovery Rate         | lỗi được phục hồi / lỗi có thể phục hồi    |               ([0,1]) | Cao hơn             | Sửa lỗi                                     |
| Latency               | mili giây                                  |        ([0,infinity)) | Thấp hơn            | Độ trễ phản ứng                              |
| Control Frequency     | hertz                                      |        ([0,infinity)) | Cao hơn, nếu ổn định | Tốc độ cập nhật                              |
| Throughput            | action step hoặc chunk mỗi giây            |        ([0,infinity)) | Cao hơn             | Năng lực tính toán                           |
| Generalization Gap    | SR trên seen trừ SR trên unseen            |              ([-1,1]) | Gần 0               | Độ bền vững trước dịch chuyển phân phối      |
| Pearson correlation   | điểm số thực so với mô phỏng               |              ([-1,1]) | Cao hơn             | Mức tương hợp tuyến tính sim-real            |
| MMRV                  | trung bình vi phạm thứ hạng tệ nhất        |              ([0,1])* | Thấp hơn            | Độ trung thực thứ hạng sim-real              |
| SPL                   | thành công được gia quyền theo hiệu quả đường đi |         ([0,1]) | Cao hơn             | Thành công và hiệu quả điều hướng            |

\*MMRV nằm trong ([0,1]) khi các giá trị hiệu năng trong thế giới thực bên dưới được chuẩn hóa về ([0,1]).

## 4. Metric hoàn thành tác vụ

### 4.1. Success Rate (SR)

Rollout metric VLA phổ biến nhất là thành công tác vụ dạng nhị phân:

$$
SR = \frac{1}{N}\sum_{i=1}^{N} S_i
$$

| Thuộc tính | Giá trị                                                                       |
| ---------- | ----------------------------------------------------------------------------- |
| Phạm vi    | ([0,1]), hoặc 0-100%                                                          |
| Tốt nhất   | 1 hoặc 100%                                                                    |
| Tệ nhất    | 0                                                                             |
| Được dùng bởi | các đánh giá VLA trên LIBERO, RLBench, RoboCasa, SimplerEnv, các đánh giá robot thực |

Ví dụ: nếu 82 trong 100 rollout thành công, SR là (0.82 = 82%\).

Tiêu chí thành công phải được định nghĩa chính xác. “Đặt chiếc cốc lên khay” có thể yêu cầu chiếc cốc được thả ra, đứng thẳng, nằm hoàn toàn bên trong ranh giới khay và ổn định trong một khoảng thời gian cố định.

#### Hạn chế

- Không ghi nhận công sức cho việc hoàn thành một phần.
- Phụ thuộc nhiều vào success predicate của tác vụ.
- Có thể che giấu hành vi không an toàn xảy ra trước khi cuối cùng thành công.
- Kết quả từ 10 lần thử kém tin cậy hơn nhiều so với cùng tỷ lệ phần trăm từ 1.000 lần thử.

### 4.2. Mean Success Rate trên các tác vụ

Với (T) tác vụ:

$$
\mathrm{MeanSR} = \frac{1}{T}\sum_{j=1}^{T}SR_j
$$

Phạm vi: ([0,1]). Cao hơn là tốt hơn.

Đây là **macro-average**: mỗi tác vụ nhận trọng số bằng nhau ngay cả khi các tác vụ có số episode khác nhau.

Ngược lại, **micro-average** gộp tất cả episode:

$$
\mathrm{MicroSR} =
\frac{\sum_j N_jSR_j}{\sum_j N_j}
$$

Micro-average và macro-average chỉ giống nhau khi mỗi tác vụ dùng cùng số lần thử.

### 4.3. Task Progress hoặc Progress Score (PS)

Với một tác vụ được chia thành (K) giai đoạn có thứ tự:

$$
\mathrm{PS}_i = \frac{1}{K}\sum_{k=1}^{K} c_{i,k}
$$

trong đó (c_{i,k}=1) nếu giai đoạn (k) đã hoàn thành trong episode (i).

Giá trị trung bình được báo cáo là:

$$
\mathrm{MeanPS} = \frac{1}{N}\sum_i \mathrm{PS}_i
$$

Phạm vi: ([0,1]). Cao hơn là tốt hơn.

VLABench dùng một dạng có trọng số, ghi nhận riêng việc chọn đúng object/receptacle và việc hoàn thành các bước của tác vụ:

$$
\mathrm{PS} =
\alpha\frac{n_{\mathrm{correct}}}{n_{\mathrm{targets}}}
+(1-\alpha)\frac{k_{\mathrm{completed}}}{K}
$$

với trọng số quyết định mặc định (\alpha=0.2). Giá trị này tự nhiên nằm trong ([0,1]), mặc dù các bảng có thể báo cáo theo thang 0-100. [Bài báo VLABench](https://arxiv.org/abs/2412.18194)

Progress Score có giá trị đối với các tác vụ dài, nhưng người thiết kế benchmark phải chọn các giai đoạn và trọng số có ý nghĩa.

### 4.4. Thành công của composite-task hoặc toàn bộ giai đoạn

Với tác vụ yêu cầu mọi giai đoạn:

$$
S_i = \prod_{k=1}^{K} c_{i,k}
$$

Phạm vi: (\{0,1\}) cho mỗi episode và ([0,1]) sau khi lấy trung bình.

Điều này giải thích vì sao thành công dài hạn giảm nhanh: một giai đoạn thất bại khiến toàn bộ episode không thành công.

### 4.5. Completion Time

$$
\mathrm{Time}_i = t_i^{\mathrm{finish}} - t_i^{\mathrm{start}}
$$

| Thuộc tính          | Giá trị                                                  |
| ------------------- | -------------------------------------------------------- |
| Phạm vi             | ([0,infinity)) giây                                      |
| Tốt nhất            | Thấp hơn, trong số các episode thành công và an toàn     |
| Xử lý thất bại      | Báo cáo riêng hoặc gán timeout                           |

Không lấy trung bình thời gian hoàn thành thành công trong khi âm thầm loại bỏ các thất bại; làm vậy có thể khiến một policy yếu trông có vẻ nhanh.

### 4.6. Episode Return

Chủ yếu được dùng cho reinforcement learning:

$$
G_i = \sum_{t=0}^{H_i-1}\gamma^t r_{i,t}
$$

trong đó (r_{i,t}) là reward và (\gamma \in [0,1]) là discount factor.

Phạm vi: tùy theo environment và có thể âm. Cao hơn là tốt hơn. Không thể so sánh giá trị Return giữa các benchmark trừ khi định nghĩa reward và cách chuẩn hóa giống hệt nhau.

## 5. Metric dài hạn

### 5.1. CALVIN prefix success rates

CALVIN đánh giá chuỗi gồm năm chỉ dẫn ngôn ngữ. Định nghĩa (C_i) là số subtask liên tiếp được hoàn thành từ đầu sequence (i), với (C_i \in \{0,1,2,3,4,5\}).

Success Rate khi hoàn thành ít nhất (k) subtask là:

$$
SR_{\ge k} = \frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[C_i \ge k]
$$

Phạm vi: ([0,1]) cho mỗi (k \in \{1,2,3,4,5\}). Cao hơn là tốt hơn.

Năm con số trả lời các câu hỏi ngày càng khó hơn:

- (SR_{\ge1}): có thể hoàn thành tác vụ đầu tiên không?
- (SR_{\ge5}): có thể hoàn thành toàn bộ chuỗi năm chỉ dẫn không?

### 5.2. CALVIN average completed sequence length

$$
\mathrm{AvgLen} = \frac{1}{N}\sum_{i=1}^{N} C_i
$$

Phạm vi tự nhiên là ([0,5]), và:

$$
\mathrm{AvgLen} = \sum_{k=1}^{5}SR_{\ge k}
$$

Ví dụ: các prefix success rate ([0.90,0.70,0.50,0.30,0.10]) cho:

$$
\mathrm{AvgLen}=0.90+0.70+0.50+0.30+0.10=2.50
$$

Đây là con số chính thường được trình bày trong các bảng VLA CALVIN hiện đại. CALVIN được thiết kế riêng cho continuous control có điều kiện ngôn ngữ với horizon dài. [Bài báo CALVIN](https://arxiv.org/abs/2112.03227)

### 5.3. Subtask transition success

Với chuyển tiếp từ giai đoạn (k) sang (k+1):

$$
\mathrm{TransitionSR}_k =
\frac{\#\text{episodes completing }k+1}
{\#\text{episodes completing }k}
$$

Phạm vi: ([0,1]). Cao hơn là tốt hơn.

Metric này giúp phân biệt thất bại khi thực hiện một kỹ năng với thất bại khi nhận ra kỹ năng tiếp theo cần bắt đầu.

## 6. Metric dự đoán hành động offline

Các metric này đánh giá demonstration thuộc tập held-out mà không chạy policy trong environment.

### 6.1. Mean Absolute Error (MAE hoặc L1)

$$
\mathrm{MAE} = \frac{1}{HD}\sum_{t=1}^{H}\sum_{d=1}^{D}
|a_{t,d}-\hat{a}_{t,d}|
$$

Phạm vi: ([0,infinity)). Thấp hơn là tốt hơn.

MAE dễ diễn giải hơn nếu hành động dùng đơn vị vật lý. Nếu hành động được chuẩn hóa, con số phụ thuộc vào phương pháp chuẩn hóa.

### 6.2. Mean Squared Error (MSE hoặc L2 loss)

$$
\mathrm{MSE} = \frac{1}{HD}\sum_{t,d}
(a_{t,d}-\hat{a}_{t,d})^2
$$

Phạm vi: ([0,infinity)). Thấp hơn là tốt hơn. Metric này phạt sai số lớn mạnh hơn MAE.

### 6.3. Root Mean Squared Error (RMSE)

$$
\mathrm{RMSE}=\sqrt{\mathrm{MSE}}
$$

Phạm vi: ([0,infinity)). Thấp hơn là tốt hơn. RMSE có cùng đơn vị với hành động.

### 6.4. Action-token accuracy

Với action token rời rạc:

$$
\mathrm{TokenAcc} = \frac{1}{M}\sum_{m=1}^{M}
\mathbb{1}[y_m=\hat{y}_m]
$$

Phạm vi: ([0,1]). Cao hơn là tốt hơn.

Token Accuracy phụ thuộc vào tokenization. Không thể so sánh công bằng hai model có bin size hoặc tokenizer FAST/VQ khác nhau bằng raw Token Accuracy.

### 6.5. Exact action-sequence match

$$
\mathrm{ExactMatch}=
\frac{1}{N}\sum_i\mathbb{1}[y_i^{1:M}=\hat{y}_i^{1:M}]
$$

Phạm vi: ([0,1]). Cao hơn là tốt hơn. Metric này cực kỳ nghiêm ngặt và trở nên kém hữu ích hơn đối với sequence dài.

### 6.6. Negative log-likelihood và cross-entropy

$$
\mathrm{NLL}=-\frac{1}{M}\sum_{m=1}^{M}
\log p_\theta(y_m\mid x,y_{<m})
$$

Phạm vi: ([0,infinity)) đối với target rời rạc. Thấp hơn là tốt hơn.

Perplexity là:

$$
\mathrm{PPL}=e^{\mathrm{NLL}}
$$

Phạm vi: ([1,infinity)). Thấp hơn là tốt hơn. Các metric này đánh giá dự đoán xác suất, không phải Rollout Success.

### 6.7. Flow-matching hoặc diffusion loss

Một objective flow-matching đơn giản hóa là:

$$
\mathcal{L}_{\mathrm{FM}}=
\mathbb{E}_{A,\epsilon,\tau}
\left[\|v_\theta(A_\tau,\tau,c)-u(A,\epsilon)\|_2^2\right]
$$

Phạm vi: ([0,infinity)). Thấp hơn là tốt hơn trong quá trình training.

Đây chủ yếu là một optimization loss. Thang đo của nó phụ thuộc vào cách chuẩn hóa hành động, noise schedule, time sampling và số chiều, vì vậy nó không phải một benchmark metric có ý nghĩa để so sánh giữa các bài báo.

## 7. Metric hình học và quỹ đạo

### 7.1. Sai số vị trí end-effector

$$
e_{\mathrm{pos}} = \|\hat{p}-p^*\|_2
$$

Phạm vi: ([0,infinity)) mét. Thấp hơn là tốt hơn.

Metric này hữu ích cho các tác vụ reaching, insertion và calibration. Luôn báo cáo đơn vị và target là pose từ demonstration hay goal do tác vụ định nghĩa.

### 7.2. Sai số hướng

Với các ma trận xoay dự đoán và target (\hat{R}) và (R^*):

$$
e_{\mathrm{rot}}=
\cos^{-1}\left(\frac{\operatorname{tr}((R^*)^T\hat{R})-1}{2}\right)
$$

Phạm vi: ([0,\pi]) radian hoặc ([0,180^ degrees]). Thấp hơn là tốt hơn.

Nên tránh sai số theo từng thành phần góc Euler ở gần điểm wrap-around và singularity.

### 7.3. Average Displacement Error (ADE)

$$
\mathrm{ADE}=\frac{1}{H}\sum_{t=1}^{H}
\|\hat{p}_t-p_t^*\|_2
$$

Phạm vi: ([0,infinity)). Thấp hơn là tốt hơn. ADE đo toàn bộ đường đi được dự đoán.

### 7.4. Final Displacement Error (FDE)

$$
\mathrm{FDE}=\|\hat{p}_H-p_H^*\|_2
$$

Phạm vi: ([0,infinity)). Thấp hơn là tốt hơn. FDE chỉ đo điểm cuối.

### 7.5. Khoảng cách Dynamic Time Warping (DTW)

$$
\mathrm{DTW}(P,Q)=
\min_{\omega}\sum_{(i,j)\in\omega}d(p_i,q_j)
$$

Phạm vi: ([0,infinity)). Thấp hơn là tốt hơn. DTW căn chỉnh các quỹ đạo thực hiện chuyển động tương tự ở tốc độ khác nhau.

### 7.6. Độ dài đường đi

$$
L=\sum_{t=2}^{H}\|p_t-p_{t-1}\|_2
$$

Phạm vi: ([0,infinity)). Ngắn hơn thường hiệu quả hơn, nhưng đường đi ngắn nhất không nhất thiết an toàn nhất hoặc mượt nhất.

### 7.7. Độ mượt, gia tốc và jerk

Vận tốc, gia tốc và jerk rời rạc có thể được xấp xỉ bằng sai phân hữu hạn:

$$
v_t=\frac{p_t-p_{t-1}}{\Delta t},\qquad
a_t=\frac{v_t-v_{t-1}}{\Delta t},\qquad
j_t=\frac{a_t-a_{t-1}}{\Delta t}
$$

Một jerk score phổ biến là:

$$
J=\frac{1}{H-2}\sum_t\|j_t\|_2^2
$$

Phạm vi: ([0,infinity)). Thấp hơn nghĩa là mượt hơn. Các giá trị không thể so sánh trừ khi timestep, đơn vị, cách lọc và thời lượng quỹ đạo khớp nhau.

## 8. Metric thời gian thực và tính toán

### 8.1. End-to-end latency

$$
L_{\mathrm{e2e}} =
L_{\mathrm{camera}}+L_{\mathrm{preprocess}}+L_{\mathrm{VLM}}+
L_{\mathrm{action}}+L_{\mathrm{communication}}+L_{\mathrm{controller}}
$$

Phạm vi: ([0,infinity)) ms. Thấp hơn là tốt hơn.

Báo cáo median, p95 và latency tối đa. Chỉ báo cáo mean latency sẽ che giấu các lần lỡ deadline điều khiển.

### 8.2. Control Frequency

$$
f_{\mathrm{control}}=\frac{1}{\Delta t_{\mathrm{command}}}
$$

Phạm vi: ([0,infinity)) Hz. Cao hơn có thể cải thiện khả năng phản ứng, nhưng chỉ khi command ổn định và dựa trên observation mới.

### 8.3. Action-generation throughput

$$
\mathrm{Throughput}=\frac{\#\text{generated action steps}}{\text{second}}
$$

hoặc số chunk mỗi giây. Phạm vi: ([0,infinity)). Cao hơn là tốt hơn.

Throughput không giống Control Frequency. Một model có thể sinh nhanh một chunk 50 step nhưng chỉ làm mới observation hai lần mỗi giây.

### 8.4. Tuổi hoặc độ cũ của observation

$$
A_{\mathrm{obs}} = t_{\mathrm{execution}}-t_{\mathrm{capture}}
$$

Phạm vi: ([0,infinity)) ms. Thấp hơn là tốt hơn. Điều này đặc biệt quan trọng đối với việc thực thi action chunk bất đồng bộ.

### 8.5. Deadline Miss Rate

$$
\mathrm{DMR}=\frac{\#\{i:L_i>B\}}{N}
$$

trong đó (B) là latency budget. Phạm vi: ([0,1]). Thấp hơn là tốt hơn.

### 8.6. Real-time factor (RTF)

$$
\mathrm{RTF}=\frac{\text{simulated or generated duration}}
{\text{wall-clock duration}}
$$

Phạm vi: ([0,infinity)). RTF (>1) nhanh hơn thời gian thực; RTF (<1) chậm hơn.

### 8.7. Metric tài nguyên

| Metric              | Đơn vị/phạm vi                 | Chiều hướng                         |
| ------------------- | ------------------------------ | ----------------------------------- |
| Parameters          | số lượng, ([0,infinity))       | Thấp hơn khi chất lượng ngang nhau  |
| FLOPs per inference | phép toán, ([0,infinity))      | Thấp hơn khi chất lượng ngang nhau  |
| Peak GPU memory     | GB, ([0,infinity))             | Thấp hơn                            |
| Energy per chunk    | joule, ([0,infinity))          | Thấp hơn                            |
| Training GPU-hours  | GPU-hours, ([0,infinity))      | Thấp hơn khi chất lượng ngang nhau  |

Các metric hiệu quả luôn phải đi kèm với hiệu năng tác vụ.

## 9. Metric độ bền vững và khả năng khái quát hóa

### 9.1. Seen và unseen Success Rate

Báo cáo riêng:

$$
SR_{\mathrm{seen}},\qquad SR_{\mathrm{unseen}}
$$

Mỗi giá trị nằm trong ([0,1]). Cao hơn là tốt hơn.

“Unseen” phải nêu rõ điều gì đã thay đổi: object instance, category, cách diễn đạt chỉ dẫn, scene, tác vụ, camera hay embodiment.

### 9.2. Generalization Gap

$$
\mathrm{Gap}=SR_{\mathrm{seen}}-
SR_{\mathrm{unseen}}
$$

Phạm vi: ([-1,1]). Giá trị gần 0 chỉ đáng mong muốn khi cả hai Success Rate đều cao. Gap bằng 0 khi cả hai điểm đều bằng 0 không có ích.

### 9.3. Mức giảm hiệu năng tương đối

$$
\mathrm{RelativeDrop}=
\frac{SR_{\mathrm{clean}}-SR_{\mathrm{shift}}}
{SR_{\mathrm{clean}}}
$$

Được định nghĩa khi (SR_{\mathrm{clean}}>0). Phạm vi điển hình là (( -infinity,1]); 0 nghĩa là không giảm, giá trị dương nghĩa là suy giảm, còn giá trị âm nghĩa là cải thiện dưới dịch chuyển. Thấp hơn là tốt hơn.

### 9.4. Worst-group success

$$
\mathrm{WorstGroupSR}=\min_g SR_g
$$

Phạm vi: ([0,1]). Cao hơn là tốt hơn. Metric này làm lộ ra các điều kiện yếu bị giá trị trung bình tổng thể che khuất.

### 9.5. Hiệu quả mẫu

Các ví dụ gồm:

- số demonstration cần thiết để đạt SR mục tiêu;
- số lần thành công trên mỗi 100 demonstration;
- diện tích dưới learning curve theo số episode.

Phạm vi số đếm tự nhiên là ([0,infinity)), trong đó yêu cầu ít dữ liệu hơn là tốt hơn đối với một mục tiêu hiệu năng cố định.

## 10. Metric an toàn, thất bại và phục hồi

### 10.1. Collision Rate

$$
\mathrm{CollisionRate}=
\frac{\#\text{episodes with collision}}{N}
$$

Phạm vi: ([0,1]). Thấp hơn là tốt hơn.

Các bài báo nên phân biệt tiếp xúc object cần thiết cho tác vụ với va chạm không an toàn.

### 10.2. Constraint Violation Rate

$$
\mathrm{ViolationRate}=
\frac{\#\text{violated monitored constraints}}
{\#\text{constraint opportunities}}
$$

Phạm vi: ([0,1]). Thấp hơn là tốt hơn. Các constraint có thể gồm quy tắc an toàn về joint, workspace, force, speed hoặc thời gian.

### 10.3. Safe Success Rate

$$
\mathrm{SafeSR}=
\frac{1}{N}\sum_i
\mathbb{1}[\text{task success}_i \land \text{no safety violation}_i]
$$

Phạm vi: ([0,1]). Cao hơn là tốt hơn.

Luôn báo cáo SR thông thường bên cạnh SafeSR. Chênh lệch giữa chúng là tỷ lệ episode thành công theo cách không an toàn.

### 10.4. Intervention Rate

$$
\mathrm{InterventionRate}=
\frac{\#\text{episodes requiring human intervention}}{N}
$$

Phạm vi: ([0,1]). Thấp hơn là tốt hơn.

Một cách khác là số lần can thiệp mỗi giờ hoặc mỗi kilômét, với phạm vi ([0,infinity)).

### 10.5. Recovery Rate

$$
\mathrm{RecoveryRate}=
\frac{\#\text{failures successfully recovered}}
{\#\text{recoverable failures encountered}}
$$

Phạm vi: ([0,1]). Cao hơn là tốt hơn.

Mẫu số phải được định nghĩa: mọi thất bại, thất bại được tiêm vào, hoặc thất bại được đánh giá là có thể phục hồi.

### 10.6. Mean Time Between Failures (MTBF)

$$
\mathrm{MTBF}=\frac{\text{total operating time}}
{\#\text{failures}}
$$

Phạm vi: ([0,infinity)) đơn vị thời gian. Cao hơn là tốt hơn.

## 11. Metric học suốt đời được LIBERO sử dụng

Ban đầu LIBERO đánh giá học suốt đời, mặc dù các bài báo fine-tuning VLA hiện đại thường dùng các tác vụ LIBERO và chỉ báo cáo Mean Task Success. Benchmark gốc báo cáo Forward Transfer, Negative Backward Transfer và diện tích dưới success curve. [Bài báo LIBERO](https://arxiv.org/abs/2306.03310)

Gọi (R_{i,j}) là mức thành công trên tác vụ (j) sau khi đã học đến hết tác vụ (i).

### 11.1. Forward Transfer (FWT)

Một dạng dựa trên ma trận thường dùng là:

$$
\mathrm{FWT}=\frac{1}{T-1}\sum_{j=2}^{T}
(R_{j-1,j}-b_j)
$$

trong đó (b_j) là hiệu năng trên tác vụ (j) khi chưa học các tác vụ trước đó.

Phạm vi: ([-1,1]) khi mức thành công được chuẩn hóa. Cao hơn là tốt hơn. FWT dương có nghĩa các tác vụ trước giúp ích cho tác vụ mới.

Phần triển khai của LIBERO nhấn mạnh tốc độ học trên tác vụ mới, vì vậy giá trị chính xác phụ thuộc vào quy trình checkpoint và đường cong học của nó.

### 11.2. Backward Transfer (BWT)

$$
\mathrm{BWT}=\frac{1}{T-1}\sum_{j=1}^{T-1}
(R_{T,j}-R_{j,j})
$$

Phạm vi: ([-1,1]). Cao hơn là tốt hơn.

- dương: việc học sau này cải thiện các tác vụ trước;
- 0: hiệu năng trước đó được giữ lại;
- âm: catastrophic forgetting.

### 11.3. Negative Backward Transfer (NBT)

Một dạng quên không âm là:

$$
\mathrm{NBT}=\frac{1}{T-1}\sum_{j=1}^{T-1}
\max(0,R_{j,j}-R_{T,j})
$$

Phạm vi: ([0,1]). Thấp hơn là tốt hơn. NBT dễ đọc hơn khi hiểu là “lượng kiến thức bị quên.”

### 11.4. Area Under the Success Curve (AUC)

Với tiến độ training được chuẩn hóa (x \in [0,1]):

$$
\mathrm{AUC}=\int_0^1 SR(x)\,dx
$$

Phạm vi: ([0,1]) khi cả hai trục được chuẩn hóa. Cao hơn là tốt hơn. AUC ưu tiên các policy học nhanh và duy trì được hiệu năng.

## 12. Độ trung thực của đánh giá sim-to-real

SIMPLER dùng Rollout Success Rate trong simulation, nhưng đánh giá **chất lượng simulator với vai trò proxy xếp hạng policy** bằng các metric tương quan. [Bài báo SIMPLER](https://arxiv.org/abs/2405.05941)

Gọi (r_i) và (s_i) lần lượt là hiệu năng thực và mô phỏng của policy (i).

### 12.1. Tương quan Pearson

$$
\rho =
\frac{\sum_i(r_i-\bar r)(s_i-\bar s)}
{\sqrt{\sum_i(r_i-\bar r)^2}
 \sqrt{\sum_i(s_i-\bar s)^2}}
$$

Phạm vi: ([-1,1]). Cao hơn là tốt hơn.

- (1): quan hệ tuyến tính dương hoàn hảo;
- (0): không có quan hệ tuyến tính;
- (-1): quan hệ tuyến tính đảo ngược hoàn hảo.

### 12.2. Tương quan thứ hạng Spearman

Pearson áp dụng lên thứ hạng:

$$
\rho_s=\mathrm{corr}(\mathrm{rank}(r),\mathrm{rank}(s))
$$

Phạm vi: ([-1,1]). Cao hơn là tốt hơn. Metric này kiểm tra thứ tự thay vì mức tương hợp tuyến tính.

### 12.3. Mean Maximum Rank Violation (MMRV)

Định nghĩa một vi phạm theo cặp:

$$
V_{ij}=
\begin{cases}
|r_i-r_j|,&(r_i-r_j)(s_i-s_j)<0\\
0,&\text{otherwise}
\end{cases}
$$

Sau đó:

$$
\mathrm{MMRV}=\frac{1}{M}\sum_{i=1}^{M}\max_j V_{ij}
$$

Phạm vi: ([0,1]) khi hiệu năng thực nằm trong ([0,1]). Thấp hơn là tốt hơn. MMRV phạt simulator nặng hơn khi nó đảo thứ hạng của các policy có hiệu năng thực khác nhau đáng kể.

Các metric này đánh giá simulator, không trực tiếp đánh giá VLA.

## 13. Metric ngôn ngữ, lập kế hoạch và grounding

Các metric này được dùng khi VLA hoặc module cấp cao tạo ra văn bản, kỹ năng, tham số, box hoặc kế hoạch.

### 13.1. Skill Recall

$$
\mathrm{SkillRecall}=
\frac{|\text{predicted required skills}\cap\text{required skills}|}
{|\text{required skills}|}
$$

Phạm vi: ([0,1]). Cao hơn là tốt hơn.

### 13.2. Parameter Recall

$$
\mathrm{ParameterRecall}=
\frac{\#\text{correct required arguments}}
{\#\text{required arguments}}
$$

Phạm vi: ([0,1]). Cao hơn là tốt hơn.

### 13.3. Skill-and-parameter recall

$$
\mathrm{PairRecall}=
\frac{\#\text{correct skill-argument pairs}}
{\#\text{required skill-argument pairs}}
$$

Phạm vi: ([0,1]). Cao hơn là tốt hơn.

### 13.4. Precise hoặc Exact Matching Rate

$$
\mathrm{PM}=
\frac{\#\text{outputs exactly matching the required program or plan}}
{N}
$$

Phạm vi: ([0,1]). Cao hơn là tốt hơn.

VLABench dùng các metric này cho đánh giá VLM/thư viện kỹ năng, không phải để thay thế thành công rollout vật lý.

### 13.5. Độ chính xác chọn object

$$
\mathrm{SelectionAcc}=
\frac{\#\text{correct target-object selections}}{N}
$$

Phạm vi: ([0,1]). Cao hơn là tốt hơn.

### 13.6. Bounding-box IoU

$$
\mathrm{IoU}=\frac{|B_{\mathrm{pred}}\cap B_{\mathrm{gt}}|}
{|B_{\mathrm{pred}}\cup B_{\mathrm{gt}}|}
$$

Phạm vi: ([0,1]). Cao hơn là tốt hơn. Được dùng cho các module grounding/localization thay vì điều khiển VLA hoàn chỉnh.

## 14. Metric điều hướng đôi khi được dùng với VLA

### 14.1. Navigation Success Rate

Tỷ lệ episode kết thúc trong dung sai của goal. Phạm vi: ([0,1]). Cao hơn là tốt hơn.

### 14.2. Distance to Goal

$$
d_{\mathrm{goal}}=\|p_{\mathrm{final}}-p_{\mathrm{goal}}\|
$$

Phạm vi: ([0,infinity)) mét. Thấp hơn là tốt hơn.

### 14.3. Success weighted by Path Length (SPL)

$$
\mathrm{SPL}=\frac{1}{N}\sum_{i=1}^{N}
S_i\frac{L_i^*}{\max(L_i,L_i^*)}
$$

trong đó (L_i^*) là đường đi ngắn nhất và (L_i) là đường đi được thực thi.

Phạm vi: ([0,1]). Cao hơn là tốt hơn. SPL bằng 0 đối với episode thất bại và ưu tiên các đường đi thành công, hiệu quả.

### 14.4. Navigation Collision Rate

Số va chạm trên mỗi episode hoặc số episode có va chạm chia cho tổng episode. Cách thứ nhất có phạm vi ([0,infinity)); cách thứ hai là ([0,1]). Thấp hơn là tốt hơn.

## 15. Báo cáo thống kê

### 15.1. Độ lệch chuẩn

$$
s=\sqrt{\frac{1}{K-1}\sum_{k=1}^{K}(x_k-\bar{x})^2}
$$

Đo mức biến thiên giữa các seed, tác vụ hoặc batch đánh giá. Phạm vi: ([0,infinity)). Thấp hơn nghĩa là nhất quán hơn, không nhất thiết hiệu năng trung bình tốt hơn.

### 15.2. Sai số chuẩn

$$
\mathrm{SE}=\frac{s}{\sqrt{K}}
$$

Ước lượng độ bất định của giá trị trung bình được báo cáo. Phạm vi: ([0,infinity)). Giá trị này giảm khi có nhiều lần chạy độc lập hơn.

### 15.3. Khoảng tin cậy nhị thức cho Success Rate

Khoảng xấp xỉ 95% là:

$$
\hat p \pm 1.96\sqrt{\frac{\hat p(1-\hat p)}{N}}
$$

Nên dùng khoảng Wilson hoặc khoảng chính xác khi (N) nhỏ hoặc giá trị gần 0 và 1.

Ví dụ: 8 lần thành công trong 10 lần thử và 800 lần thành công trong 1.000 lần thử đều cho 80%, nhưng ước lượng thứ hai chính xác hơn nhiều.

### 15.4. Số rollout

Luôn báo cáo (N) cho mỗi tác vụ, số tác vụ, số seed và reset có deterministic hay không. Một metric không đi kèm số lượng mẫu là chưa đầy đủ.

## 16. Ánh xạ benchmark chính sang metric

| Benchmark                           | Metric VLA chính                         | Metric bổ sung                                                          | Phạm vi quan trọng                                  |
| ----------------------------------- | ---------------------------------------- | ----------------------------------------------------------------------- | --------------------------------------------------- |
| **LIBERO**                          | Success Rate theo tác vụ và trung bình   | Thiết lập học suốt đời gốc: FWT, NBT, AUC                               | SR ([0,1]); FWT thường ([-1,1]); NBT ([0,1])        |
| **CALVIN**                          | Average completed sequence length        | Success ở độ dài chuỗi 1-5                                              | AvgLen ([0,5]); mỗi prefix SR ([0,1])               |
| **RLBench**                         | Task Success Rate                        | Trung bình trên các tác vụ, phân tích few-shot/theo tác vụ              | ([0,1])                                             |
| **RoboCasa**                        | Success Rate                             | Phân tích seen/unseen, kết quả composite-task                            | ([0,1])                                             |
| **SimplerEnv**                      | Simulated Task Success Rate              | Tương quan Pearson và MMRV so với thứ hạng policy thực                   | SR ([0,1]), Pearson ([-1,1]), MMRV ([0,1])*         |
| **VLABench**                        | Progress Score                           | Điểm seen/unseen; recall kỹ năng/tham số VLM và precise match            | Thường là ([0,1]) hoặc 0-100                        |
| **ManiSkill / Meta-World**          | Success Rate hoặc Return                 | Trung bình theo tác vụ, normalized score                                 | SR ([0,1]); Return tùy theo environment             |
| **Tác vụ robot thực tùy chỉnh**     | Success Rate                             | Completion Time, can thiệp, va chạm, độ chính xác                        | Tùy theo metric                                     |
| **Benchmark VLA điều hướng**        | Success và SPL                           | Distance-to-goal, độ dài đường đi, va chạm                               | Success/SPL ([0,1])                                 |

Các bài báo benchmark gốc là [LIBERO](https://arxiv.org/abs/2306.03310), [CALVIN](https://arxiv.org/abs/2112.03227), [RLBench](https://arxiv.org/abs/1909.12271), [RoboCasa](https://arxiv.org/abs/2406.02523), [SIMPLER](https://arxiv.org/abs/2405.05941) và [VLABench](https://arxiv.org/abs/2412.18194).

## 17. Tập đánh giá tối thiểu được khuyến nghị

Một báo cáo VLA thao tác có chất lượng nên bao gồm ít nhất:

1. **Full-task Success Rate** cho từng tác vụ và macro-average.
2. **Progress Score** cho tác vụ dài hạn.
3. **Seen và unseen Success Rate**, kèm định nghĩa về dịch chuyển phân phối.
4. **Collision Rate hoặc Safety Violation Rate**.
5. **Recovery Rate** hoặc Intervention Rate.
6. **Median và p95 End-to-end Latency** trên phần cứng triển khai.
7. **Control Frequency**, độ dài action chunk và số hành động được thực thi trên mỗi chunk.
8. **Số rollout, seed và khoảng tin cậy**.
9. **Sai số hoặc likelihood của hành động offline**, chỉ dùng làm thông tin chẩn đoán thay vì kết quả chính.
10. **Mức sử dụng tài nguyên tính toán và bộ nhớ** khi tuyên bố về hiệu quả.

## 18. Cách đọc một bảng kết quả VLA

Giả sử một bài báo báo cáo:

```text
LIBERO mean success:        92%
Unseen-scene success:       61%
Collision rate:              8%
Median / p95 latency:       45 / 92 ms
Control frequency:          20 Hz
Đánh giá:                   20 rollout mỗi tác vụ, 10 tác vụ, 3 seed
```

Diễn giải:

- Policy mạnh trên benchmark chính.
- Generalization Gap là (0.92-0.61=0.31), một mức đáng kể.
- Tám phần trăm episode có va chạm ngay cả khi một số episode cuối cùng vẫn thành công.
- Vòng lặp 20 Hz có chu kỳ danh định 50 ms. Median Latency đáp ứng được, nhưng p95 Latency lỡ deadline đó, vì vậy cần thực thi chunk bất đồng bộ hoặc refresh rate thấp hơn.
- Số lượng mẫu đủ lớn để đáng tin cậy hơn kết quả chỉ dựa trên một vài lần thử được chọn thủ công.

Không metric đơn lẻ nào trả lời được một VLA có tốt hay không. Tổ hợp giàu thông tin nhất là:

> **thành công tác vụ + tiến độ một phần + khả năng khái quát hóa + an toàn + hiệu năng thời gian thực + độ bất định thống kê.**
