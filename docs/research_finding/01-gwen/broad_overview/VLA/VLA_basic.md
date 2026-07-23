# Kiến thức cơ bản về VLA: Từ thị giác và ngôn ngữ đến hành động vật lý

## 1. Ý tưởng chính

Mô hình **Vision-Language-Action (VLA)** là một VLM không chỉ dừng lại ở việc hiểu hoặc trả lời. Mô hình sử dụng những gì robot nhìn thấy và yêu cầu của con người, rồi quyết định **robot nên di chuyển như thế nào**.

So sánh đơn giản nhất:

| Mô hình | Đầu vào | Đầu ra |
| ----- | --------------------------------------------- | ------------ |
| VLM | Hình ảnh + văn bản | Văn bản |
| VLA | Hình ảnh + chỉ dẫn + trạng thái hiện tại của robot | Hành động của robot |

Một định nghĩa một câu hữu ích:

> **VLA biến “tôi nhìn thấy tình huống và hiểu mục tiêu” thành “đây là chuyển động tôi nên thực hiện tiếp theo.”**

## 2. Đầu vào

Tại một thời điểm, mô hình thường nhận:

- **Những gì robot nhìn thấy:** một hoặc nhiều hình ảnh từ camera.
- **Điều con người muốn:** ví dụ: “đặt chiếc cốc màu đỏ vào bồn rửa.”
- **Trạng thái hiện tại của robot:** vị trí của cánh tay và bộ kẹp, bộ kẹp đang mở hay không, đôi khi còn có các hình ảnh hoặc chuyển động gần đây.

Một số hệ thống còn sử dụng xúc giác, độ sâu hoặc âm thanh, nhưng đây là các thành phần bổ sung chứ không phải ý tưởng cốt lõi.

## 3. Luồng dữ liệu khi chạy

Robot liên tục lặp lại chu trình sau:

1. **Quan sát:** nhìn vào cảnh hiện tại.
2. **Liên hệ yêu cầu với cảnh:** xác định chiếc cốc, bồn rửa, chướng ngại vật và vị trí robot có liên quan.
3. **Chọn chuyển động tiếp theo:** vươn tới, nắm, nâng, di chuyển, thả hoặc thực hiện một bước nhỏ khác.
4. **Hành động:** gửi chuyển động đến robot.
5. **Quan sát lại:** kiểm tra những gì đã thay đổi và chọn chuyển động tiếp theo.

Sự lặp lại này là thiết yếu. Mô hình không chỉ tạo một câu trả lời rồi kết thúc. Hành động của mô hình làm thay đổi thế giới, vì vậy nó phải quan sát kết quả và tiếp tục từ tình huống mới.

Một số VLA xuất từng chuyển động nhỏ. Các VLA khác xuất một nhóm chuyển động ngắn, thực thi một phần rồi quan sát lại.

## 4. Đầu ra

Đầu ra cuối cùng phải là thứ robot có thể thực thi. Đối với cánh tay robot, đầu ra có thể mô tả:

- bàn tay nên di chuyển trong không gian 3D như thế nào;
- cổ tay nên xoay như thế nào;
- bộ kẹp nên mở hay đóng.

Đối với robot di động, đầu ra có thể mô tả chuyển động bánh xe, hướng hoặc tốc độ. Vì vậy, đầu ra chính xác phụ thuộc vào thân thể robot.

Không phải VLA nào cũng chuyển trực tiếp từ đầu vào sang lệnh động cơ. Trước tiên, hệ thống có thể tạo ra một kết quả trung gian như:

- một kế hoạch ngắn bằng ngôn ngữ: “nắm chiếc cốc, sau đó di chuyển đến bồn rửa”;
- một điểm đích: “nắm tại đây”;
- một quỹ đạo hoặc hình ảnh tương lai mong muốn.

Sau đó, kết quả trung gian được chuyển thành chuyển động có thể thực thi. Bài khảo sát đính kèm nhóm các dạng khác nhau này dưới khái niệm rộng **action token**: thông tin ngày càng hữu ích hơn cho việc tạo hành động.

## 5. VLA học như thế nào

Dữ liệu huấn luyện trực tiếp nhất là một tập các lần thị phạm. Mỗi thời điểm ghi lại:

| Chỉ dẫn | Những gì robot nhìn thấy | Trạng thái robot | Chuyển động đã thực hiện |
| ----------------------- | ------------------ | ------------------------ | ------------------------ |
| “Nhấc chiếc cốc màu đỏ lên” | Hình ảnh camera | Vị trí cánh tay và bộ kẹp | Di chuyển bàn tay về phía chiếc cốc |

Qua nhiều lần thị phạm, mô hình học rằng: **với yêu cầu và tình huống này, hãy dự đoán chuyển động mà người vận hành thành công đã thực hiện.**

Nền tảng VLM cung cấp kiến thức rộng về thị giác và ngôn ngữ. Các lần thị phạm bằng robot dạy mối liên hệ còn thiếu giữa kiến thức đó và chuyển động vật lý. Dữ liệu robot này khó thu thập và đắt đỏ hơn nhiều so với dữ liệu hình ảnh-văn bản thông thường, đây là một trong những hạn chế chính của nghiên cứu VLA hiện nay.

## 6. Ví dụ cụ thể

Nhiệm vụ: **“Đặt chiếc cốc màu đỏ vào bồn rửa.”**

| Thời điểm | Mô hình nhìn thấy | Mô hình xuất ra |
| ------ | ------------------------ | ------------------------- |
| 1 | Cốc ở xa bàn tay | Di chuyển về phía cốc |
| 2 | Bàn tay ở cạnh cốc | Căn chỉnh các ngón tay quanh cốc |
| 3 | Các ngón tay bao quanh cốc | Đóng bộ kẹp |
| 4 | Cốc đã được giữ | Nâng lên và di chuyển về phía bồn rửa |
| 5 | Cốc ở phía trên bồn rửa | Mở bộ kẹp |

Chỉ dẫn vẫn giữ nguyên, nhưng đầu ra đúng thay đổi sau mỗi lần quan sát.

## 7. Mô hình tư duy cần ghi nhớ

- **VLM** trả lời: “Ở đây có gì và yêu cầu có nghĩa là gì?”
- **VLA** trả lời: “Dựa trên hiểu biết đó, cơ thể này nên làm gì tiếp theo?”
- VLA là một **vòng lặp quan sát-hành động-quan sát liên tục**, không phải nhiệm vụ tạo chú thích một lần.
- Đầu ra của VLA gắn với một thân thể robot cụ thể. Cùng một mục tiêu có thể cần những chuyển động khác nhau trên các robot khác nhau.
- Ví dụ huấn luyện then chốt của VLA là mối quan hệ giữa **chỉ dẫn, quan sát và hành động thành công**.

Theo cách hiểu của Transformer, trực giác khởi đầu đơn giản nhất là: VLA giữ lại khả năng hiểu thị giác-ngôn ngữ quen thuộc, nhưng mở rộng việc dự đoán từ các từ sang các hành động làm thay đổi đầu vào tiếp theo.

## 8. Từ vựng cốt lõi trong nghiên cứu VLA

Các thuật ngữ này không xuất hiện trong mọi bài báo, nhưng tạo thành vốn từ vựng chung của lĩnh vực.

| Thuật ngữ | Nghĩa đơn giản | Ví dụ |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| **Observation** | Mọi thông tin mô hình có thể sử dụng tại thời điểm hiện tại. Thường gồm hình ảnh camera và trạng thái của chính robot. | Hình ảnh hiện tại + vị trí cánh tay |
| **Instruction** | Mô tả bằng ngôn ngữ về nhiệm vụ hoặc mục tiêu. | “Đặt chiếc cốc vào bồn rửa.” |
| **State** | Mô tả tình huống hiện tại. Trạng thái đầy đủ của thế giới thực thường không thể biết được, vì vậy mô hình hành động dựa trên quan sát hạn chế của nó. | Vị trí cốc, vị trí cánh tay, trạng thái bộ kẹp |
| **Proprioception** | Khả năng robot cảm nhận cơ thể của chính nó, thay vì thế giới bên ngoài. | Góc khớp, vị trí bàn tay, độ mở bộ kẹp |
| **Action** | Một lệnh robot có thể thực thi. | Di chuyển bàn tay về phía trước 2 cm và đóng bộ kẹp |
| **Action space** | Toàn bộ tập hợp và định dạng các hành động mà robot có thể thực hiện. | Chuyển động cánh tay, xoay cổ tay và điều khiển bộ kẹp |
| **Policy** | Bộ ra quyết định đã học, ánh xạ quan sát và chỉ dẫn hiện tại thành hành động. Trong hầu hết các bài báo VLA, “policy” chỉ thành phần điều khiển hành vi. | `(image, instruction, robot condition) → next movement` |
| **Embodiment** | Thân thể cụ thể mà thông qua đó mô hình hành động. Các cánh tay, bộ kẹp và robot di động khác nhau có năng lực và action space khác nhau. | Cánh tay có bộ kẹp hai ngón so với robot hình người |
| **Timestep** | Một thời điểm trong quá trình quan sát và hành động lặp lại. | Quan sát tại thời điểm `t`, sau đó dự đoán hành động `t` |
| **Control frequency** | Tần suất hệ thống tạo hoặc cập nhật lệnh robot, đo bằng số lần mỗi giây (Hz). Cao hơn không tự động tốt hơn, nhưng chuyển động tinh thường cần cập nhật thường xuyên. | 10 Hz nghĩa là mười lần cập nhật mỗi giây |
| **Trajectory** | Chuỗi quan sát và hành động có thứ tự theo thời gian. Nó mô tả diễn tiến của hành vi, không chỉ kết quả cuối cùng. | Vươn tới → nắm → nâng → di chuyển → thả |
| **Episode** | Một lần thử hoàn chỉnh cho một nhiệm vụ, từ tình huống ban đầu đến khi thành công, thất bại hoặc hết thời gian. | Một lần thử đặt chiếc cốc vào bồn rửa |
| **Rollout** | Một episode được tạo ra khi chạy policy. Các nhà nghiên cứu kiểm tra rollout để xem mô hình thực sự làm gì. | Chạy mô hình đã huấn luyện trên robot trong một lần thử |
| **Demonstration** | Một ví dụ về hành vi thành công hoặc hữu ích, thường được thu thập từ người điều khiển robot hoặc một bộ điều khiển khác. | Con người điều khiển từ xa cánh tay để nhấc một chiếc cốc |
| **Imitation learning / behavior cloning** | Huấn luyện mô hình dự đoán các hành động có trong demonstration. Cách này tương tự học có giám sát trên các cặp quan sát-hành động. | Học cách sao chép chuyển động tiếp theo của người vận hành |
| **Action token** | Một đơn vị đầu ra tổng quát liên quan đến hành động. Tùy bài báo, nó có thể là lệnh động cơ trực tiếp, điểm đích, quỹ đạo, hình ảnh mục tiêu hoặc biểu diễn nội bộ. | “Đóng bộ kẹp”, một điểm nắm 3D hoặc một giá trị chuyển động |
| **Action chunk** | Nhiều hành động tương lai được dự đoán cùng lúc thay vì từng hành động một. Cách này có thể giúp điều khiển nhanh và mượt hơn, nhưng các hành động về sau có thể trở nên lỗi thời nếu thế giới thay đổi. | Dự đoán cùng lúc 20 lệnh cánh tay tiếp theo |
| **Closed-loop control** | Robot liên tục quan sát kết quả hành động và điều chỉnh. | Di chuyển về phía cốc, quan sát lại, rồi hiệu chỉnh căn chỉnh |
| **Open-loop control** | Robot thực thi một chuỗi hành động đã chuẩn bị mà không dùng quan sát mới để hiệu chỉnh trong khi thực thi. | Thực thi cả 20 lệnh dự đoán mà không quan sát lại |
| **Generalization** | Khả năng thành công ngoài đúng những demonstration đã dùng để huấn luyện. | Nhấc một chiếc cốc mới trong một căn bếp mới |
| **Success rate** | Tỷ lệ số lần thử nhiệm vụ hoàn thành thành công. Đây là thước đo đánh giá cấp cao phổ biến nhất. | 82 episode thành công trên 100 = 82% |

### 8.1. Thuật ngữ về cánh tay robot

| Thuật ngữ | Nghĩa đơn giản |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **Joint** | Một khớp nối có thể chuyển động trong cánh tay robot. Điều khiển theo khớp chỉ định trực tiếp cách các khớp này nên chuyển động. |
| **End effector** | Bộ phận làm việc ở đầu cánh tay, thường là bàn tay, bộ kẹp hoặc công cụ. |
| **Pose** | Vị trí cộng với hướng. Pose của end effector cho biết cả vị trí của bàn tay lẫn hướng nó đang trỏ tới. |
| **Degrees of freedom (DoF)** | Số chiều chuyển động có thể điều khiển độc lập. Pose bàn tay 6-DoF thường có ba chiều vị trí và ba chiều xoay. |
| **Gripper action** | Lệnh điều khiển công cụ gắp của robot, thường là mở/đóng hoặc một giá trị độ mở liên tục. |

### 8.2. Tóm tắt vốn từ vựng trong một câu

> Tại mỗi **timestep**, **policy** VLA nhận một **observation**, **instruction** bằng ngôn ngữ và **proprioception**, sau đó dự đoán một **action** hoặc **action chunk** trong **action space** của robot; việc lặp lại quá trình này tạo ra một **rollout trajectory**, với kết quả được đo bằng **success rate**.

## 9. Những thách thức và ràng buộc cốt lõi

VLA phải kết hợp hiểu biết ngữ nghĩa với khả năng điều khiển vật lý chính xác, kịp thời và an toàn. Mô hình có thể hiểu đúng chỉ dẫn nhưng vẫn thất bại vì chuyển động quá trễ, thiếu chính xác, không an toàn, không tương thích với robot hoặc không thể phục hồi sau một lỗi nhỏ.

Các nhóm thách thức chính:

| Lĩnh vực | Câu hỏi trọng tâm |
| --------------------- | ----------------------------------------------------------------------------- |
| Dữ liệu | Dữ liệu huấn luyện có bao phủ nhiệm vụ, cảnh, lỗi và thân thể robot không? |
| Nhận thức | Mô hình có thể xác định đúng vật thể và vị trí 3D có thể thao tác của nó không? |
| Grounding ngôn ngữ | Mô hình có liên hệ chỉ dẫn với đúng vật thể, mối quan hệ và hành vi không? |
| Sinh hành động | Các lệnh có mượt, chính xác và hợp lệ với robot này không? |
| Điều khiển thời gian thực | Hành động có được tạo ra trước khi quan sát trở nên lỗi thời không? |
| Hành vi dài hạn | Mô hình có thể theo dõi tiến độ, sắp xếp các nhiệm vụ con và phục hồi sau lỗi không? |
| Khái quát hóa | Mô hình có hoạt động ngoài đúng những điều kiện có trong dữ liệu huấn luyện không? |
| An toàn | Có thể phát hiện và khống chế lỗi trước khi gây thiệt hại không? |
| Đánh giá | Chúng ta có đang đo thành công thực tế của nhiệm vụ thay vì chỉ đo lỗi dự đoán offline không? |

### 9.1. Ràng buộc về dữ liệu và học

#### 9.1.1. Dữ liệu robot khan hiếm và đắt đỏ

Dữ liệu hình ảnh-văn bản có thể được thu thập từ Internet, nhưng dữ liệu huấn luyện robot hữu ích thường cần robot thật, trình mô phỏng hoặc người vận hành. Việc thu thập demonstration trong thế giới thực chậm và có thể gặp hao mòn phần cứng, sự thiếu nhất quán giữa người vận hành, lỗi cảm biến và các episode thất bại.

Ràng buộc quan trọng là VLA không thể học kỹ năng vật lý chỉ từ kiến thức VLM. VLM có thể biết cốc có quai, nhưng dữ liệu robot phải dạy thân thể robot này cách tiếp cận, nắm, nâng và phục hồi nếu cốc bị trượt.

Các cách xử lý phổ biến gồm:

- gộp dữ liệu từ nhiều robot;
- tạo dữ liệu mô phỏng hoặc tổng hợp;
- học từ video con người;
- pretrain trên dữ liệu rộng, sau đó fine-tune bằng một tập dữ liệu nhỏ hơn dành riêng cho robot;
- thu thập hành vi hiệu chỉnh và phục hồi, không chỉ các demonstration hoàn hảo.

#### 9.1.2. Chất lượng và độ bao phủ dữ liệu quan trọng không kém dung lượng

Một dataset chỉ chứa các demonstration thành công và sạch sẽ dạy policy hành vi lý tưởng trông như thế nào, nhưng không dạy cách phục hồi sau khi lệch khỏi hành vi đó. Khi triển khai, các lỗi nhỏ đưa robot vào những tình huống có thể không tồn tại trong demonstration. Hiện tượng này được gọi là **distribution shift**.

Dữ liệu huấn luyện nên bao phủ:

- các vị trí, diện mạo và phông nền khác nhau của vật thể;
- sự thay đổi về ánh sáng, góc nhìn camera và độ lộn xộn;
- nhiều cách hợp lệ để hoàn thành nhiệm vụ;
- các lỗi một phần và chuyển động hiệu chỉnh;
- phần bắt đầu, chuyển tiếp và kết thúc nhiệm vụ;
- hành vi dừng an toàn khi nhiệm vụ trở nên bất khả thi.

#### 9.1.3. Dữ liệu đa thân thể không tự động tương thích

Các robot khác nhau có camera, số lượng khớp, bộ kẹp, hệ tọa độ và ý nghĩa hành động khác nhau. Việc kết hợp dữ liệu của chúng cần một biểu diễn chung hoặc các adapter dành riêng cho robot.

Ví dụ, hành động số `[0.1, 0, 0]` có thể mang nghĩa là chuyển động Descartes 10 cm trong một dataset, lệnh khớp đã chuẩn hóa trong dataset khác và vận tốc trong dataset thứ ba. Trộn chúng mà không có metadata và phép chuyển đổi đúng sẽ khiến mục tiêu huấn luyện trở nên vô nghĩa.

#### 9.1.4. Kiến thức VLM có thể bị lãng quên

Fine-tune mạnh trên các trajectory robot có thể làm suy giảm năng lực thị giác-ngôn ngữ tổng quát kế thừa từ VLM ban đầu. Mô hình có thể cải thiện trên một benchmark robot nhưng lại nhận diện vật thể hiếm, đọc nhãn hoặc diễn giải chỉ dẫn mới kém đi.

Do đó, các công thức huấn luyện gần đây trộn dữ liệu thị giác-ngôn ngữ với dữ liệu robot, đóng băng hoặc tách biệt một phần VLM, hoặc dùng một action expert riêng để việc học vận động không ghi đè toàn bộ kiến thức ngữ nghĩa.

### 9.2. Thách thức về nhận thức và grounding

#### 9.2.1. Hiểu biết 2D không đủ cho thao tác chính xác

Hình ảnh RGB cung cấp nhiều thông tin ngữ nghĩa nhưng không trực tiếp cho biết độ sâu chính xác, hình học tiếp xúc hoặc khối lượng vật thể. VLA phải suy ra vị trí tiếp xúc, khoảng cách di chuyển và liệu bộ kẹp có thể vươn tới mà không va chạm hay không.

Điều này trở nên khó khăn với:

- vật thể trong suốt, phản chiếu, biến dạng hoặc rất nhỏ;
- cảnh lộn xộn và che khuất;
- các vật thể có vẻ ngoài giống nhau;
- ánh sáng kém hoặc nhòe chuyển động;
- nhiệm vụ cần căn chỉnh ở cấp milimét;
- góc nhìn camera không hiển thị vùng tiếp xúc.

Camera độ sâu, camera cổ tay, point cloud, cảm biến lực và hình học đã hiệu chuẩn có thể hỗ trợ, nhưng mỗi thành phần đều làm phát sinh ràng buộc về phần cứng và đồng bộ hóa.

#### 9.2.2. Ngôn ngữ phải được grounding vào cảnh hiện tại

Mô hình phải xác định chỉ dẫn đang đề cập đến vật thể vật lý và mối quan hệ nào. “Đặt nó ở đó”, “dùng chiếc cốc sạch” hoặc “di chuyển khối gần nhất” đòi hỏi ngữ cảnh, bộ nhớ, so sánh hoặc làm rõ.

VLA có thể thất bại ngay cả khi nhận diện được mọi vật thể nếu chọn sai cá thể, hiểu sai trái/phải theo góc nhìn người dùng hoặc diễn giải từ “trên” khác với bộ đánh giá nhiệm vụ.

Chỉ dẫn dùng khi triển khai nên phù hợp với năng lực và mức độ đa dạng ngôn ngữ có trong dữ liệu huấn luyện. Sự mơ hồ ảnh hưởng đến an toàn nên kích hoạt yêu cầu làm rõ hoặc từ chối thay vì một chuyển động đầy tự tin.

#### 9.2.3. Quan sát một phần và bộ nhớ theo thời gian

Hình ảnh camera hiện tại có thể không tiết lộ mọi thông tin cần thiết cho nhiệm vụ. Vật thể có thể di chuyển ra sau cánh tay, ngăn kéo có thể che giấu đồ bên trong hoặc robot có thể cần nhớ những vật đã xử lý.

VLA chỉ dùng một frame có thể liên tục mở cùng một ngăn kéo hoặc quên rằng nó đã hoàn thành một nhiệm vụ con. Lịch sử hình ảnh dài hơn, bộ nhớ tường minh, theo dõi vật thể và theo dõi trạng thái cấp cao có thể hỗ trợ, nhưng làm tăng tính toán và độ dài ngữ cảnh.

### 9.3. Ràng buộc về sinh hành động và điều khiển

#### 9.3.1. Biểu diễn hành động phải khớp với bộ điều khiển

Trước khi huấn luyện hoặc triển khai, giao diện hành động phải được định nghĩa chính xác:

| Lựa chọn | Ví dụ |
| --------------- | --------------------------------------------------------------------------- |
| Mục tiêu điều khiển | Vị trí khớp, vận tốc khớp, pose của end effector hoặc chuyển động của end effector |
| Hệ quy chiếu | Đế robot, thế giới, camera hoặc hệ quy chiếu end effector |
| Ý nghĩa thời gian | Mục tiêu tuyệt đối, thay đổi so với giá trị hiện tại hoặc vận tốc mỗi giây |
| Định dạng phép xoay | Góc Euler, quaternion, axis-angle hoặc phép xoay 6D |
| Định dạng bộ kẹp | Mở/đóng nhị phân, độ rộng liên tục, lực hoặc vận tốc |
| Đơn vị | Mét so với milimét; radian so với độ |

Sự không khớp có thể khiến một mô hình tốt trông như hoàn toàn hỏng và cũng có thể làm hư hại phần cứng. Thống kê chuẩn hóa hành động phải lấy từ đúng dataset và phải được đảo ngược chính xác trước khi gửi lệnh đến robot.

#### 9.3.2. Độ mượt so với khả năng phản ứng

Dự đoán một action chunk tạo ra chuyển động mượt hơn và giảm tần suất phải chạy VLA lớn. Tuy nhiên, một chunk dài có thể tiếp tục di chuyển về phía mục tiêu cũ sau khi vật thể hoặc robot đã thay đổi vị trí.

Sự đánh đổi:

- **chunk dài:** tốc độ thực thi trung bình nhanh hơn và trajectory mượt hơn, nhưng phản ứng kém hơn;
- **chunk ngắn:** phản ứng tốt hơn, nhưng cần inference thường xuyên hơn và có thể tạo ra sự gián đoạn giữa các chunk.

Hầu hết hệ thống dự đoán một chunk, chỉ thực thi một phần, quan sát lại rồi thay thế phần còn lại chưa thực thi.

#### 9.3.3. Nhiều hành động đều có thể đúng

Có thể tồn tại nhiều cách nắm và quỹ đạo hợp lệ cho cùng một observation. Lấy trung bình đơn giản các demonstration đó có thể tạo ra một chuyển động không hợp lệ nằm giữa chúng. Ví dụ, lấy trung bình một cách tiếp cận từ bên trái và một cách tiếp cận từ bên phải có thể khiến bàn tay đâm thẳng vào vật thể.

Các bộ sinh hành động dựa trên diffusion và flow matching phổ biến một phần vì chúng có thể biểu diễn các phân phối hành động liên tục phức tạp. Dù vậy, các chunk được sinh ra vẫn phải nhất quán theo thời gian.

#### 9.3.4. Lỗi nhỏ tích lũy

Sai số 2 mm có thể vô hại khi tiếp cận nhưng mang tính quyết định khi cắm phích cắm hoặc kéo khóa. Việc lặp lại các dự đoán hơi sai có thể dần đưa robot ra xa khỏi phân phối huấn luyện.

Do đó, quan sát closed-loop, hiệu chỉnh thị giác, phản hồi lực, hiệu chuẩn chính xác và dữ liệu phục hồi cũng quan trọng như dự đoán hành động ban đầu.

### 9.4. Hiệu năng thời gian thực

Hiệu năng thời gian thực không đơn giản là “mức sử dụng GPU cao” hoặc số lượng lớn hành động được sinh ra mỗi giây. Điều quan trọng là tổng độ trễ từ cảm nhận đến lệnh robot hữu ích:

```text
phơi sáng và truyền dữ liệu camera
  + tiền xử lý hình ảnh
  + inference VLM
  + lập kế hoạch, nếu có
  + sinh hành động
  + giao tiếp qua mạng hoặc giữa các tiến trình
  + độ trễ bộ điều khiển
  = độ trễ điều khiển end-to-end
```

Chu kỳ điều khiển xác định quỹ thời gian sẵn có:

| Tần suất cập nhật mong muốn | Chu kỳ tối đa mỗi lần cập nhật |
| ------------------: | ------------------------: |
| 10 Hz | 100 ms |
| 20 Hz | 50 ms |
| 50 Hz | 20 ms |
| 120 Hz | khoảng 8,3 ms |

Đây là các quỹ thời gian, không phải yêu cầu chung cho mọi VLA. VLA lớn có thể chạy ở tần suất thấp hơn trong khi bộ điều khiển cấp thấp gọn nhẹ nội suy hoặc thực thi một action chunk dự đoán ở tần suất cao hơn.

#### 9.4.1. Vì sao độ trễ gây ra lỗi vật lý

Nếu hình ảnh camera được chụp tại thời điểm `t` nhưng lệnh đến robot muộn hơn nhiều, lệnh đó dựa trên một quan sát đã lỗi thời. Trong khi đó, cánh tay, vật thể mục tiêu hoặc con người có thể đã di chuyển. Điều này có thể gây dao động, vượt quá mục tiêu, va chạm, chuyển tiếp giật cục hoặc hiệu chỉnh lặp đi lặp lại.

Các phép đo quan trọng gồm:

- độ trễ trung vị và trường hợp xấu nhất từ cảm nhận đến hành động;
- độ trễ VLM và action head riêng biệt;
- control frequency thực tế trên robot thật;
- số lần lỡ deadline và độ biến thiên độ trễ, còn gọi là jitter;
- thời gian truyền hình ảnh và tensor;
- observation đã cũ bao lâu tại thời điểm hành động tương ứng được thực thi.

#### 9.4.2. Các chiến lược thời gian thực phổ biến

- Cache đặc trưng chỉ dẫn và hình ảnh không cần tính toán lại.
- Dự đoán song song nhiều hành động dưới dạng một chunk.
- Sử dụng action expert nhỏ hơn sau một lượt xử lý ngữ cảnh bằng VLM lớn.
- Lượng tử hóa hoặc biên dịch mô hình khi validation cho thấy hành vi chấp nhận được.
- Giảm số lượng hoặc độ phân giải hình ảnh một cách thận trọng.
- Chạy inference bất đồng bộ trong khi robot thực thi chunk hiện tại.
- Tái sử dụng một phần chunk trước để duy trì tính liên tục.
- Đặt các vòng lặp an toàn và điều khiển động cơ nhanh bên ngoài VLA lớn.

Thực thi bất đồng bộ cải thiện throughput nhưng khiến việc căn chỉnh thời gian khó hơn. Hệ thống phải biết timestep được dự đoán nào tương ứng với trạng thái thực tế của robot khi chunk mới bắt đầu.

### 9.5. Lập kế hoạch dài hạn và phục hồi

#### 9.5.1. Nhiệm vụ dài khuếch đại mọi điểm yếu

Nếu mỗi nhiệm vụ con thành công trong 95% số lần, một nhiệm vụ cần 20 nhiệm vụ con độc lập đều thành công sẽ có xác suất thành công tổng thể lý tưởng hóa xấp xỉ:

$$
0.95^{20} \approx 0.36
$$

Các nhiệm vụ con thực tế không độc lập, nhưng ví dụ cho thấy vì sao một policy ngắn hạn mạnh vẫn có thể hoạt động kém khi dọn phòng hoặc chuẩn bị bữa ăn.

Hệ thống dài hạn cần xác định:

- nhiệm vụ con nào nên diễn ra tiếp theo;
- nhiệm vụ con hiện tại đã hoàn thành hay chưa;
- những gì đã hoàn thành;
- khi nào nên thử lại, chọn chiến lược khác, yêu cầu trợ giúp hoặc dừng.

Planner tường minh có thể cải thiện khả năng theo dõi tiến độ nhưng bổ sung một nguồn lỗi và độ trễ khác. Policy end-to-end ngầm định đơn giản hơn nhưng khó kiểm tra và có thể lặp lại hành vi.

#### 9.5.2. Khả năng phục hồi phải được học hoặc thiết kế

Robot nên phát hiện và phục hồi trong những trường hợp như:

- nắm thất bại;
- vật thể bị rơi hoặc di chuyển;
- quỹ đạo bị chặn;
- ngăn kéo không mở;
- vật thể bị cánh tay robot che khuất;
- bộ điều khiển hết thời gian chờ;
- tương tác bất ngờ với con người.

Mô hình chỉ được huấn luyện trên trajectory thành công thường tiếp tục như thể hành động thất bại đã thành công. Dữ liệu phục hồi, bộ phát hiện thành công, giới hạn số lần thử lại, kiểm tra tiến độ và hành vi fallback an toàn nên là một phần của thiết kế hệ thống.

### 9.6. Ràng buộc về khái quát hóa và triển khai

#### 9.6.1. Khái quát hóa có nhiều cấp độ

Mô hình có thể khái quát sang vị trí mới nhưng không phải vật thể mới, hoặc sang vật thể mới nhưng không phải robot mới. Bài báo nên nêu rõ biến thể nào chưa từng xuất hiện:

- cá thể vật thể mới;
- loại vật thể mới;
- phông nền hoặc ánh sáng mới;
- vị trí camera mới;
- cách diễn đạt chỉ dẫn mới;
- tổ hợp nhiệm vụ mới;
- môi trường mới;
- thân thể robot mới.

Việc chia ngẫu nhiên các frame gần nhau từ cùng một trajectory vào tập huấn luyện và đánh giá có thể tạo ra ước tính khái quát hóa cao một cách phi thực tế. Quá trình đánh giá nên tách biệt toàn bộ cảnh, nhiệm vụ, vật thể hoặc lượt chạy robot.

#### 9.6.2. Mô phỏng không chuyển hoàn hảo sang thực tế

Mô phỏng có thể khác về texture, ánh sáng, ma sát, hành vi tiếp xúc, nhiễu camera và độ trễ actuator. Policy có thể đạt điểm cao trong mô phỏng nhưng thất bại trên phần cứng thật.

Domain randomization, fine-tune trong thế giới thực, mô hình cảm biến chính xác và bộ điều khiển thận trọng làm giảm khoảng cách này nhưng không loại bỏ hoàn toàn.

#### 9.6.3. Phần cứng và hiệu chuẩn là một phần của hệ thống mô hình

Hiệu năng VLA phụ thuộc vào các yếu tố bên ngoài mạng neural:

- hiệu chuẩn camera và độ ổn định khi lắp đặt;
- đồng bộ timestamp;
- động học robot và tinh chỉnh bộ điều khiển;
- lực bộ kẹp và độ mòn cơ học;
- độ tin cậy của mạng;
- bộ nhớ GPU sẵn có và giới hạn nhiệt;
- dừng khẩn cấp và phát hiện va chạm.

Camera bị dịch chuyển sau khi hiệu chuẩn hoặc một bộ kẹp khác có thể làm mất hiệu lực mối quan hệ đã học giữa pixel và hành động.

### 9.7. An toàn và độ tin cậy

VLA có tính xác suất và có thể tạo ra lệnh không mong muốn ngay cả với đầu vào quen thuộc. Hệ thống đã triển khai không nên gửi trực tiếp các dự đoán không bị ràng buộc đến động cơ.

Các lớp an toàn phổ biến gồm:

- giới hạn khớp, vận tốc, gia tốc và lực;
- ràng buộc workspace và tự va chạm;
- phát hiện con người và vùng được bảo vệ;
- kiểm tra tính hợp lệ của hành động;
- watchdog timeout;
- dừng khẩn cấp;
- kiểm tra độ tin cậy hoặc bất định;
- con người phê duyệt các hành động rủi ro cao;
- log liên kết observation, dự đoán và lệnh đã thực thi.

Lớp an toàn nên chạy độc lập và nhanh hơn VLA. Không được cho phép chỉ dẫn bằng ngôn ngữ ghi đè các giới hạn an toàn vật lý.

### 9.8. “Độ chính xác” có nghĩa là gì đối với VLA?

Không tồn tại một con số độ chính xác VLA duy nhất tương tự độ chính xác phân loại hình ảnh.

#### 9.8.1. Lỗi dự đoán hành động offline

Trong quá trình huấn luyện, nhà nghiên cứu có thể đo mức độ gần nhau giữa hành động dự đoán và demonstration đã ghi bằng L1, L2 hoặc token accuracy. Phép đo này hữu ích cho việc tối ưu hóa nhưng không đủ để đánh giá robot.

Lỗi offline thấp vẫn có thể tạo ra rollout kém vì:

- nhiều hành động khác nhau đều có thể hợp lệ;
- một lỗi nhỏ có thể tích lũy sau nhiều bước;
- mô hình được đánh giá trên trạng thái demonstration thay vì trạng thái do chính lỗi của nó tạo ra;
- hành động gần với hành động của người vận hành vẫn có thể vượt qua ranh giới tiếp xúc hoặc an toàn.

#### 9.8.2. Tỷ lệ thành công của nhiệm vụ

Metric cấp cao chính thường là:

$$
\text{Success rate} = \frac{\text{successful episodes}}{\text{total evaluated episodes}}
$$

Tiêu chí thành công phải được định nghĩa chính xác. “Đặt chiếc cốc lên khay” có thể yêu cầu cốc nằm hoàn toàn trong vùng được đánh dấu, đứng thẳng, đã được thả và ổn định trong vài giây.

Chỉ riêng success rate cũng có thể che giấu những khác biệt quan trọng. Một đánh giá đầy đủ nên gồm:

| Metric | Điều nó cho biết |
| ---------------------------------------- | ------------------------------------------- |
| Thành công toàn bộ nhiệm vụ | Mục tiêu cuối cùng đã hoàn thành hay chưa |
| Hoàn thành nhiệm vụ con hoặc giai đoạn | Nhiệm vụ dài thất bại ở đâu |
| Sai số vị trí và phép xoay | Độ chính xác vật lý tại mục tiêu |
| Thời gian hoàn thành | Hiệu quả |
| Độ dài quỹ đạo hoặc độ mượt chuyển động | Hành vi lãng phí hoặc giật cục |
| Tỷ lệ va chạm và vi phạm an toàn | Rủi ro |
| Tỷ lệ can thiệp | Tần suất con người phải giải cứu robot |
| Thành công khi phục hồi | Mô hình có thể hiệu chỉnh một bước thất bại hay không |
| Thành công trong điều kiện chưa từng thấy | Khả năng khái quát hóa |
| Độ trễ end-to-end và control frequency | Tính khả thi trong thời gian thực |

#### 9.8.3. Độ chính xác phụ thuộc vào nhiệm vụ

Độ chính xác không gian cần thiết thay đổi đáng kể theo nhiệm vụ. Di chuyển khăn vào giỏ có thể chấp nhận sai số hàng centimet. Cắm đầu nối có thể cần độ chính xác vị trí ở cấp milimét và căn chỉnh phép xoay chặt chẽ.

Vì vậy, success rate trong một bài báo luôn phải được diễn giải cùng với:

- độ khó và dung sai của nhiệm vụ;
- việc đánh giá diễn ra trong mô phỏng hay trên robot thật;
- số episode và random seed được sử dụng;
- cảnh, vật thể và chỉ dẫn có thực sự chưa từng xuất hiện hay không;
- con người có reset, hỗ trợ hoặc chọn các lượt thử thuận lợi hay không.

### 9.9. Danh sách kiểm tra các ràng buộc thực tế

Trước khi huấn luyện hoặc chạy VLA, hãy xác minh:

- **Hợp đồng đầu vào:** số lượng và thứ tự camera, độ phân giải, định dạng màu, crop và độ dài lịch sử.
- **Hợp đồng thời gian:** timestamp camera, tần suất mô hình, tần suất bộ điều khiển, độ dài chunk và độ trễ thực thi.
- **Hợp đồng trạng thái:** thứ tự khớp, cảm biến bị thiếu, đơn vị, chuẩn hóa và hệ tọa độ.
- **Hợp đồng hành động:** lệnh tuyệt đối hay tương đối, chế độ điều khiển, định dạng phép xoay, đơn vị và ngữ nghĩa bộ kẹp.
- **Hợp đồng thân thể:** ID robot, động học, workspace và adapter dành riêng cho embodiment.
- **Hợp đồng ngôn ngữ:** prompt template, nhiệm vụ được hỗ trợ, cách xử lý mơ hồ và lệnh dừng.
- **Hợp đồng tính toán:** bộ nhớ GPU, độ trễ trong trường hợp xấu nhất, độ chính xác số và độ ổn định nhiệt.
- **Hợp đồng an toàn:** giới hạn, kiểm tra va chạm, watchdog, dừng khẩn cấp và giám sát của con người.
- **Hợp đồng đánh giá:** định nghĩa chính xác về thành công, phép chia dữ liệu chưa từng thấy, số rollout, loại lỗi và phép đo độ trễ.
- **Hợp đồng ghi log:** lưu observation đầu vào, đầu ra mô hình, lệnh đã giải mã, lệnh đã thực thi, timestamp và kết quả cho mọi phân tích lỗi.

## 10. Nguồn

Chủ yếu dựa trên *A Survey on Vision-Language-Action Models: An Action Tokenization Perspective*, đặc biệt là khung thống nhất ở trang 1, phần tổng quan về action token ở trang 12 và phần thảo luận về hành động robot trực tiếp ở trang 31-36. Phần thảo luận về thời gian thực và phương pháp sinh hành động hiện đại cũng tham khảo các báo cáo gốc [π0](https://arxiv.org/abs/2410.24164), [GR00T N1](https://arxiv.org/abs/2503.14734), [OpenVLA-OFT](https://arxiv.org/abs/2502.19645) và [Xiaomi-Robotics-0](https://arxiv.org/abs/2602.12684).
