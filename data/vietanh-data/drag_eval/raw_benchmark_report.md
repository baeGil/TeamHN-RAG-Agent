# Kết quả Đánh giá: Baseline RAG vs DRAG trên data_raw.json

- Dataset: `D:\TeamHN-RAG-Agent\data\vietanh-data\data_raw.json`
- Số câu hỏi đánh giá: **20**
- Số tài liệu ngữ cảnh trong DB: **150**

## 1. Kết quả Tổng hợp

| Phương pháp | Recall@5 | MRR@5 | Hit@5 | RAGAS Avg | Faithfulness | Relevancy | Correctness | Avg Latency | Total Latency | Avg Tokens | Total Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.900 | 0.710 | 0.900 | 0.920 | 0.900 | 0.950 | 0.900 | 4.58s | 91.54s | 1509.3 | 30187 |
| drag | 0.900 | 0.710 | 0.900 | 0.890 | 0.855 | 0.950 | 0.865 | 5.59s | 111.75s | 3172.7 | 63454 |

## 2. So sánh Chênh lệch (Delta: DRAG - Baseline)

| Chỉ số | Chênh lệch (Delta) |
|---|---:|
| Recall@5 | +0.000 |
| MRR@5 | +0.000 |
| Hit@5 | +0.000 |
| RAGAS Avg | -0.030 |
| Avg Latency | +1.010s |
| Avg Tokens | +1663.350 |

## 3. Kết quả Chi tiết từng Câu hỏi

| QID | Phương pháp | Faithfulness | Relevancy | Ctx Precision | Ctx Recall | Correctness | Behavior | Latency | Nhãn dự đoán |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **0001-0001-0006** | **Paris là một thành phố quan trọng của nước Pháp bắt đầu từ thời gian n...** | | | | | | | | |
| | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 21.37s | - |
| | drag | 0.500 | 1.000 | 0.800 | 0.600 | 0.500 | 1.000 | 6.99s | `no_conflict` |
| **0001-0002-0002** | **Người ta đến với Paris vì điều gì?...** | | | | | | | | |
| | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 4.80s | - |
| | drag | 0.800 | 1.000 | 0.800 | 0.600 | 0.800 | 1.000 | 4.78s | `no_conflict` |
| **0001-0003-0005** | **Người dân trên thế giới thường gọi Paris với tên khác là gì?...** | | | | | | | | |
| | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 3.07s | - |
| | drag | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 5.32s | `no_conflict` |
| **0001-0004-0002** | **Hai thành phố nào được gọi là thành phố của tình yêu?...** | | | | | | | | |
| | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2.47s | - |
| | drag | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 4.20s | `no_conflict` |
| **0001-0005-0003** | **Ngoại trừ cái tên Paname, Paris còn có tên gọi khác là gì?...** | | | | | | | | |
| | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 3.27s | - |
| | drag | 0.000 | 1.000 | 0.500 | 0.500 | 0.000 | 0.000 | 4.72s | `complementary_information` |
| **0001-0006-0002** | **Công trình nào nằm trên đỉnh Montmartre?...** | | | | | | | | |
| | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2.41s | - |
| | drag | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 5.02s | `no_conflict` |
| **0001-0007-0004** | **Rừng Vincennes nằm ở đâu của Paris?...** | | | | | | | | |
| | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2.75s | - |
| | drag | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 5.60s | `no_conflict` |
| **0001-0008-0005** | **Bản đồ địa chất được bắt đầu vẽ ở đâu?...** | | | | | | | | |
| | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 4.32s | - |
| | drag | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 7.78s | `no_conflict` |
| **0001-0009-0009** | **Thành phần bên trong lòng đất Paris chủ yếu là gì?...** | | | | | | | | |
| | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 4.94s | - |
| | drag | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 5.86s | `no_conflict` |
| **0001-0010-0002** | **Điểm cuối của kênh Saint-Denis ở đâu?...** | | | | | | | | |
| | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2.97s | - |
| | drag | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 4.80s | `no_conflict` |
| **0001-0011-0004** | **Khí hậu Paris có đặc điểm gì?...** | | | | | | | | |
| | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 3.68s | - |
| | drag | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 5.23s | `no_conflict` |
| **0001-0012-0002** | **Paris trở thành nơi thịnh vượng nhất châu Âu vào thời kì nào?...** | | | | | | | | |
| | baseline | 0.000 | 1.000 | 0.500 | 0.500 | 0.000 | 0.000 | 3.13s | - |
| | drag | 0.800 | 1.000 | 1.000 | 0.800 | 1.000 | 1.000 | 6.13s | `no_conflict` |
| **0001-0013-0003** | **Con người bắt đầu tập trung cư trú nhiều bên Sen vào lúc nào?...** | | | | | | | | |
| | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 3.29s | - |
| | drag | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 7.34s | `no_conflict` |
| **0001-0014-0006** | **Paris có tên gì sau khi bị xâm chiếm bởi đế chế Roma?...** | | | | | | | | |
| | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 5.65s | - |
| | drag | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 4.89s | `no_conflict` |
| **0001-0015-0005** | **Những chiến binh Viking đã tấn công Paris lần đầu vào lúc nào?...** | | | | | | | | |
| | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 4.04s | - |
| | drag | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 4.80s | `no_conflict` |
| **0001-0016-0001** | **Paris đã trở lại làm thủ đô từ vua nào lên ngôi?...** | | | | | | | | |
| | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 3.28s | - |
| | drag | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 5.66s | `no_conflict` |
| **0001-0017-0002** | **Bên bờ phải sông Sen đã phát triển mạnh về hoạt động gì vào thế kỉ XII...** | | | | | | | | |
| | baseline | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3.74s | - |
| | drag | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 6.17s | `no_conflict` |
| **0001-0018-0004** | **Vì sao vua di chuyển đến Hôtel Saint-Pol?...** | | | | | | | | |
| | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 6.32s | - |
| | drag | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 6.04s | `no_conflict` |
| **0001-0019-0003** | **Henry IV đã không thể quay lại Paris cho đến năm nào?...** | | | | | | | | |
| | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 3.08s | - |
| | drag | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 5.27s | `no_conflict` |
| **0001-0020-0002** | **Ai đã thay Louis XIV điều hành Paris?...** | | | | | | | | |
| | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2.96s | - |
| | drag | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 5.15s | `no_conflict` |