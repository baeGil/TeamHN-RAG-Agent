# Kết quả Đánh giá: Baseline RAG vs DRAG trên data_raw.json

- Dataset: `D:\TeamHN-RAG-Agent\data\vietanh-data\data_raw.json`
- Số câu hỏi đánh giá: **20**
- Số tài liệu ngữ cảnh trong DB: **150**

## 1. Kết quả Tổng hợp

| Phương pháp | Recall@5 | MRR@5 | Hit@5 | RAGAS Avg | Faithfulness | Relevancy | Correctness | Avg Latency | Total Latency | Avg Tokens | Total Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.900 | 0.710 | 0.900 | 0.950 | 0.950 | 0.950 | 0.950 | 3.19s | 63.74s | 1513.1 | 30262 |
| drag | 0.900 | 0.710 | 0.900 | 0.925 | 0.925 | 0.975 | 0.900 | 6.06s | 121.16s | 3174.5 | 63490 |

## 2. So sánh Chênh lệch (Delta: DRAG - Baseline)

| Chỉ số | Chênh lệch (Delta) |
|---|---:|
| Recall@5 | +0.000 |
| MRR@5 | +0.000 |
| Hit@5 | +0.000 |
| RAGAS Avg | -0.025 |
| Avg Latency | +2.871s |
| Avg Tokens | +1661.400 |

## 3. Kết quả Chi tiết từng Câu hỏi

| QID | Phương pháp | Faithfulness | Relevancy | Ctx Precision | Ctx Recall | Correctness | Behavior | Latency | Nhãn dự đoán |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **0001-0001-0006** | **Paris là một thành phố quan trọng của nước Pháp bắt đầu từ thời gian n...** | | | | | | | | |
| | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 23.19s | - |
| | drag | 0.500 | 1.000 | 1.000 | 0.500 | 0.000 | 0.000 | 7.63s | `no_conflict` |
| **0001-0002-0002** | **Người ta đến với Paris vì điều gì?...** | | | | | | | | |
| | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.97s | - |
| | drag | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 5.89s | `no_conflict` |
| **0001-0003-0005** | **Người dân trên thế giới thường gọi Paris với tên khác là gì?...** | | | | | | | | |
| | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2.08s | - |
| | drag | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 5.17s | `no_conflict` |
| **0001-0004-0002** | **Hai thành phố nào được gọi là thành phố của tình yêu?...** | | | | | | | | |
| | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2.40s | - |
| | drag | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 6.22s | `no_conflict` |
| **0001-0005-0003** | **Ngoại trừ cái tên Paname, Paris còn có tên gọi khác là gì?...** | | | | | | | | |
| | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2.44s | - |
| | drag | 0.000 | 0.500 | 0.500 | 0.500 | 0.000 | 0.000 | 5.63s | `complementary_information` |
| **0001-0006-0002** | **Công trình nào nằm trên đỉnh Montmartre?...** | | | | | | | | |
| | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.76s | - |
| | drag | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 6.90s | `no_conflict` |
| **0001-0007-0004** | **Rừng Vincennes nằm ở đâu của Paris?...** | | | | | | | | |
| | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 5.11s | - |
| | drag | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 5.79s | `no_conflict` |
| **0001-0008-0005** | **Bản đồ địa chất được bắt đầu vẽ ở đâu?...** | | | | | | | | |
| | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2.04s | - |
| | drag | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 6.21s | `no_conflict` |
| **0001-0009-0009** | **Thành phần bên trong lòng đất Paris chủ yếu là gì?...** | | | | | | | | |
| | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.78s | - |
| | drag | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 6.43s | `no_conflict` |
| **0001-0010-0002** | **Điểm cuối của kênh Saint-Denis ở đâu?...** | | | | | | | | |
| | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2.10s | - |
| | drag | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 6.69s | `no_conflict` |
| **0001-0011-0004** | **Khí hậu Paris có đặc điểm gì?...** | | | | | | | | |
| | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2.46s | - |
| | drag | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 5.16s | `no_conflict` |
| **0001-0012-0002** | **Paris trở thành nơi thịnh vượng nhất châu Âu vào thời kì nào?...** | | | | | | | | |
| | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2.23s | - |
| | drag | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 7.78s | `no_conflict` |
| **0001-0013-0003** | **Con người bắt đầu tập trung cư trú nhiều bên Sen vào lúc nào?...** | | | | | | | | |
| | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.79s | - |
| | drag | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 5.86s | `no_conflict` |
| **0001-0014-0006** | **Paris có tên gì sau khi bị xâm chiếm bởi đế chế Roma?...** | | | | | | | | |
| | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.55s | - |
| | drag | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 4.67s | `no_conflict` |
| **0001-0015-0005** | **Những chiến binh Viking đã tấn công Paris lần đầu vào lúc nào?...** | | | | | | | | |
| | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.47s | - |
| | drag | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 4.16s | `no_conflict` |
| **0001-0016-0001** | **Paris đã trở lại làm thủ đô từ vua nào lên ngôi?...** | | | | | | | | |
| | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.72s | - |
| | drag | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 6.80s | `conflicting_opinions` |
| **0001-0017-0002** | **Bên bờ phải sông Sen đã phát triển mạnh về hoạt động gì vào thế kỉ XII...** | | | | | | | | |
| | baseline | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.77s | - |
| | drag | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 6.47s | `no_conflict` |
| **0001-0018-0004** | **Vì sao vua di chuyển đến Hôtel Saint-Pol?...** | | | | | | | | |
| | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.85s | - |
| | drag | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 5.50s | `no_conflict` |
| **0001-0019-0003** | **Henry IV đã không thể quay lại Paris cho đến năm nào?...** | | | | | | | | |
| | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.68s | - |
| | drag | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 4.61s | `no_conflict` |
| **0001-0020-0002** | **Ai đã thay Louis XIV điều hành Paris?...** | | | | | | | | |
| | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2.36s | - |
| | drag | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 7.60s | `no_conflict` |