# So sánh câu trả lời Baseline vs DRAG

- PDF: `D:\TeamHN-RAG-Agent\data\vietanh-data\DRAG.pdf`
- Dataset: `D:\TeamHN-RAG-Agent\data\vietanh-data\drag_eval\drag_mock_qa.jsonl`
- Số câu hỏi: **1**

File này dùng để đọc thủ công: mỗi câu có nhãn conflict đúng, nhãn DRAG dự đoán, câu trả lời baseline và câu trả lời DRAG.

## D020 - conflicting_opinions

**Câu hỏi:** Theo phân tích lỗi, vanilla model thường mắc lỗi gì với câu hỏi thuộc loại conflicting opinions?

**Expected:** Chúng thường chỉ trình bày một quan điểm, hoặc trình bày nhiều quan điểm nhưng thiên lệch mạnh về một phía, thay vì phản ánh bất đồng một cách trung lập.

**Expected behavior:** Nêu đúng lỗi hành vi ở nhóm conflicting opinions.

| Trường | Baseline | DRAG |
|---|---:|---:|
| Recall@5 | 1.000 | 1.000 |
| MRR@5 | 0.500 | 1.000 |
| RAGAS avg | 1.000 | 1.000 |
| Behavior alignment | 1.000 | 1.000 |

**DRAG phân loại:** `conflicting_opinions` (confidence=0.90)

**Lý do DRAG:** The retrieved passages indicate that models often struggle with conflicting opinions, either presenting a single viewpoint or showing a strong bias towards one perspective. This aligns with the user's question about the common errors of vanilla models in handling conflicting opinions.

**Policy DRAG:** The answer should acknowledge the presence of conflicting opinions and highlight the challenges faced by vanilla models in addressing them. It should also suggest that models may either present biased views or fail to capture the full range of perspectives.

**Baseline answer:**

```markdown
Theo phân tích lỗi, với câu hỏi thuộc loại **conflicting opinions**, vanilla model thường **chỉ đưa ra một quan điểm duy nhất hoặc nhiều quan điểm nhưng thiên lệch mạnh về một phía**, thay vì phản ánh đầy đủ các góc nhìn liên quan [2][3]. Trong bảng kết quả, đây cũng là loại có độ chính xác hành vi kỳ vọng thấp nhất dưới prompt Vanilla, chỉ đạt **36.2%** [3][4].
```

**DRAG answer:**

```markdown
Theo phân tích lỗi, với câu hỏi thuộc loại **conflicting opinions**, vanilla model thường **chỉ प्रस्तुत một quan điểm duy nhất** hoặc **nêu nhiều quan điểm nhưng bị lệch mạnh về một phía**, thay vì trung lập và bao quát đầy đủ các góc nhìn [2]. Đây cũng là loại khó nhất, với độ chính xác chỉ **36.2%** dưới vanilla prompt [1][3].
```

**Top chunks:**

- Baseline: `[38, 42, 43, 45, 22]`
- DRAG: `[43, 42, 45, 22, 9]`
