# So sánh câu trả lời Baseline vs DRAG

- Dataset: `D:\TeamHN-RAG-Agent\data\data_conflict\conflict_test_qa.csv`
- Số câu hỏi: **2**

## DC014 - Simple w. condition - temporal_scope

**Câu hỏi:** Nếu câu hỏi hỏi chính sách giảm GTGT áp dụng trong tháng 8/2024, phải dùng nghị định nào và hiệu lực ra sao?

**Expected:** Phải dùng NĐ 72/2024/NĐ-CP vì nghị định này có hiệu lực từ 01/07/2024 đến hết 31/12/2024.

**Evidence:** pdf/ND_72_2024_GTGT_excerpt_8p.pdf:p3

**Baseline:** RAGAS=1.000, Behavior=1.000, Latency=9.7s

Nếu hỏi chính sách giảm GTGT áp dụng trong **tháng 8/2024**, thì phải dùng **Nghị định số 72/2024** vì nghị định này có hiệu lực từ **ngày 01/7/2024 đến hết ngày 31/12/2024**, nên tháng 8/2024 nằm trong thời gian hiệu lực của nghị định này. [1]

Nghị định số 174/2025 có hiệu lực từ **ngày 01/7/2025 đến hết ngày 31/12/2026**, nên không áp dụng cho tháng 8/2024. [2]

**DRAG:** RAGAS=1.000, Behavior=1.000, Latency=5.6s, Predicted=temporal_scope

Nếu hỏi chính sách giảm GTGT áp dụng trong **tháng 8/2024**, thì phải dùng **Nghị định 72/2024/NĐ-CP** vì nghị định này có hiệu lực từ **01/7/2024 đến hết 31/12/2024** [1].

Các nghị định khác trong tài liệu không áp dụng cho tháng 8/2024:
- **Nghị định 180/2024/NĐ-CP**: hiệu lực từ **01/01/2025 đến 30/6/2025** [3].
- **Nghị định 174/2025/NĐ-CP**: hiệu lực từ **01/7/2025 đến 31/12/2026** [2].

**Judge baseline:** The answer correctly identifies NĐ 72/2024 as the applicable decree for the VAT reduction policy in August 2024, supported by the context that states its effective period. It also appropriately excludes NĐ 174/2025 based on the temporal scope, aligning with the expected conflict behavior.

**Judge DRAG:** The answer correctly identifies Nghị định 72/2024/NĐ-CP as the applicable decree for the VAT reduction policy in August 2024, supported by the context that states its effective period. It also appropriately excludes Nghị định 180/2024 and Nghị định 174/2025 based on their effective dates, aligning with the expected conflict behavior.

## DC089 - False premise - false_premise

**Câu hỏi:** NĐ 72/2024 quy định chính sách giảm GTGT theo Nghị quyết 174/2024/QH15 phải không?

**Expected:** Không. NĐ 72/2024 quy định theo Nghị quyết 142/2024/QH15 ngày 29/06/2024. Nghị quyết 174/2024/QH15 là căn cứ của NĐ 180/2024.

**Evidence:** pdf/ND_72_2024_GTGT_excerpt_8p.pdf:p1; pdf/ND_180_2024_GTGT_excerpt_8p.pdf:p1

**Baseline:** RAGAS=1.000, Behavior=1.000, Latency=3.5s

Có. Trong ngữ cảnh, Nghị định 180/2024/GTGT nêu rõ việc giảm thuế giá trị gia tăng được thực hiện “theo Nghị quyết số 174/2024/QH15” khi ghi chú trên hóa đơn đối với cơ sở kinh doanh thuộc đối tượng giảm thuế GTGT. [1]

Tuy nhiên, ngữ cảnh cũng cho thấy Nghị định 72/2024/GTGT là văn bản có hiệu lực từ ngày 01/7/2024 đến hết ngày 31/12/2024 và ghi chú giảm thuế theo “Nghị quyết số 142/2024/QH15”, không phải 174/2024/QH15. [2][3]

Vì vậy, theo ngữ cảnh cung cấp, **NĐ 72/2024 không phải là nghị định quy định chính sách giảm GTGT theo Nghị quyết 174/2024/QH15**; nội dung gắn với Nghị quyết 174/2024/QH15 xuất hiện ở Nghị định 180/2024/GTGT. [1][2][3]

**DRAG:** RAGAS=0.840, Behavior=0.800, Latency=6.2s, Predicted=freshness

Không hẳn. NĐ 72/2024 là quy định giảm GTGT cho giai đoạn **01/7/2024 đến 31/12/2024** và trong nội dung của nó còn dẫn chiếu **Nghị quyết số 142/2024/QH15** [2][3].  

Còn theo văn bản mới hơn, chính sách giảm GTGT được áp dụng từ **01/01/2025 đến 30/6/2025** và ghi chú hóa đơn theo **Nghị quyết số 174/2024/QH15** [1].  

Vì vậy, nếu hỏi theo chính sách hiện hành trong giai đoạn mới thì **không phải NĐ 72/2024**, mà là quy định tại **NĐ 180/2024** [1].

**Judge baseline:** The answer correctly identifies that NĐ 72/2024 does not regulate the reduction of GTGT according to Nghị quyết 174/2024/QH15, and it provides the correct context and reasoning to support this claim. It also aligns with the expected conflict behavior by correcting the false premise regarding the resolution numbers.

**Judge DRAG:** The answer correctly identifies that NĐ 72/2024 is related to Nghị quyết 142/2024/QH15 and not Nghị quyết 174/2024/QH15, but it does not explicitly correct the false premise about the resolution numbers as expected. It provides relevant context and details about the effective dates and policies, but the answer could be clearer in directly addressing the false premise. Overall, it aligns well with the expected conflict behavior.
