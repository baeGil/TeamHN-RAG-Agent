# Hướng dẫn tạo câu hỏi cho dataset RAG

Tài liệu này quy ước cách tạo câu hỏi cho bộ 67 file PDF tiếng Việt trong `data/67_files`.
Mục tiêu là tạo dataset có thể dùng để đánh giá RAG: truy hồi đúng, trả lời bám tài liệu, tổng hợp được nhiều phần, và biết từ chối khi tài liệu không có thông tin.

## 0. Chủ đề thu thập và quota file

Nhóm dùng 8 chủ đề chính để bảo đảm bộ 200 PDF đủ đa dạng nhưng vẫn dễ quản lý.

| Chủ đề | Số file cho toàn nhóm | Số file trong phần 67 file |
|---|---:|---:|
| Pháp luật - chính sách | 30 | 10 |
| Giáo dục - đào tạo | 25 | 8 |
| Tài chính - kinh tế - doanh nghiệp | 25 | 8 |
| Y tế - sức khỏe cộng đồng | 25 | 8 |
| Khoa học - công nghệ - AI | 25 | 8 |
| Lịch sử - văn hóa - xã hội | 20 | 7 |
| Môi trường - năng lượng - giao thông | 25 | 9 |
| Sản phẩm - dịch vụ - bảng giá | 25 | 9 |
| **Tổng** | **200** | **67** |

Ghi chủ đề trong metadata bằng slug không dấu để dễ lọc:

| Chủ đề | Giá trị đề xuất cho cột `chu_de` |
|---|---|
| Pháp luật - chính sách | `phap_luat_chinh_sach` |
| Giáo dục - đào tạo | `giao_duc_dao_tao` |
| Tài chính - kinh tế - doanh nghiệp | `tai_chinh_kinh_te_doanh_nghiep` |
| Y tế - sức khỏe cộng đồng | `y_te_suc_khoe_cong_dong` |
| Khoa học - công nghệ - AI | `khoa_hoc_cong_nghe_ai` |
| Lịch sử - văn hóa - xã hội | `lich_su_van_hoa_xa_hoi` |
| Môi trường - năng lượng - giao thông | `moi_truong_nang_luong_giao_thong` |
| Sản phẩm - dịch vụ - bảng giá | `san_pham_dich_vu_bang_gia` |

Ưu tiên tài liệu có nguồn rõ ràng, PDF tiếng Việt, mở được, không trùng nội dung, và có đủ thông tin để tạo câu hỏi ở nhiều mức độ. Với chủ đề sản phẩm - dịch vụ - bảng giá, nên ưu tiên brochure, bảng giá, catalogue, chính sách bán hàng, FAQ sản phẩm và tài liệu so sánh phiên bản.

Không cần ép mỗi lĩnh vực hoặc mỗi PDF phải phủ đủ mọi loại câu hỏi. Nguyên tắc là:

```text
Toàn bộ 67 file phải phủ đa dạng question_types.
Từng lĩnh vực chỉ cần phủ những loại câu hỏi tự nhiên với lĩnh vực đó.
```

Gợi ý coverage tự nhiên theo lĩnh vực:

| Lĩnh vực | Question types dễ sinh tự nhiên |
|---|---|
| Pháp luật - chính sách | `definition`, `condition`, `comparison`, `multi_hop`, `unanswerable` |
| Giáo dục - đào tạo | `factoid`, `definition`, `list`, `condition`, `single_hop` |
| Tài chính - kinh tế - doanh nghiệp | `factoid`, `table_lookup`, `calculation`, `comparison`, `multi_hop` |
| Y tế - sức khỏe cộng đồng | `definition`, `condition`, `list`, `multi_hop`, `clarification_needed` |
| Khoa học - công nghệ - AI | `definition`, `abstract`, `multi_hop`, `comparison`, `calculation` |
| Lịch sử - văn hóa - xã hội | `factoid`, `list`, `abstract`, `multi_hop`, `definition` |
| Môi trường - năng lượng - giao thông | `factoid`, `table_lookup`, `comparison`, `calculation`, `multi_hop` |
| Sản phẩm - dịch vụ - bảng giá | `factoid`, `table_lookup`, `comparison`, `clarification_needed`, `single_hop` |

Khi review toàn bộ 67 file, nên kiểm tra coverage theo `question_types` để tránh lệch quá nhiều về câu dễ/single-hop. Bảng mục tiêu tham khảo:

| Question type | Mức xuất hiện mục tiêu | Lĩnh vực nên ưu tiên |
|---|---|---|
| `factoid` | nhiều | tất cả lĩnh vực |
| `definition` | vừa | pháp luật, giáo dục, y tế, khoa học |
| `list` | vừa | giáo dục, y tế, sản phẩm, lịch sử |
| `single_hop` | nhiều | tất cả lĩnh vực |
| `multi_hop` | vừa | pháp luật, tài chính, khoa học, môi trường |
| `specific` | nhiều | tất cả lĩnh vực |
| `abstract` | vừa | khoa học, lịch sử, chính sách |
| `table_lookup` | vừa | tài chính, sản phẩm, môi trường |
| `comparison` | vừa | sản phẩm, pháp luật, tài chính |
| `calculation` | ít-vừa | tài chính, môi trường, kỹ thuật |
| `clarification_needed` | ít | sản phẩm, y tế, chính sách |
| `unanswerable` | ít | tất cả lĩnh vực |

## 1. Schema một dòng dataset

Mỗi dòng câu hỏi nên có các cột sau:

| Cột | Ý nghĩa |
|---|---|
| `doc_id` | Mã tài liệu, ví dụ `DOC001`. |
| `question_id` | Mã câu hỏi, ví dụ `DOC001_Q01`. |
| `question` | Câu hỏi người dùng bằng tiếng Việt tự nhiên. |
| `expected_answer` | Câu trả lời kỳ vọng, chỉ dựa trên tài liệu. |
| `evidence` | Trang, mục, bảng, hoặc trích đoạn ngắn làm căn cứ. |
| `difficulty` | Mức độ khó: `easy`, `medium`, `hard`. |
| `difficulty_explanation` | Giải thích mức độ dựa trên retrieval complexity, reasoning depth, semantic distance và document complexity. |
| `question_types` | Một hoặc nhiều nhãn, mỗi nhãn kèm giải thích ngắn trong ngoặc. |
| `human_steps` | Các bước con người sẽ làm khi tra cứu tài liệu để trả lời. |
| `answerability` | `answerable`, `unanswerable`, hoặc `clarification_needed`. |
| `notes` | Tạm thời có thể bỏ trống; chỉ dùng khi cần ghi chú QC. |

Mỗi PDF nên có 5 câu:

| Mức độ | Số câu |
|---|---:|
| `easy` | 2 |
| `medium` | 2 |
| `hard` | 1 |

Câu `unanswerable` không bắt buộc trong mỗi PDF ở vòng hiện tại. Nếu nhóm muốn kiểm tra chống bịa đặt ở vòng sau, có thể thêm riêng và ghi `answerability = unanswerable`; không mặc định xem câu unanswerable là hard.

## 2. Bộ nhãn `question_types`

Mỗi câu có thể gắn nhiều nhãn. Trong cột `question_types`, ghi theo dạng:

```text
factoid (hỏi một giá trị cụ thể); single_hop (chỉ cần trang giá); specific (câu hỏi hẹp)
```

| Nhãn | Ý nghĩa | Cơ sở tham khảo |
|---|---|---|
| `factoid` | Hỏi một thông tin cụ thể như giá, số, ngày, tên, địa điểm, thông số. | TREC QA dùng factoid questions để đánh giá câu trả lời ngắn, chính xác. |
| `definition` | Hỏi định nghĩa hoặc khái niệm được tài liệu giải thích. | TREC QA có nhóm definition questions. |
| `list` | Hỏi một danh sách nhiều mục cùng loại. | TREC QA có nhóm list questions. |
| `single_hop` | Chỉ cần một đoạn, một trang, một bảng, hoặc một mục chính để trả lời. | Ragas dùng single-hop query trong tạo testset RAG. |
| `multi_hop` | Cần kết hợp nhiều đoạn, trang, bảng, hoặc mục để trả lời. | Ragas và HotpotQA dùng multi-hop để kiểm tra truy hồi và suy luận nhiều bước. |
| `specific` | Câu hỏi hẹp, nhắm vào một thông tin rõ ràng. | Ragas phân biệt specific và abstract queries. |
| `abstract` | Câu hỏi yêu cầu diễn giải, tổng hợp, nhận xét, hoặc rút ra ý chính từ tài liệu. | Ragas phân biệt abstract queries để kiểm tra khả năng tổng hợp. |
| `condition` | Cần hiểu điều kiện áp dụng, ngoại lệ, phạm vi, hoặc ràng buộc được nêu trong tài liệu. | Thường gặp trong văn bản pháp luật, chính sách, y tế, giáo dục và tài liệu hướng dẫn. |
| `table_lookup` | Đáp án nằm trong bảng, cần đọc dòng/cột/ô. | Các benchmark table QA như HybridQA kiểm tra khả năng trả lời từ bảng và văn bản. |
| `comparison` | Cần đối chiếu hai hoặc nhiều đối tượng, chính sách, phiên bản, hoặc mốc số liệu. | HotpotQA có dạng comparison questions; cũng hữu ích cho tài liệu sản phẩm/bảng giá. |
| `calculation` | Cần tính toán từ số liệu trong tài liệu. | Các bài toán QA trên bảng/số liệu thường yêu cầu thao tác tính toán trước khi trả lời. |
| `clarification_needed` | Câu hỏi thiếu thông tin nên cần hỏi lại hoặc trả lời theo nhiều khả năng. | Dùng cho tình huống người dùng thật hỏi mơ hồ, ví dụ không nêu mẫu xe/thời điểm/loại giá. |
| `unanswerable` | Tài liệu không chứa đủ thông tin để trả lời. | SQuAD 2.0 và Natural Questions dùng câu không có đáp án/null để kiểm tra khả năng không bịa. |

## 3. Vì sao cần phân loại

Phân loại giúp nhóm biết mỗi câu đang kiểm tra năng lực nào của RAG:

| Nhóm nhãn | Năng lực được kiểm tra |
|---|---|
| `factoid`, `definition`, `list` | Kiểu đáp án: giá trị cụ thể, định nghĩa, hay danh sách. |
| `single_hop`, `multi_hop` | Độ phức tạp truy hồi: tìm một chỗ hay phải kết hợp nhiều chỗ. |
| `specific`, `abstract` | Mức độ cụ thể của nhu cầu người dùng: hỏi chi tiết hẹp hay hỏi tổng hợp/diễn giải. |
| `unanswerable` | Khả năng chống bịa đặt khi tài liệu không có thông tin. |

## 4. Quy tắc chia độ khó

Độ khó không nên dựa vào cảm giác câu dài hay ngắn. Hãy dựa vào số evidence và thao tác cần thực hiện.

Difficulty là nhãn tổng hợp nội bộ dựa trên 4 khía cạnh:

| Khía cạnh | Câu hỏi kiểm tra |
|---|---|
| `retrieval_complexity` | Evidence nằm một chỗ hay nhiều chỗ? Dễ tìm hay rải rác? |
| `reasoning_depth` | Cần bao nhiêu bước suy luận/kết hợp sau khi tìm được evidence? |
| `semantic_distance` | Câu hỏi dùng từ gần với tài liệu hay phải diễn giải từ cách hỏi khác? |
| `document_complexity` | Evidence nằm trong text thường, bảng, ghi chú nhỏ, layout phức tạp, scan/OCR? |

| Mức độ | Tiêu chí |
|---|---|
| `easy` | Tất cả hoặc gần như tất cả khía cạnh đều thấp: evidence nằm một chỗ, đáp án trực tiếp, từ khóa gần với tài liệu, text dễ đọc. |
| `medium` | Có ít nhất một khía cạnh ở mức trung bình: cần đọc ghi chú/điều kiện, đối chiếu nhẹ, đọc bảng đơn giản, hoặc câu hỏi diễn đạt khác tài liệu một chút. |
| `hard` | Có một hoặc nhiều khía cạnh cao: evidence rải nhiều trang/mục, cần multi-hop, so sánh/tính toán/tổng hợp, câu hỏi trừu tượng hơn tài liệu, hoặc evidence nằm trong bảng/layout phức tạp. |
Lưu ý: `unanswerable` không phải là giá trị của `difficulty`. Nếu tài liệu không đủ thông tin để trả lời, ghi `answerability = unanswerable`, còn `difficulty` vẫn chọn `easy`, `medium` hoặc `hard` theo mức độ rà soát cần thiết.

Trong cột `difficulty_explanation`, ghi ngắn gọn theo dạng:

```text
Retrieval: thấp, ...
Reasoning: thấp, ...
Semantic: thấp, ...
Document: thấp, ...
```

Nếu ghi trong bảng Markdown, dùng `<br>` để xuống dòng trong cùng một ô.

Ví dụ:

| Câu hỏi | difficulty | difficulty_explanation | question_types |
|---|---|---|---|
| Giá xe VinFast VF 3 kèm pin là bao nhiêu? | `easy` | Retrieval: thấp, chỉ cần trang giá.<br>Reasoning: thấp, đọc trực tiếp.<br>Semantic: thấp, từ hỏi gần tài liệu.<br>Document: thấp, thông tin giá rõ. | `factoid (hỏi một giá trị cụ thể); single_hop (chỉ cần trang giá); specific (câu hỏi hẹp)` |
| Quãng đường 215 km của VF 3 có phải luôn đạt được trong thực tế không? | `medium` | Retrieval: thấp, cùng một trang.<br>Reasoning: trung bình, phải hiểu ghi chú NEDC.<br>Semantic: thấp.<br>Document: trung bình, thông tin nằm trong ghi chú. | `single_hop (chỉ cần một trang); specific (hỏi một thông số và điều kiện)` |
| Nếu xét cả giá mua, khả năng di chuyển và hậu mãi, brochure cung cấp những điểm chính nào về VF 3? | `hard` | Retrieval: cao, cần nhiều trang.<br>Reasoning: cao, ghép nhiều khía cạnh.<br>Semantic: trung bình.<br>Document: trung bình, có thông số và ghi chú. | `multi_hop (cần kết hợp nhiều trang); specific (hỏi các khía cạnh đã nêu rõ)` |
| Brochure có nêu giá VF 3 sau ưu đãi theo từng tỉnh thành không? | `medium` | Retrieval: trung bình, cần rà phần giá/chính sách.<br>Reasoning: thấp, chỉ xác nhận thiếu thông tin.<br>Semantic: thấp, câu hỏi hẹp.<br>Document: trung bình, cần kiểm tra phần giá và ghi chú. | `unanswerable (tài liệu không có đáp án); specific (hỏi một thông tin hẹp)` |

## 5. Cơ sở tham khảo

- Ragas - Testset Generation for RAG: https://docs.ragas.io/en/v0.2.12/concepts/test_data_generation/rag/
- HotpotQA - A Dataset for Diverse, Explainable Multi-hop Question Answering: https://arxiv.org/abs/1809.09600
- TREC 2003 Question Answering Track Overview: https://trec.nist.gov/pubs/trec12/papers/QA.OVERVIEW.pdf
- SQuAD 2.0 - Know What You Don't Know: https://arxiv.org/abs/1806.03822
- Natural Questions - Google Research: https://research.google/pubs/natural-questions-a-benchmark-for-question-answering-research/
- HybridQA - Multi-Hop QA over Tabular and Textual Data: https://arxiv.org/abs/2004.07347
