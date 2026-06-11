"""Kiểm thử nhanh ConflictRAG trên cặp tài liệu "khó".

Chạy:  python -m scripts.test_conflict_hard   (từ thư mục backend/)
"""
from app.config import get_settings
from app.conflict.detector import ConflictDetector

DOCS = [
    {
        "title": "Độ phức tạp thuật toán Dijkstra",
        "text": (
            "Thuật toán Dijkstra tìm đường đi ngắn nhất từ một đỉnh nguồn có độ phức tạp "
            "thời gian là O(V^2), trong đó V là số đỉnh của đồ thị. Đây là cận khi ta cài đặt "
            "bằng cách duyệt mảng tuyến tính để chọn ra đỉnh chưa thăm có khoảng cách tạm thời "
            "nhỏ nhất ở mỗi bước."
        ),
    },
    {
        "title": "Phân tích hiệu năng Dijkstra trên đồ thị thưa",
        "text": (
            "Độ phức tạp thời gian của thuật toán Dijkstra là O((V + E) log V), với V là số đỉnh "
            "và E là số cạnh. Nhờ sử dụng hàng đợi ưu tiên dạng binary heap để lấy đỉnh có khoảng "
            "cách nhỏ nhất, thuật toán chạy nhanh hơn đáng kể trên các đồ thị thưa."
        ),
    },
]

QUERIES = [
    "Độ phức tạp thời gian của thuật toán Dijkstra là bao nhiêu?",
    "Dijkstra dùng cấu trúc dữ liệu gì để chọn đỉnh?",  # nên ra no-conflict dễ hơn
]


def main() -> None:
    s = get_settings()
    detector = ConflictDetector(
        model_dir=s.conflict_model_dir,
        openai_api_key=s.openai_api_key,
        openai_base_url=s.openai_base_url,
        embedding_model=s.embed_model,
        embedding_dim=s.embed_dim or 1536,
        threshold=s.conflict_threshold,
        llm_model=s.llm_model_fast,
        tau_c=s.conflict_tau_c,
        enable_stage2=s.conflict_enable_stage2,
    )

    for q in QUERIES:
        pred = detector.predict_pair(q, DOCS[0]["text"], DOCS[1]["text"])
        print("=" * 70)
        print(f"Câu hỏi: {q}")
        print(f"  conflict             = {pred['conflict']}")
        print(f"  conflict_probability = {pred['conflict_probability']:.4f}  (Stage 1 MLP)")
        print(f"  stage1_confidence    = {pred['stage1_confidence']:.4f}  (tau_c={detector.tau_c})")
        print(f"  type_label           = {pred['type_label']}")
        print(f"  stage                = {pred['stage']}")
        if pred["stage"] == "stage2":
            print(f"  stage2_summary       = {pred.get('stage2_summary')}")
            print(f"  stage2_confidence    = {pred.get('stage2_confidence')}")


if __name__ == "__main__":
    main()
