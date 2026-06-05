from types import SimpleNamespace

from app.indexing.store import KnowledgeBase, _ParentNode, _ScoredHit
from app.retrieval.hybrid import FusedHit


class _Repo:
    def get_chunks(self, chunk_ids):
        return {
            cid: {
                "id": cid,
                "document_id": 1,
                "chunk_index": cid - 10,
                "text": f"chunk {cid}",
                "page": 1,
                "section": "A",
                "doc_title": "Doc",
                "doc_source": "test",
                "doc_source_type": "text",
            }
            for cid in chunk_ids
        }


def test_hierarchical_boost_adds_and_marks_child_chunks():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.settings = SimpleNamespace(
        hierarchical_parent_chunk_window=3,
        hierarchical_parent_boost=0.5,
        turbovec_bit_width=4,
    )
    kb.repo = _Repo()
    kb.parent_nodes = {
        1: _ParentNode(
            parent_id=1,
            document_id=1,
            doc_title="Doc",
            section="A",
            page_start=1,
            page_end=1,
            chunk_ids=[10, 11, 12],
            text="Doc | A\nchunk 10\nchunk 11\nchunk 12",
        )
    }
    kb.chunk_to_parent = {10: 1, 11: 1, 12: 1}
    meta = {10: {"id": 10}}
    ordered = [
        _ScoredHit(
            chunk_id=10,
            rrf_score=0.1,
            bm25_score=1.0,
            dense_score=0.9,
            rerank_score=None,
            score=0.2,
        )
    ]
    parent_hits = {1: FusedHit(chunk_id=1, rrf_score=0.6)}

    boosted = kb._apply_hierarchical_boost(ordered, parent_hits, meta)

    assert [h.chunk_id for h in boosted] == [10, 11, 12]
    assert boosted[0].hierarchical_boosted is True
    assert boosted[0].hierarchical_parent_id == 1
    assert meta[11]["text"] == "chunk 11"
    assert meta[12]["text"] == "chunk 12"
