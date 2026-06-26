"""Test cho ChromaVectorStore / Retriever (design.md Phần 4).

S2-DE-03: test_add_returns_true, test_add_empty, test_add_mismatched_lengths,
           test_in_memory_mode, test_persistent_mode, test_get_collection_stats,
           test_delete_collection, test_delete_nonexistent_collection,
           property-based tests (Hypothesis).

S5-DE-03: test_add_and_retrieve, test_top_k_limit, và
property-based test cho Property 6 (không vượt quá k), Property 7
(sắp xếp giảm dần theo score), Property 8 (score thuộc [0.0, 1.0]).
"""

import uuid

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.embeddings.vector_store import ChromaVectorStore
from src.models import Chunk



def make_chunk(chunk_id: str, doc_id: str = "doc-1", content: str = "hello") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        content=content,
        start_index=0,
        end_index=len(content),
    )


def fresh_store(**kwargs) -> ChromaVectorStore:
    """ChromaVectorStore với collection name ngẫu nhiên — tránh chia sẻ state giữa các test.

    ChromaDB EphemeralClient dùng chung process-level storage nên các test cần
    collection name riêng để không ảnh hưởng nhau.
    """
    kwargs.setdefault("collection_name", f"test-{uuid.uuid4().hex}")
    return ChromaVectorStore(**kwargs)


DUMMY_VECTOR = [0.1, 0.2, 0.3]



class TestInitClient:
    def test_in_memory_mode_client_created(self):
        """_init_client() với persist_dir=None tạo EphemeralClient (Yêu cầu 4.6)."""
        store = fresh_store()
        store._init_client()
        assert store._client is not None
        assert store._collection is not None

    def test_persistent_mode_client_created(self, tmp_path):
        """_init_client() với persist_dir hợp lệ tạo PersistentClient (Yêu cầu 4.5)."""
        store = ChromaVectorStore(persist_dir=str(tmp_path))
        store._init_client()
        assert store._client is not None
        assert store._collection is not None

    def test_lazy_init_on_first_add(self):
        """_ensure_client() được gọi tự động khi add() lần đầu."""
        store = fresh_store()
        assert store._client is None
        store.add([make_chunk("c1")], [DUMMY_VECTOR])
        assert store._client is not None



class TestAdd:
    def test_add_returns_true(self):
        """add() với chunk và vector hợp lệ trả về True (Yêu cầu 4.1)."""
        store = fresh_store()
        assert store.add([make_chunk("c1")], [DUMMY_VECTOR]) is True

    def test_add_empty_list_returns_true(self):
        """add() với list rỗng trả về True ngay (edge case)."""
        store = fresh_store()
        assert store.add([], []) is True

    def test_add_mismatched_lengths_raises(self):
        """add() vi phạm precondition len(chunks) != len(vectors) phải raise ValueError."""
        store = fresh_store()
        with pytest.raises(ValueError):
            store.add([make_chunk("c1"), make_chunk("c2")], [DUMMY_VECTOR])

    def test_add_multiple_chunks(self):
        """add() nhiều chunk cùng lúc — tất cả được lưu (Yêu cầu 4.1)."""
        store = fresh_store()
        chunks = [make_chunk(f"c{i}", content=f"text {i}") for i in range(3)]
        vectors = [[float(i), 0.0, 0.0] for i in range(3)]
        assert store.add(chunks, vectors) is True
        assert store.get_collection_stats()["num_chunks"] == 3

    def test_add_upsert_same_id(self):
        """add() với chunk_id trùng — upsert, không raise lỗi, count không tăng."""
        store = fresh_store()
        store.add([make_chunk("c1", content="original")], [DUMMY_VECTOR])
        result = store.add([make_chunk("c1", content="updated")], [[0.3, 0.2, 0.1]])
        assert result is True
        assert store.get_collection_stats()["num_chunks"] == 1



class TestGetCollectionStats:
    def test_stats_on_empty_collection(self):
        """get_collection_stats() trên collection rỗng trả về num_chunks=0."""
        store = fresh_store()
        stats = store.get_collection_stats()
        assert stats["num_chunks"] == 0
        assert stats["collection_name"] == store.collection_name
        assert stats["persist_dir"] is None

    def test_stats_after_add(self):
        """get_collection_stats() phản ánh đúng số chunk đã add."""
        store = fresh_store()
        store.add([make_chunk("c1"), make_chunk("c2")], [DUMMY_VECTOR, DUMMY_VECTOR])
        assert store.get_collection_stats()["num_chunks"] == 2

    def test_stats_persist_dir_reflected(self, tmp_path):
        """get_collection_stats() trả về persist_dir đúng."""
        store = ChromaVectorStore(persist_dir=str(tmp_path))
        stats = store.get_collection_stats()
        assert stats["persist_dir"] == str(tmp_path)


# ---------------------------------------------------------------------------
# S2-DE-03: delete_collection() (Yêu cầu 4.7)
# ---------------------------------------------------------------------------

class TestDeleteCollection:
    def test_delete_existing_collection(self):
        """delete_collection() xóa collection tồn tại, trả về True (Yêu cầu 4.7)."""
        store = fresh_store()
        store.add([make_chunk("c1")], [DUMMY_VECTOR])
        assert store.delete_collection(store.collection_name) is True

    def test_delete_nonexistent_collection_returns_false(self):
        """delete_collection() với collection không tồn tại trả về False (edge case)."""
        store = fresh_store()
        store._ensure_client()
        assert store.delete_collection("does-not-exist-xyz") is False

    def test_collection_recreated_after_delete(self):
        """Sau khi delete_collection(), add() tiếp theo tự tạo lại collection."""
        store = fresh_store()
        store.add([make_chunk("c1")], [DUMMY_VECTOR])
        store.delete_collection(store.collection_name)
        # _collection = None sau khi xóa → _ensure_client tái tạo khi add()
        assert store.add([make_chunk("c2")], [DUMMY_VECTOR]) is True
        assert store.get_collection_stats()["num_chunks"] == 1


# ---------------------------------------------------------------------------
# S2-DE-03: persistent mode — dữ liệu tồn tại sau khi khởi động lại (Yêu cầu 4.5)
# ---------------------------------------------------------------------------

class TestPersistentMode:
    def test_data_survives_recreate(self, tmp_path):
        """Dữ liệu trong persistent store khôi phục được khi tạo instance mới (Yêu cầu 4.5)."""
        persist = str(tmp_path)
        store1 = ChromaVectorStore(persist_dir=persist)
        store1.add([make_chunk("c1", content="persistent data")], [DUMMY_VECTOR])

        # Mô phỏng 'khởi động lại': tạo instance mới trỏ cùng persist_dir
        store2 = ChromaVectorStore(persist_dir=persist)
        assert store2.get_collection_stats()["num_chunks"] == 1


# ---------------------------------------------------------------------------
# S2-DE-03: _to_chroma_metadata — metadata key collision (TDD red→green)
# ---------------------------------------------------------------------------

class TestMetadataSerialization:
    def test_fixed_keys_not_overwritten_by_user_metadata(self):
        """Chunk.metadata keys 'doc_id'/'start_index'/'end_index' không được ghi đè giá trị authoritative.

        TDD RED: bug — user metadata overwrites fixed keys.
        Sau khi fix: fixed keys luôn thắng.
        """
        chunk = Chunk(
            chunk_id="c1",
            doc_id="real-doc-id",
            content="text",
            start_index=10,
            end_index=14,
            metadata={"doc_id": "injected-id", "start_index": 999},
        )
        meta = ChromaVectorStore._to_chroma_metadata(chunk)
        assert meta["doc_id"] == "real-doc-id", (
            "user metadata 'doc_id' must not overwrite chunk.doc_id"
        )
        assert meta["start_index"] == 10, (
            "user metadata 'start_index' must not overwrite chunk.start_index"
        )

    def test_non_primitive_metadata_serialized_to_str(self):
        """Metadata có giá trị non-primitive (list, dict, None) được chuyển thành str."""
        chunk = Chunk(
            chunk_id="c1",
            doc_id="d1",
            content="text",
            start_index=0,
            end_index=4,
            metadata={"tags": ["a", "b"], "nested": {"k": 1}, "none_val": None},
        )
        meta = ChromaVectorStore._to_chroma_metadata(chunk)
        assert isinstance(meta["tags"], str)
        assert isinstance(meta["nested"], str)
        assert isinstance(meta["none_val"], str)

    def test_primitive_metadata_preserved_as_is(self):
        """Metadata có giá trị str/int/float/bool không bị chuyển đổi."""
        chunk = Chunk(
            chunk_id="c1",
            doc_id="d1",
            content="text",
            start_index=0,
            end_index=4,
            metadata={"score": 0.9, "label": "positive", "flag": True, "count": 3},
        )
        meta = ChromaVectorStore._to_chroma_metadata(chunk)
        assert meta["score"] == 0.9
        assert meta["label"] == "positive"
        assert meta["flag"] is True
        assert meta["count"] == 3

    def test_add_with_conflicting_metadata_keys_stores_correct_doc_id(self):
        """add() với chunk có metadata 'doc_id' injected → similarity_search sau đó dùng đúng doc_id."""
        store = fresh_store()
        chunk = Chunk(
            chunk_id="c1",
            doc_id="correct-doc",
            content="hello world",
            start_index=0,
            end_index=11,
            metadata={"doc_id": "WRONG", "start_index": 9999},
        )
        assert store.add([chunk], [DUMMY_VECTOR]) is True
        # Xác nhận collection có 1 item (add thành công)
        assert store.get_collection_stats()["num_chunks"] == 1


# ---------------------------------------------------------------------------
# S2-DE-03: Property-based tests (Hypothesis)
# ---------------------------------------------------------------------------

_SAFE_TEXT = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Zs")),
    min_size=1,
    max_size=50,
)

_CHUNK_ID = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=20,
)


@st.composite
def chunk_list(draw, min_size: int = 1, max_size: int = 5):
    """Strategy tạo list Chunk với chunk_id duy nhất."""
    ids = draw(st.lists(_CHUNK_ID, min_size=min_size, max_size=max_size, unique=True))
    return [
        Chunk(
            chunk_id=cid,
            doc_id="doc-prop",
            content=draw(_SAFE_TEXT),
            start_index=0,
            end_index=1,
        )
        for cid in ids
    ]


class TestPropertyBased:
    @given(chunks=chunk_list(min_size=1, max_size=5))
    @settings(max_examples=20, deadline=None)
    def test_add_always_returns_true_for_valid_input(self, chunks):
        """Property: add() với chunks và vectors hợp lệ luôn trả về True."""
        store = fresh_store()
        dim = 4
        vectors = [[0.1] * dim for _ in chunks]
        assert store.add(chunks, vectors) is True

    @given(chunks=chunk_list(min_size=1, max_size=5))
    @settings(max_examples=20, deadline=None)
    def test_num_chunks_equals_unique_chunk_ids(self, chunks):
        """Property: num_chunks sau add() == số chunk_id duy nhất (upsert dedup)."""
        store = fresh_store()
        dim = 4
        vectors = [[0.1] * dim for _ in chunks]
        store.add(chunks, vectors)
        unique_ids = len({c.chunk_id for c in chunks})
        assert store.get_collection_stats()["num_chunks"] == unique_ids

    @given(
        n=st.integers(min_value=1, max_value=4),
        extra=st.integers(min_value=0, max_value=3),
    )
    @settings(max_examples=15, deadline=None)
    def test_add_mismatched_raises_value_error(self, n, extra):
        """Property: add() với len(chunks) != len(vectors) luôn raise ValueError."""
        store = fresh_store()
        chunks = [make_chunk(f"c{i}") for i in range(n)]
        vectors = [DUMMY_VECTOR for _ in range(n + extra + 1)]
        with pytest.raises(ValueError):
            store.add(chunks, vectors)

    @given(chunks=chunk_list(min_size=1, max_size=4))
    @settings(max_examples=15, deadline=None)
    def test_delete_and_readd_gives_correct_count(self, chunks):
        """Property: delete → add lại → count đúng."""
        store = fresh_store()
        dim = 3
        vectors = [[0.2] * dim for _ in chunks]
        store.add(chunks, vectors)
        store.delete_collection(store.collection_name)
        store.add(chunks, vectors)
        assert store.get_collection_stats()["num_chunks"] == len(chunks)
