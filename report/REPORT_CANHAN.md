# Bao Cao Ca Nhan -- Lab 7: Embedding & Vector Store

**Ho ten:** Nguyen Minh Tuan
**Nhom:** 2A202602420
**Ngay:** 2026-09-20

> Nop 1 ban / sinh vien. Phan nhom nop chung 1 ban trong REPORT_NHOM.md.

**Tong diem phan ca nhan: 60** = Khoi dong (5) + Huong tiep can (10) + Hoan thien code (30) + Du doan tuong tu (5) + Ket qua truy xuat (10).

---

## 1. Khoi dong (Warm-up) -- Ca nhan (5 diem)

### Do tuong tu Cosine (Bai tap 1.1)

**Do tuong tu cosine cao nghia la gi?**
> Do tuong tu cosine do goc giua hai vector embedding. Khi hai van ban co cosine similarity cao (gan 1.0), chung bieu dat cung chu de hoac y nghia tuong dong, du dung tu ngu khac nhau -- chung "huong ve cung mot phia" trong khong gian ngu nghia.

**Vi du CO DO TUONG TU CAO:**
- Cau A: "Nguoi mua co 15 ngay de yeu cau tra hang hoan tien tren Shopee."
- Cau B: "Thoi han gui yeu cau doi tra va hoan tien cho Nguoi mua la 15 ngay ke tu khi nhan hang."
- Tai sao: Ca hai deu noi cung quy dinh (thoi han 15 ngay).

**Vi du CO DO TUONG TU THAP:**
- Cau A: "Thoi tiet hom nay rat dep, nang am va mat me."
- Cau B: "Chinh sach bao hanh thiet bi dien tu tren Shopee co hieu luc tu ngay giao hang."
- Tai sao: Hai linh vuc hoan toan khac nhau (thoi tiet vs. chinh sach TMDT).

**Tai sao cosine similarity duoc uu tien hon Euclidean distance?**
> Khoang cach Euclid bi anh huong boi do dai vector -- van ban dai se co vector lon hon, du y nghia giong nhau. Cosine similarity chi quan tam den **goc** (huong), khong phu thuoc vao do dai van ban.

---

### Bai toan Chunking (Bai tap 1.2)

**10,000 ky tu, chunk_size=500, overlap=50 -> bao nhieu chunks?**

```
step = 500 - 50 = 450
so_chunk = ceil((10000 - 50) / 450) = ceil(22.11) = 23 chunks
```
**Dap an: 23 chunks**

**Overlap tang len 100 thi so chunks thay doi the nao?**

```
step = 500 - 100 = 400
so_chunk = ceil((10000 - 100) / 400) = ceil(24.75) = 25 chunks
```
So chunks tang tu 23 len 25. Ta muon tang overlap de dam bao thong tin o **ranh gioi giua cac chunk** khong bi bo sot -- thong tin quan trong xuat hien o ca chunk truoc lan chunk sau.

---

## 2. Huong tiep can cua toi (My Approach) -- Ca nhan (10 diem)

### Chunking Functions

**SentenceChunker.chunk:**
> Dung regex `re.split(r"(?<=[.!?])\s+|(?<=\.)\n+", text)` de nhan biet ranh gioi cau. Sau do gop toi da `max_sentences_per_chunk` cau moi chunk va noi bang dau cach. Van ban rong tra ve `[]`. Chuoi sau split chi toan khoang trang duoc loc qua `.strip()`.

**RecursiveChunker._split:**
> Thuat toan de quy thu lan luot tung dau phan cach `["\n\n", "\n", ". ", " ", ""]`. Base case: do dai <= chunk_size -> tra ve nguyen doan. Dau phan cach khong chia duoc -> chuyen xuong cap ke. Khi chia duoc: dung buffer gop cac phan nho truoc khi cat, tranh tao qua nhieu chunk qua nho.

**HeadingChunker.chunk (custom -- R3 strategy):**
> Dung regex `^(#{1,N})\s+.+` de nhan dien dong tieu de Markdown. Moi khi gap tieu de moi, flush section hien tai vao chunks. Tieu de duoc giu nguyen lam dong dau cua chunk de chunk tu mo ta. Neu khong co tieu de, tra ve ca van ban la 1 chunk.

### EmbeddingStore

**add_documents + search:**
> Moi Document duoc chuyen thanh dict record (id, content, metadata, embedding). Luu vao danh sach `self._store` (in-memory). Khi search: tao embedding cho query roi tinh **dot product** voi tung record -- ty le thuan voi cosine similarity vi mock embedder tra vector chuan hoa. Sap xep giam dan theo diem, lay top-k.

**search_with_filter + delete_document:**
> `search_with_filter` loc truoc (pre-filter): giu lai nhung record co metadata khop hoan toan voi `metadata_filter`, sau do chay search tren tap da loc. `delete_document` xay dung lai danh sach moi, loai bo record co doc_id trung, tra ve True neu store co lai.

### KnowledgeBaseAgent.answer

> Luong RAG 3 buoc: (1) search(question, top_k) -> (2) noi content cac chunk thanh context (`\n\n`) -> (3) prompt `"Context:\n{context}\n\nQuestion: {question}\nAnswer:"` gui vao llm_fn.

---

## 3. Hoan thien code (Core Implementation) -- Ca nhan (30 diem)

**42 / 42 tests passed (pytest)**

```
============================= test session starts =============================
tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

============================= 42 passed in 0.10s ==============================
```

---

## 4. Du doan do tuong tu (Similarity Predictions) -- Ca nhan (5 diem)

Du doan truoc, doi chieu voi ket qua thuc te (mock embedder):

| Cap | Cau A (tom tat) | Cau B (tom tat) | Du doan | Diem thuc te | Dung? |
|-----|----------------|----------------|---------|--------------|-------|
| 1 | "Chinh sach tra hang Shopee cho nguoi mua" | "Quy dinh hoan tien nguoi mua Shopee" | cao | -0.0609 | No |
| 2 | "Nguoi ban can nop bang chung khieu nai" | "Shopee xu ly khieu nai trong 3-5 ngay" | cao | +0.1459 | Gan dung |
| 3 | "Thoi tiet hom nay rat dep" | "Chinh sach bao hanh thiet bi dien tu" | thap | +0.2222 | No (cao hon ky vong) |
| 4 | "Thoi han tra hang la 15 ngay" | "Thoi han phan hoi khieu nai la 2 ngay" | cao | -0.1597 | No |
| 5 | "Video dong goi hang la bang chung" | "Bien ban dong kiem voi buu ta" | cao | +0.3181 | Dung |

**Ket qua bat ngo nhat:** Cap 1 va Cap 4 cho diem am du cung chu de. Mock embedder chi dua tren thong ke ky tu, khong hieu nghia. Voi embedder thuc (Gemini / sentence-transformers), Cap 1 va Cap 4 chac chan cho diem cao.

---

## 5. Ket qua truy xuat cua toi (Competition Results) -- Ca nhan (10 diem)

**Chien luoc:** SentenceChunker(max_sentences_per_chunk=3)
**Cau hinh:** 7 tai lieu Shopee -> 41 chunks | EmbeddingStore in-memory | mock embedder

### 5 Query + Gold Answer + Retrieval

| # | Query | Gold Answer (trich tu tai lieu that) | Nguon / Section | Filter | Score | Ket qua |
|---|-------|--------------------------------------|-----------------|--------|-------|---------|
| Q1 | Nguoi ban phai phan hoi tra hang trong bao nhieu ngay? | "Nguoi ban can phan hoi trong vong 3 NGAY LAM VIEC ke tu khi nhan duoc thong bao." | shopee_seller_dispute_response / Thoi gian phan hoi | -- | 0.3128 | top-3 |
| Q2 | Hang thuc pham tuoi song bi khieu nai, nguoi ban phan hoi trong bao lau? | "Don hang thuc pham tuoi song & dong lanh: Phan hoi trong vong 24 GIO." | shopee_seller_dispute_response / Cac truong hop dac biet | audience=seller | 0.1735 | top-3 |
| Q3 | Nguoi mua yeu cau tra hang dien tu loi nha san xuat (DOA) can gi? | "Thoi han: 30 NGAY ke tu ngay nhan hang. Dieu kien: San pham chua qua su dung, con nguyen seal va phu kien." | shopee_buyer_return_timeline / San pham DOA | audience=buyer | 0.3515 | **TOP-1** |
| Q4 | Hau qua khi nguoi ban khong phan hoi khieu nai dung han? | "Shopee se TU DONG CHAP NHAN yeu cau. Viec phan hoi tre co the anh huong den DIEM DANH GIA cua hang." | shopee_seller_dispute_response / Hau qua khi khong phan hoi | -- | 0.3072 | MISS |
| Q5 | Video bang chung dong goi phai dap ung yeu cau gi? | "Video: xuyen suot, khong cat ghep, KHONG QUA 100MB (toi da 1 phut). Dinh dang: MP4, AVI." | shopee_seller_evidence_guide / Yeu cau ve bang chung | -- | 0.2864 | top-3 |

**Top-3 hits: 4/5 | Top-1 hits: 1/5**

**Xac nhan gold answer trich duoc tu tai lieu that:**
- Q1: shopee_seller_dispute_response.md dong 22: "trong vong 3 ngay lam viec"
- Q2: shopee_seller_dispute_response.md dong 26: "Phan hoi trong vong 24 gio"
- Q3: shopee_buyer_return_timeline.md dong 33: "30 ngay ke tu ngay nhan hang"
- Q4: shopee_seller_dispute_response.md dong 47-48: "tu dong chap nhan" + "diem danh gia"
- Q5: shopee_seller_evidence_guide.md dong 60: "khong qua 100MB/video (toi da 1 phut)"

### Phan tich ket qua

Q3 hit TOP-1 nho metadata filter `audience=buyer` thu hep search space -- minh chung gia tri cua search_with_filter.
Q4 MISS vi mock embedder khong hieu "hau qua" <-> "tu dong chap nhan" (thong ke ky tu, khong phai ngu nghia).

### So sanh voi HeadingChunker (R3)

| Cau | SentenceChunker (toi) | HeadingChunker (R3) |
|-----|----------------------|---------------------|
| Q1 | top-3 | TOP-1 |
| Q2 | top-3 | TOP-1 |
| Q3 | TOP-1 | miss |
| Q4 | miss | miss |
| Q5 | top-3 | miss |
| Tong top-3 | 4/5 | 2/5 |

HeadingChunker thang o Q1/Q2 vi heading phan anh truc tiep tu khoa query. SentenceChunker thang o Q3 vi cau day du chua tu khoa ky thuat (DOA). Tren toan bo: SentenceChunker tot hon (4/5 vs 2/5).

**Bai hoc lon nhat:** Benchmark-driven chunking: xac dinh gold answers truoc, kiem tra chung nam trong chunk nao, roi moi chon chien luoc. Metadata filter thay the duoc mot phan chat luong embedder (Q3: tu miss -> top-1 chi voi audience filter).

---

## Tu Danh Gia (Phan Ca Nhan)

| Tieu chi | Diem tu danh gia |
|----------|------------------|
| Khoi dong (Warm-up) | 5 / 5 |
| Huong tiep can (My Approach) | 10 / 10 |
| Hoan thien code (42/42 tests) | 30 / 30 |
| Du doan do tuong tu | 5 / 5 |
| Ket qua truy xuat | 8 / 10 |
| **Tong phan ca nhan** | **58 / 60** |
