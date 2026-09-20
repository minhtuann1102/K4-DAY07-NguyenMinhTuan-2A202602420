# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Minh Tuấn  
**Nhóm:** 2A202602420  
**Ngày:** 2026-09-20  

> **Nộp 1 bản / sinh viên.** Phần nhóm nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao nghĩa là gì?**
> Độ tương tự cosine đo góc giữa hai vector embedding trong không gian nhiều chiều. Khi hai đoạn văn bản có độ tương tự cosine cao (tiến gần 1.0), điều đó biểu thị chúng mang cùng chủ đề hoặc ý nghĩa ngữ nghĩa tương đồng, dù có thể dùng từ ngữ khác nhau — chúng "hướng về cùng một phía" trong không gian vector.

**Ví dụ CÓ ĐỘ TƯƠNG TỰ CAO:**
- **Câu A:** "Người mua có 15 ngày để yêu cầu trả hàng hoàn tiền trên Shopee."
- **Câu B:** "Thời hạn gửi yêu cầu đổi trả và hoàn tiền cho Người mua là 15 ngày kể từ khi nhận hàng."
- **Tại sao:** Cả hai câu đều diễn đạt cùng một quy định nghiệp vụ (thời hạn đổi trả 15 ngày) nên các vector embedding sẽ nằm rất gần nhau trong không gian ngữ nghĩa.

**Ví dụ CÓ ĐỘ TƯƠNG TỰ THẤP:**
- **Câu A:** "Thời tiết hôm nay rất đẹp, trời nắng ấm và gió mát."
- **Câu B:** "Chính sách bảo hành thiết bị điện tử trên Shopee có hiệu lực từ ngày giao hàng."
- **Tại sao:** Hai câu thuộc hai lĩnh vực hoàn toàn độc lập, không có sự liên quan về ngữ nghĩa hay ngữ cảnh (thời tiết vs. quy định TMĐT).

**Tại sao cosine similarity được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Khoảng cách Euclid phụ thuộc trực tiếp vào độ dài (độ lớn magnitude) của vector — các văn bản dài hơn thường có vector lớn hơn dù cùng một nội dung ý nghĩa. Ngược lại, Cosine similarity chỉ đo **góc (hướng)** giữa hai vector mà bỏ qua độ dài văn bản, do đó phản ánh chuẩn xác và khách quan hơn mức độ tương đồng về mặt ngữ nghĩa.

---

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50 → bao nhiêu chunks?**

> **Phép tính:**
> ```text
> step = chunk_size - overlap = 500 - 50 = 450
> số_chunk = làm_tròn_lên((10000 - 50) / 450)
>           = làm_tròn_lên(9950 / 450)
>           = làm_tròn_lên(22.11)
>           = 23 chunks
> ```
> **Đáp án:** **23 chunks**

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**

> **Phép tính:**
> ```text
> step = 500 - 100 = 400
> số_chunk = làm_tròn_lên((10000 - 100) / 400)
>           = làm_tròn_lên(9900 / 400)
>           = làm_tròn_lên(24.75)
>           = 25 chunks
> ```
> Số lượng chunk tăng từ **23 lên 25** (tăng thêm 2 chunks).  
> **Lý do muốn tăng overlap:** Trong các văn bản điều khoản, những thông tin quan trọng (như mốc thời gian, điều kiện ràng buộc) rất dễ rơi trúng vào **ranh giới cắt giữa hai chunk**. Độ chồng chéo (overlap) lớn giúp bảo toàn ngữ cảnh liên tục, đảm bảo thông tin quan trọng xuất hiện trọn vẹn ở cả chunk trước và chunk sau, ngăn chặn việc retrieval bị mất ngữ nghĩa.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`:**
> Tôi sử dụng biểu thức chính quy (regex) `re.split(r"(?<=[.!?])\s+|(?<=\.)\n+", text.strip())` để nhận diện chính xác điểm kết thúc của câu (sau dấu chấm, chấm than, chấm hỏi hoặc ngắt dòng sau dấu chấm). Sau đó, các câu đơn lẻ được gom nhóm lại thành từng chunk với tối đa `max_sentences_per_chunk` câu (ở đây là 3 câu) và nối lại bằng dấu cách. Xử lý triệt để các trường hợp biên: văn bản rỗng trả về `[]`, lọc sạch các khoảng trắng thừa bằng `.strip()`.

**`RecursiveChunker._split`:**
> Thuật toán đệ quy thử phân tách văn bản lần lượt theo danh sách các dấu phân cách ưu tiên từ lớn đến nhỏ: `["\n\n", "\n", ". ", " ", ""]`. Trường hợp cơ sở (base case): nếu độ dài đoạn văn nhỏ hơn hoặc bằng `chunk_size` thì giữ nguyên. Nếu dấu phân cách hiện tại không thể chia nhỏ văn bản, hàm tự động hạ bậc xuống dấu phân cách tiếp theo. Khi chia được, thuật toán sử dụng một cơ chế bộ đệm (buffer) để gộp các mảnh nhỏ liền kề nhau trước khi cắt, tránh việc phân mảnh văn bản quá vụn.

**`HeadingChunker.chunk` (Chiến lược riêng theo Heading Markdown):**
> Nhận diện các dòng tiêu đề Markdown bằng biểu thức chính quy `^(#{1,N})\s+.+`. Mỗi khi bắt gặp một tiêu đề mới, section hiện tại sẽ được chốt lại (flush) thành một chunk hoàn chỉnh. Tiêu đề luôn được giữ lại ở dòng đầu tiên của chunk để chunk tự mang đầy đủ ngữ cảnh của mục đó.

### Lớp EmbeddingStore

**`add_documents` + `search`:**
> Mỗi `Document` được chuẩn hóa qua hàm `_make_record`, tạo vector embedding và sao chép `metadata` (đảm bảo luôn tồn tại trường `doc_id`). Toàn bộ bản ghi được lưu trữ an toàn trong danh sách bộ nhớ `self._store` (in-memory). Khi tìm kiếm (`search`), hàm `_search_records` tạo embedding cho câu query và tính tích vô hướng (`_dot`) với từng bản ghi trong store (tỷ lệ thuận với cosine similarity do vector đã được chuẩn hóa L2). Kết quả được sắp xếp giảm dần theo điểm và lấy ra `top_k` phần tử tốt nhất (đã loại bỏ vector embedding để không làm nặng dữ liệu trả về).

**`search_with_filter` + `delete_document`:**
> `search_with_filter` thực hiện cơ chế **lọc trước (pre-filtering)**: quét qua toàn bộ store, chỉ giữ lại các bản ghi khớp chính xác với tất cả các cặp key-value trong `metadata_filter`, sau đó mới chuyển tập hợp này qua thuật toán tìm kiếm tương đồng. Điều này ngăn chặn triệt để tình trạng mất kết quả hợp lệ do top-k bị chiếm bởi tài liệu sai đối tượng. `delete_document` lọc bỏ mọi bản ghi có `doc_id` hoặc `id` trùng khớp, trả về `True` nếu số lượng phần tử trong store giảm đi.

### Tác tử KnowledgeBaseAgent

**`KnowledgeBaseAgent.answer`:**
> Triển khai quy trình RAG chuẩn 3 bước:  
> 1. Truy xuất: Kiểm tra store (nếu rỗng thì trả về thông báo ngay, tránh gọi LLM vô ích), gọi `store.search(question, top_k)` để lấy các chunk phù hợp nhất.  
> 2. Dựng ngữ cảnh: Đánh số thứ tự từng chunk `[1]`, `[2]`, `[3]` kèm nguồn tham chiếu `(Nguồn: ...)` nhằm đáp ứng tiêu chí **Truy xuất nguồn gốc (Source Traceability)**.  
> 3. Tạo prompt và suy luận: Thiết lập ràng buộc chống bịa đặt (hallucination), yêu cầu mô hình chỉ trả lời dựa trên ngữ cảnh được cung cấp và trích dẫn rõ số thứ tự nguồn, sau đó chuyển qua `llm_fn`.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

**Kết quả kiểm thử tự động: 42 / 42 bài test PASSED (pytest tests/ -v)**

```text
============================= test session starts =============================
platform win32 -- Python 3.11.4, pytest-9.1.1, pluggy-1.6.0
collected 42 items

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

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Dự đoán trước khi chạy hàm `compute_similarity()`, đối chiếu với kết quả thực tế (dùng mock embedder):

| Cặp | Câu A (tóm tắt) | Câu B (tóm tắt) | Dự đoán | Điểm thực tế | Đúng? |
|:---:|:---|:---|:---:|:---:|:---:|
| 1 | "Chính sách trả hàng Shopee cho người mua" | "Quy định hoàn tiền người mua Shopee" | Cao | **-0.0609** | ❌ Sai |
| 2 | "Người bán cần nộp bằng chứng khiếu nại" | "Shopee xử lý khiếu nại trong 3-5 ngày" | Cao | **+0.1459** | ⚠️ Gần đúng |
| 3 | "Thời tiết hôm nay rất đẹp" | "Chính sách bảo hành thiết bị điện tử" | Thấp | **+0.2222** | ❌ Sai (cao hơn kỳ vọng) |
| 4 | "Thời hạn trả hàng là 15 ngày" | "Thời hạn phản hồi khiếu nại là 2 ngày" | Cao | **-0.1597** | ❌ Sai |
| 5 | "Video đóng gói hàng là bằng chứng" | "Biên bản đồng kiểm với bưu tá" | Cao | **+0.3181** | ✅ Đúng |

**Kết quả bất ngờ nhất và nhận xét:**  
Bất ngờ nhất là Cặp 1 và Cặp 4 bị điểm âm dù cùng chủ đề chính sách đổi trả. Nguyên nhân là `MockEmbedder` băm chuỗi MD5 và tạo vector ngẫu nhiên dựa trên ký tự chứ không mã hóa ngữ nghĩa thực. Với một mô hình embedding ngôn ngữ thực thụ (như Gemini Embedding hay Vietnamese Sentence-Transformers), Cặp 1 và Cặp 4 chắc chắn sẽ có cosine similarity cao (>0.8).

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

* **Chiến lược:** `SentenceChunker(max_sentences_per_chunk=3)`  
* **Cấu hình:** 7 tài liệu Shopee → 41 chunks | `EmbeddingStore` in-memory | Mock embedder  

### Bảng 5 Query + Gold Answer + Kết quả Retrieval

| # | Query | Gold Answer (Trích từ tài liệu thật) | Nguồn / Section | Metadata Filter | Score | Kết quả |
|:---:|:---|:---|:---|:---:|:---:|:---:|
| **Q1** | Người bán phải phản hồi yêu cầu trả hàng trong bao nhiêu ngày? | "Người bán cần phản hồi yêu cầu Trả hàng/Hoàn tiền của Người mua trong vòng **3 ngày làm việc** kể từ khi nhận được thông báo." | `shopee_seller_dispute_response` / Mục 2. Thời gian phản hồi | — | +0.3128 | ⚠️ top-3 |
| **Q2** | Khi giao hàng thực phẩm tươi sống bị khiếu nại, người bán phải phản hồi trong bao lâu? | "Đơn hàng thực phẩm tươi sống & đông lạnh: Phản hồi trong vòng **24 giờ**." | `shopee_seller_dispute_response` / Các trường hợp đặc biệt | `audience=seller` | +0.1735 | ⚠️ top-3 |
| **Q3** | Người mua cần cung cấp gì khi yêu cầu trả hàng điện tử lỗi nhà sản xuất (DOA)? | "Thời hạn: **30 ngày** kể từ ngày nhận hàng. Điều kiện: Sản phẩm chưa qua sử dụng, còn nguyên seal và phụ kiện." | `shopee_buyer_return_timeline` / Mục 2.3. Sản phẩm DOA | `audience=buyer` | +0.3515 | ✅ **TOP-1** |
| **Q4** | Hậu quả của việc người bán không phản hồi khiếu nại đúng hạn là gì? | "Shopee sẽ **tự động chấp nhận** yêu cầu. Việc không phản hồi hoặc phản hồi trễ có thể ảnh hưởng đến **điểm đánh giá cửa hàng**." | `shopee_seller_dispute_response` / Mục 4. Hậu quả | — | +0.3072 | ❌ MISS |
| **Q5** | Video bằng chứng đóng gói phải đáp ứng yêu cầu gì khi gửi cho Shopee? | "Dung lượng video tối đa: **100 MB**. Thời lượng video tối đa: **1 phút**. Định dạng hỗ trợ: MP4, AVI, MOV." | `shopee_seller_evidence_guide` / Mục 4. Yêu cầu bằng chứng | — | +0.2864 | ⚠️ top-3 |

**Tỷ lệ truy xuất đạt:** Top-3 hits: **4 / 5** (80%) | Top-1 hits: **1 / 5** (20%)

**Xác nhận vị trí Gold Answer trong tài liệu nguồn:**
- **Q1:** [shopee_seller_dispute_response.md dòng 22](file:///c:/Users/NITRO/ProjectLab1/K4-DAY07-NguyenMinhTuan-2A202602420/data/ecommerce_policy/shopee_seller_dispute_response.md#L22): "trong vòng 3 ngày làm việc"
- **Q2:** [shopee_seller_dispute_response.md dòng 26](file:///c:/Users/NITRO/ProjectLab1/K4-DAY07-NguyenMinhTuan-2A202602420/data/ecommerce_policy/shopee_seller_dispute_response.md#L26): "Phản hồi trong vòng 24 giờ"
- **Q3:** [shopee_buyer_return_timeline.md dòng 33](file:///c:/Users/NITRO/ProjectLab1/K4-DAY07-NguyenMinhTuan-2A202602420/data/ecommerce_policy/shopee_buyer_return_timeline.md#L33): "30 ngày kể từ ngày nhận hàng"
- **Q4:** [shopee_seller_dispute_response.md dòng 47-48](file:///c:/Users/NITRO/ProjectLab1/K4-DAY07-NguyenMinhTuan-2A202602420/data/ecommerce_policy/shopee_seller_dispute_response.md#L47-L48): "tự động chấp nhận" + "điểm đánh giá"
- **Q5:** [shopee_seller_evidence_guide.md dòng 60](file:///c:/Users/NITRO/ProjectLab1/K4-DAY07-NguyenMinhTuan-2A202602420/data/ecommerce_policy/shopee_seller_evidence_guide.md#L60): "không quá 100MB/video (tối đa 1 phút)"

### Phân tích kết quả
- **Câu Q3 đạt Top-1:** Nhờ sử dụng `metadata_filter={"audience": "buyer"}`, không gian tìm kiếm được thu hẹp, loại bỏ toàn bộ các tài liệu của Người bán, giúp truy xuất chuẩn xác chunk DOA.
- **Câu Q4 bị Miss:** Do `_mock_embed` không hiểu ngữ nghĩa của từ "hậu quả" liên quan đến "tự động chấp nhận yêu cầu" và "điểm đánh giá".

### So sánh với HeadingChunker (R3)

| Câu hỏi | SentenceChunker (Tôi) | HeadingChunker (R3) |
|:---|:---:|:---:|
| Q1 (Phản hồi trả hàng 3 ngày) | ⚠️ top-3 | ✅ **TOP-1** |
| Q2 (Thực phẩm tươi sống 24h) | ⚠️ top-3 | ✅ **TOP-1** |
| Q3 (Hàng điện tử DOA 30 ngày) | ✅ **TOP-1** | ❌ miss |
| Q4 (Hậu quả không phản hồi) | ❌ miss | ❌ miss |
| Q5 (Yêu cầu video đóng gói) | ⚠️ top-3 | ❌ miss |
| **Tổng kết Top-3** | **4 / 5** | **2 / 5** |

**Đánh giá:** `HeadingChunker` vượt trội ở Q1 và Q2 vì tiêu đề mục chứa trực tiếp từ khóa tìm kiếm. Tuy nhiên, trên toàn cục 5 câu, `SentenceChunker` đạt tỷ lệ lọt Top-3 cao hơn hẳn (4/5 so với 2/5) do giữ được ngữ cảnh câu đầy đủ.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|:---|:---:|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — 42/42 tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 8 / 10 |
| **Tổng điểm cá nhân tự đánh giá** | **58 / 60** |
