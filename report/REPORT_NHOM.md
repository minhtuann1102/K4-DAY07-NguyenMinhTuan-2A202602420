# Báo Cáo Nhóm — Lab 7: Embedding & Vector Store

**Nhóm:** 2A202602420
**Thành viên:** R1 (Data Lead) · R2 — Nguyễn Minh Tuấn (Benchmark Lead) · R3 (Strategy Lead)
**Ngày:** 2026-09-20

> **Nộp 1 bản / nhóm.** Phần cá nhân (hướng tiếp cận, kết quả riêng, dự đoán…) mỗi thành viên nộp riêng trong `REPORT_CANHAN.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần nhóm: 40** = Lựa chọn tài liệu (10) + Thiết kế chiến lược (15) + Chất lượng truy xuất (10) + Thuyết trình (5).

---

## 1. Lựa chọn tài liệu (Document Set Quality) — Nhóm (10 điểm)

### Chủ đề (Domain) & Lý Do Chọn

**Chủ đề:** Chính sách Đổi trả, Bảo hành và Quy định Người mua/Người bán trên Shopee (Thương mại điện tử Việt Nam)

**Tại sao nhóm chọn chủ đề này?**
> Chính sách đổi trả Shopee có sự phân chia rõ ràng giữa hai nhóm người dùng (`buyer` / `seller`), tạo điều kiện lý tưởng để kiểm thử metadata filter trong RAG. Nội dung tài liệu chứa nhiều con số cụ thể (3 ngày, 24 giờ, 30 ngày, 100MB…) giúp xây dựng câu hỏi benchmark có câu trả lời vàng (gold answer) có thể trích dẫn trực tiếp. Shopee là nền tảng phổ biến tại Việt Nam, nội dung chính sách công khai và đồng bộ về cấu trúc Markdown.

### Danh sách tài liệu (Data Inventory)

| # | Tên tài liệu | Nguồn (Source URL) | Ngày lấy | Số ký tự | Metadata |
|---|-------------|-------------------|----------|----------|----------|
| 1 | shopee_buyer_return_timeline | help.shopee.vn/…/79180 | 2026-09-20 | ~2 800 | audience=buyer, category=chinh-sach-doi-tra |
| 2 | shopee_buyer_return_condition | help.shopee.vn/…/79140 | 2026-09-20 | ~7 500 | audience=buyer, category=chinh-sach-doi-tra |
| 3 | shopee_buyer_refund_timeline | help.shopee.vn/…/79180 | 2026-09-20 | ~5 100 | audience=buyer, category=chinh-sach-doi-tra |
| 4 | shopee_buyer_non_returnable | help.shopee.vn/…/79244 | 2026-09-20 | ~2 100 | audience=buyer, category=danh-muc-gioi-han |
| 5 | shopee_buyer_warranty_policy | help.shopee.vn/…/79250 | 2026-09-20 | ~2 700 | audience=buyer, category=chinh-sach-bao-hanh |
| 6 | shopee_seller_dispute_response | banhang.shopee.vn/…/1823 | 2026-09-20 | ~3 500 | audience=seller, category=quy-dinh-nguoi-ban |
| 7 | shopee_seller_evidence_guide | banhang.shopee.vn/…/1824 | 2026-09-20 | ~3 800 | audience=seller, category=quy-dinh-nguoi-ban |

**Danh sách kiểm tra quản trị dữ liệu (Data governance checklist):**
- [x] Tập tài liệu (Corpus) chỉ chứa nguồn công khai/được phép dùng và không chứa dữ liệu cá nhân, thông tin đăng nhập hoặc tài liệu nội bộ.
- [x] Mỗi tài liệu có `source_url`, `retrieved_at`, `document_version` (hoặc ngày hiệu lực) trong metadata.

### Cấu trúc Metadata (Metadata Schema)

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `doc_id` | string | `shopee_seller_dispute_response` | Định danh duy nhất để xoá/cập nhật từng tài liệu trong store |
| `audience` | enum | `buyer` / `seller` | Cho phép dùng `search_with_filter` để chỉ tìm trong tài liệu liên quan đến người mua hoặc người bán — tránh hallucination xuyên nhóm |
| `category` | string | `chinh-sach-doi-tra` | Lọc theo nghiệp vụ (đổi trả, bảo hành, giới hạn, khiếu nại) |
| `source_url` | string | `https://help.shopee.vn/…` | Truy xuất nguồn gốc cho câu trả lời của agent |
| `retrieved_at` | date | `2026-09-20` | Kiểm tra tính cập nhật của chính sách |

---

## 2. Thiết kế chiến lược (Strategy Design) — Nhóm (15 điểm)

> Mỗi thành viên thử **một chiến lược khác nhau** trên cùng bộ tài liệu; nhóm tổng hợp và so sánh ở đây.

### Phân tích đường cơ sở (Baseline Analysis)

Chạy `ChunkingStrategyComparator().compare()` trên 3 tài liệu đại diện (chunk_size=300):

| Tài liệu | Chiến lược | Số chunks | AvgLen (chars) | Giữ ngữ cảnh? |
|----------|-----------|-----------|----------------|---------------|
| shopee_buyer_return_timeline | FixedSize | 7 | 296.0 | ❌ Có thể cắt giữa câu |
| shopee_buyer_return_timeline | SentenceChunker | 7 | 277.6 | ✅ Giữ nguyên câu |
| shopee_buyer_return_timeline | RecursiveChunker | 11 | 175.8 | ✅ Bảo toàn đoạn |
| shopee_buyer_return_timeline | **HeadingChunker** | **9** | **215.0** | **✅ Mỗi chunk = 1 mục** |
| shopee_seller_dispute_response | FixedSize | 9 | 282.6 | ❌ |
| shopee_seller_dispute_response | SentenceChunker | 7 | 338.9 | ✅ |
| shopee_seller_dispute_response | RecursiveChunker | 12 | 196.9 | ✅ |
| shopee_seller_dispute_response | **HeadingChunker** | **12** | **196.7** | **✅** |
| shopee_buyer_non_returnable | FixedSize | 5 | 293.6 | ❌ |
| shopee_buyer_non_returnable | SentenceChunker | 4 | 345.2 | ✅ |
| shopee_buyer_non_returnable | RecursiveChunker | 7 | 196.6 | ✅ |
| shopee_buyer_non_returnable | **HeadingChunker** | **9** | **152.3** | **✅** |

### Chiến lược của từng thành viên

**Thành viên R1 (Data Lead)**
- **Loại chiến lược:** FixedSizeChunker
- **Mô tả & lý do chọn:** Chia đều theo số ký tự (chunk_size=500, overlap=50). Chiến lược đơn giản, dễ kiểm soát kích thước chunk, phù hợp khi chưa biết cấu trúc văn bản. Overlap 50 ký tự đảm bảo không bỏ sót thông tin ở ranh giới chunk.
- **Code snippet:**
```python
FixedSizeChunker(chunk_size=500, overlap=50)
```

**Thành viên R2 — Nguyễn Minh Tuấn (Benchmark Lead)**
- **Loại chiến lược:** SentenceChunker
- **Mô tả & lý do chọn:** Tách theo ranh giới câu (dấu `.`, `!`, `?`), gộp tối đa 3 câu/chunk. Phù hợp với văn bản chính sách Shopee vì mỗi câu thường chứa một quy định hoàn chỉnh — tách theo câu giữ trọn vẹn ý nghĩa, không bị cắt giữa chừng.
- **Code snippet:**
```python
SentenceChunker(max_sentences_per_chunk=3)
```

**Thành viên R3 (Strategy Lead)**
- **Loại chiến lược:** HeadingChunker *(custom, thêm vào `src/chunking.py`)*
- **Mô tả & lý do chọn:** Tách tại mỗi tiêu đề Markdown (`#`, `##`, `###`), giữ nguyên heading làm dòng đầu chunk. Chính sách Shopee có cấu trúc đề mục rõ ràng (mỗi section `##` tương ứng một nhóm quy định), nên HeadingChunker tạo ra chunk tự mô tả nhất — query về "thời hạn phản hồi" sẽ khớp ngay với chunk có tiêu đề `## 2. Thời gian phản hồi`.
- **Code snippet:**
```python
class HeadingChunker:
    """Tách Markdown tại các tiêu đề #, ##, ###."""
    def __init__(self, min_heading_level: int = 3) -> None:
        self.min_heading_level = min_heading_level
        self._heading_re = re.compile(
            r"^(#{1," + str(min_heading_level) + r"})\s+.+", re.MULTILINE
        )

    def chunk(self, text: str) -> list[str]:
        lines = text.splitlines(keepends=True)
        chunks, current = [], []
        for line in lines:
            if self._heading_re.match(line.rstrip()):
                if section := "".join(current).strip():
                    chunks.append(section)
                current = [line]
            else:
                current.append(line)
        if section := "".join(current).strip():
            chunks.append(section)
        return chunks or ([text.strip()] if text.strip() else [])
```

### So Sánh Giữa Các Thành Viên

Kết quả dựa trên bộ 5 câu hỏi benchmark (mock embedder):

| Thành viên | Chiến lược | Hits top-1 / 5 | Hits top-3 / 5 | Điểm mạnh | Điểm yếu |
|-----------|-----------|---------------|---------------|-----------|----------|
| R1 | FixedSize (500, overlap=50) | *(chờ R1 chạy)* | | Đơn giản, kích thước ổn định | Cắt giữa câu, mất ngữ cảnh |
| R2 | SentenceChunker (3 câu) | 1/5 | 4/5 | Câu hoàn chỉnh, chunk nhỏ gọn | Câu dài có thể tạo chunk lớn |
| R3 | HeadingChunker (###) | 2/5 | 2/5 | Chunk tự mô tả, dễ debug | Phụ thuộc chất lượng đề mục |

**Chiến lược nào tốt nhất cho chủ đề này? Tại sao?**
> Với mock embedder, cả hai chiến lược đều hạn chế vì vector không phản ánh ngữ nghĩa thực. **HeadingChunker** có top-1 cao hơn nhờ chunk chứa tiêu đề heading — tiêu đề giúp mock embedder tìm đúng hơn vì từ khoá trong query thường lặp lại trong heading. Với embedder thực (multilingual), **SentenceChunker** kỳ vọng vượt trội hơn vì câu đầy đủ mang ngữ nghĩa rõ hơn heading. Lý tưởng nhất là **HeadingChunker** giữ heading + **SentenceChunker** cho nội dung bên trong — kết hợp cả hai.

---

## 3. Câu hỏi đánh giá & Chất lượng truy xuất (Retrieval Quality) — Nhóm (10 điểm)

### Câu hỏi đánh giá & Câu trả lời chuẩn (R2 thiết kế)

> **5 câu hỏi**, đa dạng, có thể kiểm chứng; **Q2 và Q3** cần lọc metadata (`audience`) mới trả lời đúng đối tượng.

| # | Câu hỏi (Query) | Câu trả lời chuẩn (Gold Answer — trích từ tài liệu thật) | Chunk nguồn |
|---|----------------|----------------------------------------------------------|-------------|
| Q1 | Người bán phải phản hồi yêu cầu trả hàng trong bao nhiêu ngày? | "Người bán cần phản hồi yêu cầu Trả hàng/Hoàn tiền của Người mua trong vòng **3 ngày làm việc** kể từ khi nhận được thông báo." | `shopee_seller_dispute_response` / `## 2. Thời gian phản hồi` |
| Q2 *(filter: seller)* | Khi giao hàng thực phẩm tươi sống bị khiếu nại, người bán phải phản hồi trong bao lâu? | "Đơn hàng thực phẩm tươi sống & đông lạnh: Phản hồi trong vòng **24 giờ**." | `shopee_seller_dispute_response` / `### Các trường hợp đặc biệt` |
| Q3 *(filter: buyer)* | Người mua cần cung cấp gì khi yêu cầu trả hàng điện tử lỗi nhà sản xuất (DOA)? | "Thời hạn: **30 ngày** kể từ ngày nhận hàng. Điều kiện: Sản phẩm chưa qua sử dụng, còn nguyên seal và phụ kiện." | `shopee_buyer_return_timeline` / `### 2.3. Sản phẩm điện tử có hư hỏng từ nhà sản xuất (DOA)` |
| Q4 | Hậu quả của việc người bán không phản hồi khiếu nại đúng hạn là gì? | "Nếu Người bán không phản hồi trong thời gian quy định, Shopee sẽ **tự động chấp nhận** yêu cầu Trả hàng/Hoàn tiền của Người mua. Việc không phản hồi hoặc phản hồi trễ có thể ảnh hưởng đến **điểm đánh giá cửa hàng**." | `shopee_seller_dispute_response` / `## 4. Hậu quả khi không phản hồi đúng hạn` |
| Q5 | Video bằng chứng đóng gói phải đáp ứng yêu cầu gì khi gửi cho Shopee? | "Video: xuyên suốt, không cắt ghép, dung lượng **không quá 100MB/video (tối đa 1 phút)**. Định dạng: MP4, AVI." | `shopee_seller_evidence_guide` / `## 4. Yêu cầu về bằng chứng` |

### Tổng hợp chất lượng truy xuất của nhóm

| # | Câu hỏi | SentenceChunker (R2) | HeadingChunker (R3) | FixedSize (R1) | Ghi chú |
|---|---------|---------------------|--------------------|--------------|---------| 
| Q1 | Thời hạn phản hồi seller | ⚠️ top-3 | ✅ top-1 | *(chờ R1)* | HeadingChunker thắng nhờ heading `## 2.` |
| Q2 | Thực phẩm tươi 24h (filter=seller) | ⚠️ top-3 | ✅ top-1 | *(chờ R1)* | filter audience cần thiết |
| Q3 | DOA 30 ngày (filter=buyer) | ✅ top-1 | ❌ miss | *(chờ R1)* | SentenceChunker thắng cho câu buyer |
| Q4 | Hậu quả không phản hồi | ❌ miss | ❌ miss | *(chờ R1)* | Cả hai MISS vì mock embedder |
| Q5 | Yêu cầu video bằng chứng | ⚠️ top-3 | ❌ miss | *(chờ R1)* | Nội dung nằm sâu trong section |

**Lọc bằng metadata có giúp ích không? Ở câu hỏi nào?**
> Có, metadata filter `audience` giúp ích rõ ràng ở **Q2** và **Q3** — không dùng filter thì store trả về 0 kết quả (score = 0.0) vì filter được áp trước khi tính similarity. Sau khi thêm cột `audience` vào `sources.csv` và gán đúng metadata khi `add_documents`, Q2 và Q3 đã có kết quả trong top-3. Đây là minh chứng trực tiếp cho giá trị của metadata filter trong RAG: cùng một query "thực phẩm tươi sống + phản hồi" sẽ có nghĩa khác nhau tùy theo là người mua hay người bán.

---

## 4. Thuyết trình (Demo) & Bài học nhóm — Nhóm (5 điểm)

**Những phân tích (insights) hay nhất nhóm sẽ trình bày:**
> 1. **Metadata filter thực sự quan trọng**: Q2 và Q3 từ score=0.0 → có kết quả ngay khi thêm `audience` vào metadata. Không có filter, RAG cho người bán lẫn câu trả lời dành cho người mua và ngược lại.
> 2. **HeadingChunker vs SentenceChunker**: HeadingChunker thắng ở các câu truy vấn về thời hạn/quy trình vì heading giúp mock embedder nhận diện section đúng. SentenceChunker thắng ở câu kỹ thuật (DOA) vì câu đầy đủ chứa từ khóa đặc thù.
> 3. **Hạn chế của mock embedder**: Q4 MISS cả 2 chiến lược — câu về "hậu quả" không có từ khoá trùng với tên heading `## 4. Hậu quả`, cho thấy giới hạn của mock embedder và tầm quan trọng của semantic embedder thực.

**Bài học rút ra khi so sánh trong nhóm:**
> Cùng một bộ tài liệu (7 file Shopee) nhưng chiến lược chunking khác nhau dẫn đến số chunk rất khác (41 vs 93): SentenceChunker tạo 41 chunk nhỏ gọn, HeadingChunker tạo 93 chunk tự mô tả hơn. Kết quả retrieval phụ thuộc mạnh vào loại query — không có chiến lược nào thắng hoàn toàn. Điều này dẫn đến bài học cốt lõi: **chiến lược chunking phải được chọn dựa trên cấu trúc tài liệu cụ thể**, không phải theo quy tắc chung.

**Nếu làm lại, nhóm sẽ thay đổi gì trong chiến lược dữ liệu?**
> Sẽ dùng embedder thực (Gemini text-embedding-004 hoặc `sentence-transformers/paraphrase-multilingual`) ngay từ đầu thay vì mock, để benchmark phản ánh đúng chất lượng retrieval. Ngoài ra sẽ thiết kế bộ câu hỏi trước khi chunking (benchmark-driven chunking): xác định câu hỏi → xem câu trả lời gold nằm ở đâu → chọn chiến lược chunk đảm bảo gold answer nằm trong một chunk nguyên vẹn.

---

## Tự Đánh Giá (Phần Nhóm)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Lựa chọn tài liệu (Document Set Quality) | 9 / 10 |
| Thiết kế chiến lược (Strategy Design) | 13 / 15 |
| Chất lượng truy xuất (Retrieval Quality) | 8 / 10 |
| Thuyết trình (Demo) | *(chờ demo)* / 5 |
| **Tổng phần nhóm** | **30+ / 40** |
