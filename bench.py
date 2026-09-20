"""
Script benchmark theo yêu cầu Checkpoint 5 & 6 (Lab 07).
Chạy: python bench.py
In ra số chunk đã nạp và kết quả top-3 cho cả 5 câu hỏi benchmark, đồng thời ghi vào ket_qua_benchmark.txt.
"""
import sys
import io
from pathlib import Path
from src.chunking import SentenceChunker
from src.store import EmbeddingStore
from src.embeddings import _mock_embed
from src.models import Document

# Fix console encoding on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DATA_DIR = Path("data/ecommerce_policy")
SOURCES_FILE = DATA_DIR / "sources.csv"
OUTPUT_FILE = Path("ket_qua_benchmark.txt")

BENCHMARK_QUERIES = [
    {
        "id": "Q1",
        "query": "Người bán phải phản hồi yêu cầu trả hàng trong bao nhiêu ngày?",
        "gold": "Người bán cần phản hồi yêu cầu Trả hàng/Hoàn tiền của Người mua trong vòng 3 ngày làm việc kể từ khi nhận được thông báo.",
        "gold_doc": "shopee_seller_dispute_response",
        "filter": None,
    },
    {
        "id": "Q2",
        "query": "Khi giao hàng thực phẩm tươi sống bị khiếu nại, người bán phải phản hồi trong bao lâu?",
        "gold": "Đơn hàng thực phẩm tươi sống & đông lạnh: Phản hồi trong vòng 24 giờ.",
        "gold_doc": "shopee_seller_dispute_response",
        "filter": {"audience": "seller"},
    },
    {
        "id": "Q3",
        "query": "Người mua cần cung cấp gì khi yêu cầu trả hàng điện tử lỗi nhà sản xuất (DOA)?",
        "gold": "Thời hạn: 30 ngày kể từ ngày nhận hàng. Điều kiện: Sản phẩm chưa qua sử dụng, còn nguyên seal và phụ kiện.",
        "gold_doc": "shopee_buyer_return_timeline",
        "filter": {"audience": "buyer"},
    },
    {
        "id": "Q4",
        "query": "Hậu quả của việc người bán không phản hồi khiếu nại đúng hạn là gì?",
        "gold": "Nếu Người bán không phản hồi trong thời gian quy định, Shopee sẽ tự động chấp nhận yêu cầu Trả hàng/Hoàn tiền của Người mua. Việc không phản hồi hoặc phản hồi trễ có thể ảnh hưởng đến điểm đánh giá cửa hàng.",
        "gold_doc": "shopee_seller_dispute_response",
        "filter": None,
    },
    {
        "id": "Q5",
        "query": "Video bằng chứng đóng gói phải đáp ứng yêu cầu gì khi gửi cho Shopee?",
        "gold": "Dung lượng video tối đa: 100 MB. Thời lượng video tối đa: 1 phút. Định dạng hỗ trợ: MP4, AVI, MOV.",
        "gold_doc": "shopee_seller_evidence_guide",
        "filter": None,
    },
]

def load_documents():
    docs = []
    if SOURCES_FILE.exists():
        import csv
        with open(SOURCES_FILE, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fp = Path(row["file_path"])
                if not fp.exists():
                    continue
                text = fp.read_text(encoding="utf-8")
                if text.startswith("---"):
                    import re
                    text = re.sub(r"^---.*?---\s*", "", text, flags=re.DOTALL)
                meta = {
                    "doc_id": row["doc_id"],
                    "audience": row.get("audience", ""),
                    "category": row.get("category", ""),
                    "title": row.get("title", ""),
                }
                docs.append((row["doc_id"], text, meta))
    return docs

def run_benchmark():
    docs = load_documents()
    chunker = SentenceChunker(max_sentences_per_chunk=3)
    store = EmbeddingStore(embedding_fn=_mock_embed)
    
    total_chunks = 0
    chunk_id_counter = 0
    for doc_id, text, meta in docs:
        chunks = chunker.chunk(text)
        total_chunks += len(chunks)
        doc_objs = []
        for i, c in enumerate(chunks):
            chunk_id_counter += 1
            chunk_meta = dict(meta)
            chunk_meta["chunk_index"] = i
            doc_objs.append(Document(id=f"{doc_id}_{i}", content=c, metadata=chunk_meta))
        store.add_documents(doc_objs)

    lines = []
    lines.append(f"=== KẾT QUẢ BENCHMARK — CHIẾN LƯỢC: SentenceChunker(max_sentences=3) ===")
    lines.append(f"Tổng số tài liệu nạp: {len(docs)}")
    lines.append(f"Tổng số chunk đã nạp: {len(store._store)}")
    lines.append("-" * 70)

    for item in BENCHMARK_QUERIES:
        qid = item["id"]
        query = item["query"]
        filt = item["filter"]
        lines.append(f"\n[{qid}] Query: {query}")
        if filt:
            lines.append(f"    Filter: {filt}")
            results = store.search_with_filter(query, top_k=3, metadata_filter=filt)
        else:
            results = store.search(query, top_k=3)

        lines.append(f"    Gold Answer: {item['gold']}")
        lines.append("    Top-3 Retrieved:")
        for rank, res in enumerate(results, 1):
            doc_id = res.get("id", "N/A")
            score = res.get("score", 0.0)
            preview = res.get("content", "").replace("\n", " ")[:100]
            lines.append(f"      #{rank} [{score:+.4f}] id={doc_id} | {preview}...")

    output_text = "\n".join(lines)
    print(output_text)
    
    OUTPUT_FILE.write_text(output_text, encoding="utf-8")
    print(f"\nĐã ghi kết quả vào: {OUTPUT_FILE.resolve()}")

if __name__ == "__main__":
    run_benchmark()
