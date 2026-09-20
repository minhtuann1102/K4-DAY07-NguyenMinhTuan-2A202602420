"""
Benchmark CP5 — R2 Lead
5 queries + gold answers, kiểm tra retrieval thực tế với HeadingChunker (R3)
và SentenceChunker (R2), có dùng metadata filter.
"""
import sys, io, re, csv
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, '.')
from pathlib import Path
from src.chunking import SentenceChunker, HeadingChunker
from src.store import EmbeddingStore
from src.embeddings import _mock_embed
from src.models import Document

DATA_DIR = Path("data/ecommerce_policy")
SOURCES = DATA_DIR / "sources.csv"

# ============================================================
# 5 QUERIES + GOLD ANSWERS (trích nguyên văn từ tài liệu thật)
# ============================================================
BENCHMARK = [
    {
        "id": "Q1",
        "query": "Người bán phải phản hồi yêu cầu trả hàng trong bao nhiêu ngày?",
        "gold": "Người bán cần phản hồi yêu cầu Trả hàng/Hoàn tiền của Người mua trong vòng 3 ngày làm việc kể từ khi nhận được thông báo.",
        "gold_doc": "shopee_seller_dispute_response",
        "gold_section": "## 2. Thời gian phản hồi",
        "needs_filter": False,
        "audience_filter": None,
    },
    {
        "id": "Q2",
        "query": "Khi giao hàng thực phẩm tươi sống bị khiếu nại, người bán phải phản hồi trong bao lâu?",
        "gold": "Đơn hàng thực phẩm tươi sống & đông lạnh: Phản hồi trong vòng 24 giờ.",
        "gold_doc": "shopee_seller_dispute_response",
        "gold_section": "### Các trường hợp đặc biệt",
        "needs_filter": True,
        "audience_filter": "seller",  # chỉ áp dụng cho người bán
    },
    {
        "id": "Q3",
        "query": "Người mua cần cung cấp gì khi yêu cầu trả hàng điện tử lỗi nhà sản xuất (DOA)?",
        "gold": "Thời hạn: 30 ngày kể từ ngày nhận hàng. Điều kiện: Sản phẩm chưa qua sử dụng, còn nguyên seal và phụ kiện.",
        "gold_doc": "shopee_buyer_return_timeline",
        "gold_section": "### 2.3. Sản phẩm điện tử có hư hỏng từ nhà sản xuất (DOA)",
        "needs_filter": True,
        "audience_filter": "buyer",  # chỉ áp dụng cho người mua
    },
    {
        "id": "Q4",
        "query": "Hậu quả của việc người bán không phản hồi khiếu nại đúng hạn là gì?",
        "gold": "Nếu Người bán không phản hồi trong thời gian quy định, Shopee sẽ tự động chấp nhận yêu cầu Trả hàng/Hoàn tiền của Người mua. Việc không phản hồi hoặc phản hồi trễ có thể ảnh hưởng đến điểm đánh giá cửa hàng.",
        "gold_doc": "shopee_seller_dispute_response",
        "gold_section": "## 4. Hậu quả khi không phản hồi đúng hạn",
        "needs_filter": False,
        "audience_filter": None,
    },
    {
        "id": "Q5",
        "query": "Video bằng chứng đóng gói phải đáp ứng yêu cầu gì khi gửi cho Shopee?",
        "gold": "Video: xuyên suốt, không cắt ghép, dung lượng không quá 100MB/video (tối đa 1 phút). Định dạng: MP4, AVI.",
        "gold_doc": "shopee_seller_evidence_guide",
        "gold_section": "## 4. Yêu cầu về bằng chứng",
        "needs_filter": False,
        "audience_filter": None,
    },
]


def load_store(chunker, strategy_name):
    store = EmbeddingStore(embedding_fn=_mock_embed)
    with open(SOURCES, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fp = Path(row['file_path'])
            if not fp.exists():
                continue
            text = fp.read_text(encoding='utf-8')
            if text.startswith('---'):
                text = re.sub(r'^---.*?---\s*', '', text, flags=re.DOTALL)
            chunks = chunker.chunk(text)
            docs = [
                Document(
                    id=f"{row['doc_id']}_{i}",
                    content=c,
                    metadata={
                        "doc_id": row['doc_id'],
                        "audience": row.get('audience', ''),
                        "title": row['title'],
                    }
                )
                for i, c in enumerate(chunks)
            ]
            store.add_documents(docs)
    return store


def score_query(store, bm):
    """Return (hit_top1, hit_top3, top1_score, top1_content)"""
    q = bm["query"]
    filt = bm["audience_filter"]

    if filt and bm["needs_filter"]:
        top3 = store.search_with_filter(q, metadata_filter={"audience": filt}, top_k=3)
    else:
        top3 = store.search(q, top_k=3)

    gold_doc = bm["gold_doc"]
    gold_section_kw = bm["gold_section"].replace("## ", "").replace("### ", "").split(".")[0].strip()

    hit_top1 = False
    hit_top3 = False
    top1 = top3[0] if top3 else {"content": "", "score": 0}

    for i, r in enumerate(top3):
        in_gold_doc = gold_doc in r.get("metadata", {}).get("doc_id", "")
        if in_gold_doc:
            if i == 0:
                hit_top1 = True
            hit_top3 = True
            break

    return hit_top1, hit_top3, top1.get("score", 0), top1.get("content", "")[:80]


# ======================== RUN ========================
print("=" * 70)
print("BENCHMARK CP5 — R2 Lead")
print("=" * 70)

for strategy_name, chunker in [
    ("SentenceChunker(3)", SentenceChunker(max_sentences_per_chunk=3)),
    ("HeadingChunker(3)",  HeadingChunker(min_heading_level=3)),
]:
    print(f"\n--- Chiến lược: {strategy_name} ---")
    store = load_store(chunker, strategy_name)
    print(f"    Tổng chunks: {len(store._store)}")
    hits1 = hits3 = 0
    for bm in BENCHMARK:
        h1, h3, score, preview = score_query(store, bm)
        if h1: hits1 += 1
        if h3: hits3 += 1
        filter_tag = f"[filter={bm['audience_filter']}]" if bm["needs_filter"] else ""
        top1_tag = "✅ TOP1" if h1 else ("⚠️  TOP3" if h3 else "❌ MISS")
        print(f"  {bm['id']} {top1_tag} score={score:.4f} {filter_tag}")
        print(f"     Q : {bm['query']}")
        print(f"     T1: {preview}")
    print(f"  => Hits top-1: {hits1}/5  |  Hits top-3: {hits3}/5")

print("\n" + "=" * 70)
print("GOLD ANSWERS (trích từ tài liệu thật)")
print("=" * 70)
for bm in BENCHMARK:
    print(f"\n{bm['id']} [{bm['gold_doc']} / {bm['gold_section']}]")
    print(f"  Q: {bm['query']}")
    print(f"  A: {bm['gold']}")
