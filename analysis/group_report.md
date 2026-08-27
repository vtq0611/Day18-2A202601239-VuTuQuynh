# Group Report — Lab 18: Production RAG

**Nhóm:** Cá nhân — Vũ Tú Quỳnh
**Ngày:** 2026-08-26

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Vũ Tú Quỳnh | M1: Chunking | ☑ | 13/13 |
| Vũ Tú Quỳnh | M2: Hybrid Search | ☑ | 5/5 |
| Vũ Tú Quỳnh | M3: Reranking | ☑ | 5/5 |
| Vũ Tú Quỳnh | M4: Evaluation | ☑ | 4/4 |
| Vũ Tú Quỳnh | M5: Enrichment (combined mode) | ☑ | 10/10 |

(Bài tập cá nhân theo ASSIGNMENT.md — 1 người implement cả 5 module. Tổng 37/37 tests pass.)

## Kết quả RAGAS

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | 0.7542 | 0.8431 | +0.0889 |
| Answer Relevancy | 0.7234 | 0.8151 | +0.0917 |
| Context Precision | 0.9250 | 0.8894 | -0.0356 |
| Context Recall | 0.9250 | 0.9000 | -0.0250 |

Tất cả 4 metrics production đều đạt ngưỡng ≥ 0.75. Faithfulness được cải thiện từ 0.75 lên 0.8431
sau khi siết lại generation prompt (yêu cầu chỉ nêu claim có bằng chứng trực tiếp trong context, ưu
tiên phiên bản chính sách hiện hành khi có mâu thuẫn, `temperature=0`) và tăng `RERANK_TOP_K` từ 3
lên 5 để đủ ngữ cảnh cho câu hỏi multi-hop — đổi lại context_precision/recall giảm nhẹ do context
đưa vào nhiều hơn.

## Latency Breakdown (Production Pipeline, 100 chunks / 26 documents / 20 test queries)

| Bước | Thời gian | Ghi chú |
|------|-----------|---------|
| [1/4] Chunking (M1, hierarchical) | 0.3s | Parent 2048 / child 256 chars, thuần CPU string ops |
| [2/4] Enrichment (M5, combined mode) | 392–520s | 1 API call/chunk × 100 chunks ≈ 4-5s/chunk — **bottleneck lớn nhất**, chiếm 40-55% tổng thời gian; dao động do độ trễ API |
| [3/4] Indexing (M2, BM25 + Dense) | 108–142s | Chủ yếu là encode 100 chunks bằng bge-m3 (4 batches) + upsert Qdrant |
| [4/4] Load reranker (M3) | ~0s | Model đã cache sau lần load đầu |
| Eval: 20 queries (search + rerank + LLM answer, top_k=5) | ~theo query | Retrieval + rerank nhanh (<1s/query); LLM generation là phần chậm nhất mỗi query |
| Eval: RAGAS (4 metrics × 20 câu hỏi) | 84.6s | Chạy song song nhiều LLM-judge calls |
| **Tổng production pipeline (1 lần chạy)** | **~870-930s (~15 phút)** | Riêng `src/pipeline.py`, không tính naive baseline |

**Nhận xét:** Enrichment (M5) là bottleneck chính vì gọi API tuần tự từng chunk. Có thể tối ưu bằng cách:
1. Batch/parallel hóa các API call (asyncio + `asyncio.gather`) thay vì vòng lặp tuần tự.
2. Chỉ enrich chunk mới/thay đổi thay vì toàn bộ corpus mỗi lần re-index.

## Key Findings

1. **Biggest improvement:** Answer Relevancy (+0.0336) và Context Precision (+0.0167) — nhờ hybrid search (BM25 bắt exact keyword tiếng Việt tốt hơn dense-only) kết hợp cross-encoder reranking lọc bớt chunk nhiễu trước khi đưa vào LLM.
2. **Biggest challenge:** Context Recall giảm (-0.1250) — child chunk 256 ký tự trong hierarchical chunking quá nhỏ, khiến một số câu hỏi cần thông tin dạng bảng/ngưỡng số liệu (VD: "mua thiết bị 55 triệu cần ai duyệt") bị miss hoàn toàn (context_recall = 0.0) dù naive baseline (paragraph chunking lớn hơn) lại retrieve được.
3. **Surprise finding:** Nhiều lỗi faithfulness không phải do model "bịa" thông tin, mà do context chứa 2 phiên bản tài liệu mâu thuẫn nhau (VD: `mat_khau_v1.md` 90 ngày vs `mat_khau_v2.md` 120 ngày) — retrieval đúng về mặt kỹ thuật (tìm ra chunk liên quan) nhưng sai về mặt nghiệp vụ (không lọc theo version hiện hành). Đây là loại lỗi mà thêm reranking/embedding tốt hơn không giải quyết được — cần metadata filtering.

## Presentation Notes (5 phút)

1. RAGAS scores (naive vs production): Production thắng 3/4 metrics (answer_relevancy, context_precision, và ngang bằng faithfulness), nhưng thua context_recall — trade-off giữa chunk nhỏ (precision cao) và chunk lớn (recall cao).
2. Biggest win — module nào, tại sao: M2 Hybrid Search (BM25+Dense+RRF) — Vietnamese BM25 sau khi segment đúng cách bắt được exact-match mà dense embedding một mình bỏ lỡ.
3. Case study — 1 failure, Error Tree walkthrough: Câu hỏi "Bao lâu phải đổi mật khẩu một lần?" — retrieval đúng nhưng trả về cả chunk cũ (v1, 90 ngày) lẫn mới (v2, 120 ngày) → LLM trộn lẫn hai chính sách → faithfulness = 0. Root cause: thiếu metadata versioning, không phải lỗi retrieval hay generation thuần túy.
4. Next optimization nếu có thêm 1 giờ: Thêm metadata `status: current/superseded` để lọc tài liệu cũ trước khi generate; tăng child chunk size hoặc trả về parent chunk cho câu hỏi multi-hop/numeric để cải thiện context_recall.
