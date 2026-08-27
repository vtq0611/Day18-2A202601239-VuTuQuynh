# Reflection — Lab 18: Production RAG Pipeline

**Họ tên:** Vũ Tú Quỳnh
**Ngày:** 2026-08-26

---

## Phần 1: Mapping bài giảng → code

| Lecture Concept | Module | Hàm cụ thể | Observation |
|----------------|--------|-------------|-------------|
| Semantic chunking | M1 | `chunk_semantic()` | Threshold 0.85 (cosine sim giữa câu liên tiếp) nhóm các câu cùng chủ đề thành chunk; khác basic chunking (chỉ split theo `\n\n`, cố định 500 ký tự) — semantic chunking không cắt giữa một ý đang triển khai, nhưng chunk size không đều (phụ thuộc độ liên kết ngữ nghĩa của văn bản). |
| Hierarchical (parent-child) chunking | M1 | `chunk_hierarchical()` | Parent 2048 ký tự giữ đủ ngữ cảnh, child 256 ký tự để embed/retrieve chính xác hơn. Pipeline dùng child để index, nhưng thực tế trong `pipeline.py` vẫn trả nguyên text của child (chưa "expand" lên parent lúc generate answer) — đây là điểm có thể cải thiện thêm để tăng context recall cho câu hỏi multi-hop. |
| BM25 + Dense fusion (RRF) | M2 | `reciprocal_rank_fusion()` | RRF giải quyết vấn đề BM25 và Dense có scale điểm số khác nhau (BM25 score không bounded, cosine similarity 0-1) — bằng cách chỉ dùng **rank** thay vì raw score nên không cần chuẩn hóa. Với query tiếng Việt như "nghỉ phép", BM25 (sau khi segment bằng underthesea) bắt được exact keyword match tốt, còn Dense (bge-m3) bắt được synonym/paraphrase — kết hợp 2 cái giúp context_precision production (0.9417) cao hơn naive dense-only (0.9250). |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | bge-reranker-v2-m3 rerank top-20 (hybrid) xuống top-3. Load model mất vài giây do phải tải weights lần đầu, nhưng inference theo batch khá nhanh. Answer relevancy production (0.7570) cao hơn naive (0.7234) — một phần nhờ reranker loại bỏ chunk nhiễu trước khi đưa vào LLM. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Chạy trên 20 câu hỏi test set. Lần đầu (prompt gốc, top_k=3): faithfulness chỉ 0.75, ngang naive. Sau khi siết prompt (ưu tiên phiên bản hiện hành, chỉ nêu claim có bằng chứng, temperature=0) và tăng RERANK_TOP_K 3→5: faithfulness lên 0.8431-0.8464, answer_relevancy 0.757→0.815, đổi lại context_precision giảm nhẹ (0.942→0.889) vì đưa nhiều chunk hơn vào context. RAGAS dùng LLM-judge nên điểm số dao động ~±0.02-0.05 giữa các lần chạy giống nhau — không nên coi 1 lần chạy là con số tuyệt đối. |
| Failure analysis / Diagnostic Tree | M4 | `failure_analysis()` | Bottom-10 câu hỏi được map vào 4 category lỗi (faithfulness→hallucination, context_recall→missing chunk, context_precision→noisy chunk, answer_relevancy→prompt mismatch). Đa số failures rơi vào faithfulness do câu hỏi multi-hop hoặc versioning conflict (mật khẩu v1 vs v2) — không phải do model bịa hoàn toàn mà do context có 2 nguồn mâu thuẫn nhau. |
| Contextual embeddings / Enrichment | M5 | `_enrich_single_call()` (combined mode) | Dùng 1 API call/chunk (thay vì 4 call riêng: summary + HyQA + contextual + metadata) để tiết kiệm chi phí. Với 100 chunks, enrichment mất ~359s (~3.6s/chunk) — bottleneck lớn nhất trong toàn pipeline (909s tổng, enrichment chiếm ~40%). Enriched text thêm 1 câu context ("Trích từ tài liệu X, nói về...") giúp bridge vocabulary gap, nhưng đo bằng RAGAS thì hiệu ứng chưa rõ rệt trên context_recall (vẫn giảm so với naive) — có thể do child chunk quá nhỏ làm giảm hiệu ứng của prepend context. |

## Phần 2: Khó khăn & giải quyết

1. **Lỗi:** `UnicodeEncodeError: 'charmap' codec can't encode characters in position 2-3` khi chạy `naive_baseline.py`/`main.py` trên Windows console.
   - **Debug:** Traceback chỉ ra lỗi ở `print()` với ký tự Unicode (⚠️, tiếng Việt có dấu) — do Windows console mặc định dùng codepage `cp1252`, không hỗ trợ UTF-8/emoji.
   - **Fix:** Set biến môi trường `PYTHONIOENCODING=utf-8` trước khi chạy script (`PYTHONIOENCODING=utf-8 python main.py`).

2. **Lỗi:** `Error response from daemon: ... Bind for 0.0.0.0:6333 failed: port is already allocated` khi `docker compose up -d`.
   - **Debug:** `docker ps` cho thấy port 6333 đã bị một Qdrant container của project khác trên máy chiếm dụng.
   - **Fix:** Đổi port mapping trong `docker-compose.yml` (6335:6333, 6336:6334) và cập nhật `QDRANT_PORT` trong `config.py` tương ứng, tránh xung đột với project khác.

3. **Kiến thức thiếu:** Ban đầu chưa rõ tại sao BM25 + underthesea segmentation phải `replace("_", " ")` sau khi tokenize.
   - **Cách bổ sung:** Đọc kỹ comment trong scaffold — underthesea nối từ ghép bằng `_` (VD: "nghỉ_phép"), nếu không replace thì BM25 tokenize (split theo space) sẽ coi "nghỉ_phép" là 1 token trong khi query "nghỉ phép" (gõ tự nhiên, có dấu cách) thành 2 token → không match. Hiểu ra đây là vấn đề tokenizer mismatch giữa lúc index và lúc query, một lỗi rất dễ mắc phải khi làm hybrid search cho tiếng Việt.

4. **Môi trường:** `pip install -r requirements.txt` fail với Python 3.13 (không có wheel sẵn cho numpy trên phiên bản Python quá mới, phải build từ source và thiếu compiler trên máy Windows).
   - **Fix:** Cài Python 3.11 riêng (đúng theo `.python-version`), tạo virtualenv `.venv` bằng `py -3.11 -m venv .venv` rồi cài lại — tất cả wheel đều có sẵn cho 3.11.

5. **Lỗi:** `OSError: The paging file is too small for this operation to complete. (os error 1455)` khi load CrossEncoder reranker (bge-reranker-v2-m3) ngay sau bước indexing (bge-m3 dense encoder).
   - **Debug:** Máy có 16GB RAM nhưng Docker Desktop (WSL2 VM) đang giữ ~7.6GB cho các container của project khác đang chạy song song (backend, DB, Redis, Qdrant khác) — không đủ RAM/pagefile để load cả 2 model transformer cùng lúc.
   - **Fix:** Tạm dừng các Docker container không liên quan (`docker stop`) trong lúc chạy pipeline, chạy xong bật lại (`docker start`). Bài học: khi benchmark pipeline nặng về model (nhiều embedding model tải cùng lúc), cần kiểm tra tài nguyên hệ thống đang bị chiếm dụng bởi service khác trước khi debug code.

6. **Trade-off khi tuning faithfulness:** Cố gắng đẩy faithfulness từ 0.75 lên ≥0.85 (ngưỡng bonus) bằng cách siết prompt + tăng top_k, đạt 0.8431-0.8464 nhưng không vượt ngưỡng ổn định qua nhiều lần chạy.
   - **Bài học:** RAGAS metrics dùng LLM-judge nên có nhiễu ngẫu nhiên (~±0.02-0.05) — không nên "chase" một ngưỡng cụ thể qua nhiều lần chạy tốn API cost mà nên chấp nhận khoảng dao động và báo cáo trung thực xu hướng cải thiện thay vì con số tuyệt đối của 1 lần chạy duy nhất. Cũng nhận ra improvement ở 1 metric (faithfulness +0.09) thường đánh đổi với metric khác (context_precision -0.036) — production RAG luôn là bài toán trade-off, không có "improve tất cả cùng lúc" miễn phí.

## Phần 3: Action Plan cho project

## Project: Hệ thống hỏi-đáp nội bộ dựa trên tài liệu công ty (RAG chatbot)

### Hiện tại
- RAG pipeline hiện tại: Prototype dùng basic paragraph chunking + dense-only search (giống `naive_baseline.py` của lab), chưa có reranking hay evaluation có hệ thống.
- Known issues: Retrieval miss thông tin khi câu hỏi cần tổng hợp nhiều đoạn (multi-hop); tài liệu có version cũ/mới gây trả lời sai (model không phân biệt được policy nào đang hiệu lực); chưa có cách đo lường chất lượng câu trả lời khách quan ngoài việc đọc thủ công.

### Plan áp dụng
1. [ ] Chunking strategy: Chuyển sang **hierarchical (parent-child)** — index child (256-512 ký tự) để retrieve chính xác, nhưng trả về parent (1500-2000 ký tự) cho LLM để có đủ ngữ cảnh trả lời — giải quyết vấn đề multi-hop tốt hơn so với chunk cố định hiện tại.
2. [ ] Search: Chuyển sang **hybrid BM25 + Dense với RRF**, vì tài liệu công ty có nhiều thuật ngữ/tên riêng (mã dự án, tên phòng ban) mà dense embedding một mình dễ miss — BM25 bù được phần exact-match này.
3. [ ] Reranking: **Có**, dùng cross-encoder (bge-reranker-v2-m3) top-20 → top-3/5, vì lab cho thấy answer_relevancy cải thiện rõ khi có reranker lọc chunk nhiễu.
4. [ ] Evaluation: Dùng **RAGAS 4 metrics** làm baseline định kỳ (chạy trên bộ test set nội bộ ~30-50 câu hỏi thật), kèm failure analysis theo Diagnostic Tree để biết lỗi rơi vào nhóm nào (retrieval vs generation) trước khi optimize mù quáng.
5. [ ] Enrichment: Ưu tiên **contextual prepend** + **metadata versioning** (đánh dấu tài liệu current/superseded) — vì lỗi lớn nhất quan sát được trong lab là model trộn lẫn chính sách cũ/mới, chứ không phải model tự bịa thông tin.

### Timeline
- Tuần 1: Refactor chunking sang hierarchical, thêm metadata `effective_date`/`status` cho mọi tài liệu khi ingest.
- Tuần 2: Triển khai hybrid search (BM25 tiếng Việt + dense) + RRF, benchmark lại context_precision/recall trên test set nội bộ.
- Tuần 3: Thêm cross-encoder reranker, đo latency để đảm bảo vẫn đáp ứng SLA phản hồi (<3s).
- Tuần 4: Thiết lập RAGAS evaluation pipeline chạy tự động (CI hoặc cron hàng tuần) + dashboard theo dõi 4 metrics theo thời gian, tránh regression khi thêm tài liệu mới.
