# Failure Analysis — Lab 18: Production RAG

**Thực hiện:** Vũ Tú Quỳnh

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.7542 | 0.8431 | +0.0889 |
| Answer Relevancy | 0.7234 | 0.8151 | +0.0917 |
| Context Precision | 0.9250 | 0.8894 | -0.0356 |
| Context Recall | 0.9250 | 0.9000 | -0.0250 |

Production pipeline (hierarchical chunking + M5 enrichment + hybrid search + rerank + generation
prompt siết chặt, temperature=0) cải thiện rõ rệt faithfulness và answer relevancy so với naive
baseline. Context precision/recall giảm nhẹ — đánh đổi khi tăng `RERANK_TOP_K` từ 3 lên 5 để có
đủ ngữ cảnh cho câu hỏi multi-hop (giúp faithfulness tăng mạnh) nhưng đưa thêm vài chunk ít liên
quan hơn vào context.

> **Ghi chú:** Ban đầu (prompt gốc, top_k=3) production faithfulness chỉ đạt 0.75, ngang naive.
> Sau khi (1) thêm chỉ dẫn ưu tiên phiên bản chính sách hiện hành khi có mâu thuẫn, (2) yêu cầu chỉ
> nêu claim có bằng chứng trực tiếp trong context, (3) đặt `temperature=0`, và (4) tăng top_k lên 5
> để đủ ngữ cảnh cho câu hỏi multi-hop, faithfulness tăng lên 0.8431–0.8464 qua các lần chạy lại
> (RAGAS dùng LLM-judge nên có dao động ~±0.02 giữa các lần chạy).

## Bottom-5 Failures

### #1
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Thời hạn thanh toán 15 ngày. Quá hạn 5 ngày, phí 2%/tháng trên 15tr = 300k/tháng (pro-rata ~50k cho 5 ngày).
- **Got:** Faithfulness = 0.111 — vẫn là lỗi tính toán, dù đã yêu cầu "chỉ dùng đúng số liệu trong context và nêu công thức".
- **Worst metric:** Faithfulness (0.11)
- **Error Tree:** Output sai → Context đúng? Đúng, chunk chứa đủ rule (15 ngày hạn, 2%/tháng) → Query OK? Có → nhưng câu hỏi yêu cầu tính toán số học nhiều bước (20-15=5 ngày quá hạn → pro-rata 5/30 ngày × 2% × 15tr) → LLM generation vẫn tính sai dù retrieval đúng 100%.
- **Root cause:** Loại lỗi "numeric reasoning" — retrieval không phải vấn đề, model yếu về tính toán nhiều bước dù đã ràng buộc prompt.
- **Suggested fix:** Thêm few-shot example minh họa phép tính pro-rata trong prompt; hoặc tách bước tính toán ra khỏi LLM (dùng code/calculator tool thay vì để LLM tự tính).

### #2
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Theo chính sách hiện hành (v2.0), mật khẩu phải được thay đổi mỗi 120 ngày. Chính sách cũ yêu cầu 90 ngày nhưng đã bị thay thế.
- **Got:** Faithfulness = 0.0 — vẫn lỗi dù đã thêm chỉ dẫn "ưu tiên phiên bản hiện hành khi có mâu thuẫn".
- **Worst metric:** Faithfulness (0.0)
- **Error Tree:** Output sai → Context đúng? Retrieval vẫn trả về CẢ 2 version (v1 90 ngày và v2 120 ngày) → Query OK? Có → Prompt instruction chỉ dựa vào suy luận ngôn ngữ ("mới nhất/hiện hành") nhưng context không có metadata rõ ràng đánh dấu version nào current — model không đủ tín hiệu để chọn đúng.
- **Root cause:** Đây là giới hạn của prompt-engineering thuần túy: không có metadata `status: current/superseded` gắn vào chunk lúc index, nên dù prompt yêu cầu ưu tiên bản mới, model vẫn phải đoán dựa trên nội dung — không đáng tin cậy 100%.
- **Suggested fix:** Bổ sung metadata filtering ở tầng retrieval (M5 auto_metadata / M2 payload filter) để loại chunk `superseded` TRƯỚC khi đưa vào context, thay vì dựa vào prompt để LLM tự lọc.

### #3
- **Question:** Thông tin lương thuộc cấp độ phân loại dữ liệu nào?
- **Expected:** (câu hỏi lookup đơn giản về data classification)
- **Got:** Answer relevancy = 0.0 — câu trả lời không match câu hỏi.
- **Worst metric:** Answer Relevancy (0.0)
- **Error Tree:** Output sai → Context đúng? Context có thể chứa thông tin nhưng answer bị lệch chủ đề (có thể do prompt mới quá strict khiến model trả "Không tìm thấy" hoặc trả lời sai hướng) → Query OK? Query rõ ràng → vấn đề nằm ở generation không bám sát câu hỏi.
- **Root cause:** Answer relevancy đo bằng cách LLM sinh lại câu hỏi giả định từ answer rồi so với câu hỏi gốc — nếu answer quá ngắn/không đủ ngữ cảnh (do prompt yêu cầu "ngắn gọn, không thêm câu thừa"), embedding của câu hỏi tái tạo có thể lệch xa câu hỏi gốc.
- **Suggested fix:** Cân bằng lại constraint "ngắn gọn" — không nên cắt bỏ ngữ cảnh cần thiết để câu trả lời vẫn giữ đủ từ khóa liên quan đến câu hỏi.

### #4
- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** (câu hỏi lookup ngưỡng phê duyệt theo số ngày nghỉ không lương)
- **Got:** Context recall = 0.5 — chỉ retrieve được một phần thông tin cần thiết.
- **Worst metric:** Context Recall (0.5)
- **Error Tree:** Output sai → Context đúng? Một phần — retrieve được chunk liên quan đến "nghỉ không lương" nhưng thiếu bảng ngưỡng phê duyệt theo số ngày cụ thể → Query OK? Có → đây là retrieval failure do structure-aware/hierarchical chunking có thể tách bảng ngưỡng phê duyệt ra khỏi ngữ cảnh chính.
- **Root cause:** Numeric threshold lookup (tương tự các câu hỏi "phê duyệt theo mức tiền") — thông tin dạng bảng dễ bị cắt rời khỏi phần diễn giải khi chunking.
- **Suggested fix:** Đảm bảo bảng (table) không bị chunking cắt đứt khỏi tiêu đề/ngữ cảnh; cân nhắc để nguyên bảng threshold thành 1 chunk riêng có `chunk_type: table`.

### #5
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** Context recall = 0.5 — cải thiện so với lần chạy đầu (context_recall=0.0 khi top_k=3), nhờ tăng RERANK_TOP_K lên 5 giúp giữ được nhiều hơn cả 2 nguồn (leave + salary), nhưng vẫn chưa đủ.
- **Worst metric:** Context Recall (0.5)
- **Error Tree:** Output sai → Context đúng? Một phần — multi-hop cần 2 chunk từ 2 tài liệu khác nhau (nghỉ phép + bảng lương), top-5 sau rerank đôi khi vẫn ưu tiên các chunk tương tự nhau hơn là đa dạng nguồn → Query OK? Có, nhưng single dense/BM25 query không tách được 2 sub-intent (leave computation + salary lookup).
- **Root cause:** Multi-hop reasoning đòi hỏi kết hợp nhiều nguồn — 1 lần retrieve/rerank duy nhất không đảm bảo đa dạng hóa đủ nguồn thông tin.
- **Suggested fix:** Query decomposition — tách câu hỏi thành 2 sub-query ("nghỉ phép năm 9 năm thâm niên" + "lương Senior") rồi retrieve riêng, merge context trước khi generate.

## Case Study (cho presentation)

**Question chọn phân tích:** "Bao lâu phải đổi mật khẩu một lần?" (#2 — versioning conflict)

**Error Tree walkthrough:**
1. Output đúng? → Vẫn sai sau khi đã sửa prompt (Faithfulness = 0.0), cho thấy đây không phải lỗi có thể fix chỉ bằng prompt engineering.
2. Context đúng? → Retrieval đúng về mặt kỹ thuật (tìm ra cả 2 chunk liên quan) nhưng thiếu tín hiệu phân biệt version.
3. Query rewrite OK? → Không có disambiguation theo effective_date.
4. Fix ở bước: Indexing/metadata — cần gắn `status: current/superseded` vào payload lúc index (M5), rồi filter ở tầng retrieval (M2) trước khi đưa vào context, thay vì kỳ vọng prompt-only fix.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Metadata filtering theo version/effective_date ngay ở tầng retrieval (không chỉ dựa vào prompt).
- Query decomposition cho câu hỏi multi-hop.
- Giữ bảng (table) nguyên vẹn trong 1 chunk khi structure-aware chunking gặp Markdown table.
- Cân bằng lại độ dài câu trả lời — hiện tại prompt "ngắn gọn" đôi khi làm giảm answer_relevancy.
