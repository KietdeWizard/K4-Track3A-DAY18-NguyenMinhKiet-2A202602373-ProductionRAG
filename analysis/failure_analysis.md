# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Nguyen Minh Kiet
**Khóa:** K4 - Track 3A  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 1.0000 | 1.0000 | +0.0000 |
| Answer Relevancy | 0.6541 | 0.6938 | +0.0397 |
| Context Precision | 1.0000 | 1.0000 | +0.0000 |
| Context Recall | 0.7859 | 0.7413 | -0.0446 |

## Bottom-5 Failures

### #1
- **Question:** Laptop 30 triệu cho nhân viên mới: ai phê duyệt và cần gì từ CNTT?
- **Expected:** Director phê duyệt; cần xác nhận cấu hình CNTT và ít nhất 3 báo giá.
- **Got:** Context lấy nhầm chính sách hoàn chi đào tạo.
- **Worst metric:** context_recall
- **Error Tree:** Output sai → Context chưa đủ → Query rõ → lỗi retrieval/chunk ranking.
- **Root cause:** Candidate chứa thông tin chi phí nhưng thiếu đoạn mua sắm liên quan.
- **Suggested fix:** Tăng BM25 weight cho số tiền và thuật ngữ laptop/CNTT; rerank lại top-20.

### #2
- **Question:** Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày và lương bao nhiêu?
- **Expected:** 18 ngày phép; lương Senior 20–35 triệu/tháng.
- **Got:** Context chủ yếu là nghỉ phép đặc biệt, thiếu bảng lương.
- **Worst metric:** context_recall
- **Error Tree:** Output sai → Context thiếu bảng lương → Query cần multi-hop → lỗi retrieval.
- **Root cause:** Một câu hỏi yêu cầu ghép hai tài liệu nhưng chỉ lấy được một nguồn.
- **Suggested fix:** Multi-query retrieval và bắt buộc giữ ít nhất một chunk từ mỗi nguồn liên quan.

### #3
- **Question:** Lương thử việc của Junior mức cao nhất là bao nhiêu?
- **Expected:** 85% × 20.000.000 = 17.000.000 VNĐ/tháng.
- **Got:** Context có tỷ lệ thử việc nhưng chưa có bảng lương Junior.
- **Worst metric:** answer_relevancy
- **Error Tree:** Output chưa trực tiếp → Context một phần đúng → Query rõ → thiếu dữ liệu tính toán.
- **Root cause:** Retrieval không đưa đồng thời chính sách thử việc và bảng lương.
- **Suggested fix:** Detect câu hỏi tính toán và truy xuất thêm tài liệu bảng lương.

### #4
- **Question:** Mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Tổng Giám đốc (CEO).
- **Got:** Context có quy trình mua sắm nhưng câu trả lời bị dài và không chốt rõ CEO.
- **Worst metric:** answer_relevancy
- **Error Tree:** Output lan man → Context đúng một phần → Query rõ → lỗi answer prompt.
- **Root cause:** Prompt chưa buộc trả lời một kết luận ngắn cho câu hỏi định lượng.
- **Suggested fix:** Thêm format answer: kết luận trước, lý do sau, không đưa chính sách không liên quan.

### #5
- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** CEO phê duyệt vì thuộc khoảng 16–30 ngày.
- **Got:** Context có quy trình nhưng bị cắt giữa bảng phân quyền.
- **Worst metric:** context_recall
- **Error Tree:** Output có thể sai → Context thiếu dòng cuối → Query rõ → lỗi child chunk boundary.
- **Root cause:** Child chunk 256 ký tự làm tách rời quy tắc phê duyệt.
- **Suggested fix:** Dùng parent context khi trả lời và giữ nguyên bảng/list trong structure-aware chunks.

## Case Study (cho presentation)

**Question chọn phân tích:**

**Error Tree walkthrough:**
1. Output đúng? →
2. Context đúng? →
3. Query rewrite OK? →
4. Fix ở bước:

**Nếu có thêm 1 giờ, sẽ optimize:**
- Tải đủ model thật, chạy RAGAS với API key, sau đó so sánh lại heuristic fallback với điểm RAGAS chính thức.
