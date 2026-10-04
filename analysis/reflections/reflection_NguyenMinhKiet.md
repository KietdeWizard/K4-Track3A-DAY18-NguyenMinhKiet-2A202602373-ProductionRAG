# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Nguyen Minh Kiet  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 2026-10-04

## Phần 1: Lecture Mapping

| Lecture Concept | Module | Hàm cụ thể | Observation |
|---|---|---|---|
| Semantic chunking | M1 | `chunk_semantic()` | So sánh cosine giữa các câu và tạo boundary khi similarity thấp hơn threshold 0.85. |
| Parent-child chunking | M1 | `chunk_hierarchical()` | Child tăng precision khi tìm kiếm; parent giữ context khi trả lời. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | Header Markdown được giữ trong chunk và lưu vào metadata `section`. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | BM25 bắt số hiệu/số liệu; dense bắt tương đồng ngữ nghĩa; RRF gộp theo rank. |
| Vietnamese tokenization | M2 | `segment_vietnamese()` | Thay `_` bằng khoảng trắng để query “nghỉ phép” khớp đúng tài liệu. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Chấm lại top candidates bằng cặp query-document và lấy top 3. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Đo faithfulness, answer relevancy, context precision và context recall. |
| Contextual enrichment | M5 | `_enrich_single_call()` | Một API call có thể tạo context, summary, câu hỏi giả định và metadata. |

## Phần 2: Challenges & Debugging

- **Dependency/runtime:** Qdrant import yêu cầu thêm protobuf và grpcio; xử lý bằng cách cài đúng dependency và giữ in-memory fallback khi không có Docker.
- **API compatibility:** Qdrant client mới dùng `query_points()` thay cho `search()`; code đã dùng API mới.
- **Model availability:** Khi transformer chưa tải được, pipeline dùng encoder deterministic fallback và reranker lexical fallback để vẫn chạy được.
- **PDF scan:** Hai PDF scan không có text layer nên bị bỏ qua; cần OCR nếu muốn đưa chúng vào corpus.
- **Evaluation:** Khi RAGAS không import được vì thiếu dependency, pipeline dùng fallback overlap metric và ghi rõ lỗi trong log. Môi trường nộp bài cần cài đầy đủ requirements để nhận điểm RAGAS chính thức.

## Phần 3: Action Plan

### Project: Vietnamese Enterprise Knowledge Assistant

#### Hiện tại

- Pipeline hiện tại dùng chunking cơ bản và dense search.
- Bottleneck là câu hỏi nhiều bước, tài liệu có phiên bản cũ/mới và truy xuất bảng biểu.

#### Kế hoạch cải tiến

1. Dùng hierarchical chunking cho policy và structure-aware chunking cho Markdown/table.
2. Dùng hybrid BM25 + dense với RRF để giữ số hiệu, ngày tháng và thuật ngữ chính xác.
3. Dùng `BAAI/bge-reranker-v2-m3` cho top 20 candidate và chỉ gửi top 3 cho LLM.
4. Dùng RAGAS cùng bộ câu hỏi có ground truth; theo dõi riêng version-conflict và negation cases.
5. Dùng contextual prepend trước; chỉ bật HyQA cho các nhóm tài liệu có vocabulary gap lớn.

#### Timeline

- Tuần 1: Chuẩn hóa corpus, metadata và version/effective-date.
- Tuần 2: Triển khai hierarchical chunking, hybrid retrieval và reranking.
- Tuần 3: Tạo evaluation set, chạy RAGAS và phân tích bottom failures.
- Tuần 4: Tối ưu latency, chi phí enrichment và triển khai monitoring.
