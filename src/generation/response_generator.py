"""ResponseGenerator (design.md §1.2) — backlog mở rộng (stretch goal).

design.md liệt kê file này trong cấu trúc thư mục nhưng không đặc tả
function signature riêng — logic sinh câu trả lời hiện được điều phối trực
tiếp trong `RAGPipeline.query()` (S3-PE-03). Tách ra thành class riêng nếu
nhóm cần thực nghiệm nhiều chiến lược sinh câu trả lời khác nhau.
"""
