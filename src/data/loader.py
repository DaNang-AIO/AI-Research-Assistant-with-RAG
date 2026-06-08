"""DocumentLoader (design.md §2.3) — kế thừa `src.interfaces.BaseLoader`.

Triển khai: S1-DE-01 (txt/md, load_directory) và S1-DE-02 (PDF + xử lý lỗi).
"""

import hashlib
import os
from typing import Callable, Dict, List

from src.interfaces import BaseLoader
from src.models import Document, DocumentType

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - PyMuPDF nằm trong requirements.txt
    fitz = None


_EXTENSION_TO_DOC_TYPE = {
    ".txt": DocumentType.TXT,
    ".md": DocumentType.MARKDOWN,
    ".markdown": DocumentType.MARKDOWN,
    ".pdf": DocumentType.PDF,
}


class DocumentLoader(BaseLoader):
    """Tải tài liệu từ file và trả về Document object.

    Hỗ trợ: PDF (PyMuPDF), TXT, Markdown.
    """

    def __init__(self):
        # extension -> hàm trích xuất nội dung text tương ứng
        self._loaders: Dict[str, Callable[[str], str]] = {
            ".txt": self._load_text,
            ".md": self._load_text,
            ".markdown": self._load_text,
            ".pdf": self._load_pdf,
        }

    def load(self, file_path: str) -> Document:
        """Tải file, tự động chọn loader phù hợp theo extension."""
        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                f"Không tìm thấy file hoặc không thể đọc: '{file_path}'"
            )

        extension = self._get_extension(file_path)
        load_content = self._loaders.get(extension)
        if load_content is None:
            raise ValueError(
                f"Định dạng file không được hỗ trợ: '{extension}' "
                f"(chỉ hỗ trợ: {', '.join(sorted(self._loaders))})"
            )

        content = load_content(file_path)
        return Document(
            doc_id=self._generate_doc_id(file_path),
            file_path=file_path,
            doc_type=_EXTENSION_TO_DOC_TYPE[extension],
            content=content,
            metadata={"source": os.path.basename(file_path)},
        )

    def load_directory(self, dir_path: str) -> List[Document]:
        """Tải tất cả tài liệu hỗ trợ trong một thư mục."""
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"Không tìm thấy thư mục: '{dir_path}'")

        documents = []
        for entry in sorted(os.listdir(dir_path)):
            full_path = os.path.join(dir_path, entry)
            if os.path.isfile(full_path) and self.supports(full_path):
                documents.append(self.load(full_path))
        return documents

    def supports(self, file_path: str) -> bool:
        """Kiểm tra có loader cho loại file này không."""
        return self._get_extension(file_path) in self._loaders

    def _get_extension(self, file_path: str) -> str:
        """Trích xuất extension (chữ thường, kèm dấu chấm) từ đường dẫn."""
        return os.path.splitext(file_path)[1].lower()

    def _generate_doc_id(self, file_path: str) -> str:
        """Tạo doc_id duy nhất bằng hash SHA-256 của đường dẫn tuyệt đối."""
        absolute_path = os.path.abspath(file_path)
        return hashlib.sha256(absolute_path.encode("utf-8")).hexdigest()[:16]

    def _load_text(self, file_path: str) -> str:
        """Đọc nội dung file văn bản thuần (.txt, .md) dưới dạng UTF-8."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _load_pdf(self, file_path: str) -> str:
        """Trích xuất nội dung text từ file PDF bằng PyMuPDF."""
        if fitz is None:
            raise RuntimeError(
                "Thư viện PyMuPDF (fitz) chưa được cài đặt — không thể tải file PDF. "
                "Hãy chạy: pip install PyMuPDF"
            )
        try:
            with fitz.open(file_path) as pdf_document:
                return "\n".join(page.get_text() for page in pdf_document)
        except Exception as exc:
            raise ValueError(f"Không thể đọc file PDF '{file_path}': {exc}") from exc
