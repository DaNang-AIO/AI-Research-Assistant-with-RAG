"""DocumentLoader (design.md §2.3) — kế thừa `src.interfaces.BaseLoader`.

Triển khai: S1-DE-01 (txt/md, load_directory) và S1-DE-02 (PDF + xử lý lỗi).
"""

import os
import hashlib
from typing import List

from src.interfaces import BaseLoader
from src.models import Document, DocumentType

class PDFLoader:
    """Bộ nạp dữ liệu chuyên xử lý file PDF bằng PyMuPDF"""
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> str:
        try:
            import fitz  # PyMuPDF — lazy import để không fail khi module chưa cài
        except ImportError as e:
            raise ImportError(
                "PyMuPDF (fitz) chưa được cài — chạy: pip install PyMuPDF"
            ) from e
        text = ""
        # Sử dụng context manager (with) để tự động đóng file sau khi đọc xong
        with fitz.open(self.file_path) as doc:
            for page in doc:
                text += page.get_text() + "\n"
        return text.strip()

class TEXTLoader:
    """Bộ nạp dữ liệu chuyên xử lý file văn bản thuần túy"""
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> str:
        with open(self.file_path, "r", encoding="utf-8") as f:
            return f.read()

SUPPORTED_LOADERS = {
    '.pdf': PDFLoader,
    '.txt': TEXTLoader,
    '.md': TEXTLoader
}

class DocumentLoader(BaseLoader):
    """
    Tải tài liệu từ file và trả về Document object.
    Hỗ trợ: PDF (PyMuPDF), TXT, Markdown.
    """

    def __init__(self):
        """Khởi tạo DocumentLoader (không cần argument)."""
        pass

    def load(self, file_path: str) -> Document:
        """Tải file, tự động chọn loader phù hợp theo extension"""
        if not self.supports(file_path):
            ext = self._get_extension(file_path)
            raise ValueError(f"Định dạng file {ext} không được hỗ trợ")

        ext = self._get_extension(file_path)
        loader_class = SUPPORTED_LOADERS.get(ext)
        if loader_class is None:
            raise ValueError(f"Không tìm thấy loader cho extension {ext}")

        loader_instance = loader_class(file_path)
        content = loader_instance.load()

        doc_typemap = {
            ".txt": DocumentType.TXT,
            ".md": DocumentType.MARKDOWN,
            ".pdf": DocumentType.PDF,
            ".html": DocumentType.HTML,
        }
        doc_type = doc_typemap.get(ext, DocumentType.TXT)

        return Document(
            doc_id=self._generate_doc_id(file_path),
            file_path=file_path,
            doc_type=doc_type,
            content=content,
        )

    def load_directory(self, dir_path: str) -> List[Document]:
        """Tải tất cả tài liệu hỗ trợ trong một thư mục"""
        documents: List[Document] = []
        for root, _, files in os.walk(dir_path):
            for name in files:
                file_path = os.path.join(root, name)
                if self.supports(file_path):
                    documents.append(self.load(file_path))
        return documents

    def supports(self, file_path: str) -> bool:
        """Kiểm tra có loader cho loại file này không"""
        ext = self._get_extension(file_path)
        return ext in SUPPORTED_LOADERS

    def _get_extension(self, file_path: str) -> str:
        """Trích xuất extension từ đường dẫn"""
        return os.path.splitext(file_path)[1].lower()

    def _generate_doc_id(self, file_path: str) -> str:
        """Tạo doc_id duy nhất (hash của đường dẫn tuyệt đối)"""
        abs_path = os.path.abspath(file_path)
        return hashlib.sha256(abs_path.encode()).hexdigest()