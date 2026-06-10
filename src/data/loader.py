"""DocumentLoader (design.md §2.3) — kế thừa `src.interfaces.BaseLoader`.

Triển khai: S1-DE-01 (txt/md, load_directory) và S1-DE-02 (PDF + xử lý lỗi).
"""

import os
import hashlib
from src.interfaces import BaseLoader
from src.models import Document, DocumentType
from typing import Dict, List

class DocumentLoader(BaseLoader):
    """
    Tải tài liệu từ file và trả về Document object.
    Hỗ trợ: PDF (PyMuPDF), TXT, Markdown.
    """
    def __init__(self):
        self._loaders: Dict[str, BaseLoader] = {}  # extension -> loader
        self._loaders = {
            '.txt': self._read_text_file,
            '.md': self._read_text_file
        }

    def load(self, file_path: str) -> Document:
        """Tải file, tự động chọn loader phù hợp theo extension"""
        if not self.supports(file_path):
            raise ValueError(f"Định dạng file không được hỗ trợ")
        content = self._loaders(file_path)
        doc_typemap = {
            ".txt": DocumentType.TXT,
            ".md": DocumentType.MARKDOWN,
            ".pdf": DocumentType.PDF,
            ".html": DocumentType.HTML
        }
        ext = self._get_extension(file_path)
        doc_type = doc_typemap[ext]
        return Document(
            doc_id = self._generate_doc_id(file_path),
            file_path = file_path,
            doc_type = doc_type,
            content = content
        )       
            

    def load_directory(self, dir_path: str) -> List[Document]:
        """Tải tất cả tài liệu hỗ trợ trong một thư mục"""
        documents = []
        for root, _, files in os.walk(dir_path):
            for name in files:
                file_path = os.path.join(root,name)
                if self.supports(file_path):
                    documents.append(self.load(file_path))
        return documents


    def supports(self, file_path: str) -> bool:
        """Kiểm tra có loader cho loại file này không"""
        ext = self._get_extension(file_path)
        return True if ext in self._loaders else False

    def _get_extension(self, file_path: str) -> str:
        """Trích xuất extension từ đường dẫn"""
        return os.path.splitext(file_path)[1].lower()

    def _generate_doc_id(self, file_path: str) -> str:
        """Tạo doc_id duy nhất (hash của đường dẫn tuyệt đối)"""
        abs_path = os.path.abspath(file_path)
        return hashlib.sha256(abs_path.encode()).hexdigest()
    
    def _read_text_file(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
        