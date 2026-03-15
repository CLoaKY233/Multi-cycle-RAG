from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.core import SimpleDirectoryReader

from src.core.exceptions import RAGException
from src.core.interfaces import Document


class DocumentLoader:
    """Document loader with async support"""

    def __init__(self):
        self.supported_extensions = {".txt", ".md", ".pdf", ".docs", ".html"}

    async def load_from_directory(self, directory_path: str) -> List[Document]:
        """Load documents from directory asynchronously"""
        try:
            path = Path(directory_path)
            if not path.exists():
                raise RAGException(f"Directory not found: {directory_path}")
            reader = SimpleDirectoryReader(
                input_dir=str(path),
                recursive=True,
                required_exts=list(self.supported_extensions),
            )

            # Load Documents
            docs = reader.load_data()

            # Convert to our document format
            documents = []
            for doc in docs:
                doc = Document(
                    content=doc.text,
                    metadata={
                        "source": str(doc.metadata.get("file_path", "")),
                        "file_name": doc.metadata.get("file_name", ""),
                        "file_type": doc.metadata.get("file_type", ""),
                        **doc.metadata,
                    },
                )
                documents.append(doc)
            return documents
        except Exception as e:
            raise RAGException(f"Failed to Load Documents: {e}")

    async def load_from_text(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """Load a single document from text"""
        return Document(content=text, metadata=metadata or {})
