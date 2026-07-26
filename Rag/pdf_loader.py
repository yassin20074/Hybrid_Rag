from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentProcessor:
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 120):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_pdf(self, file_path: str) -> list[Document]:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
           
        split_docs = self.text_splitter.split_documents(docs)
        return split_docs
