from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

class HybridReRankRetriever:
    def __init__(self, documents: list[Document]):
        # 1. Dense Embeddings Model (خفيف وسريع للـ CPU)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 2. Dense Vector Store (Chroma)
        self.vectorstore = Chroma.from_documents(documents, self.embeddings)
        self.dense_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
        
        # 3. Sparse Retriever (BM25) للكلمات المفتاحية والمصطلحات الدقيقة
        self.sparse_retriever = BM25Retriever.from_documents(documents)
        self.sparse_retriever.k = 10
        
        # 4. Ensemble Retriever (دمج الـ Sparse والـ Dense)
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever if hasattr(self, 'bm25_retriever') else self.sparse_retriever, self.dense_retriever],
            weights=[0.5, 0.5]
        )
        
        # 5. Cross-Encoder للـ Re-ranking النهائي
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def get_relevant_documents(self, query: str, top_k: int = 3) -> list[Document]:
        # جلب المرشحين الأوليين من البحث الهجين (20 مستند)
        initial_docs = self.ensemble_retriever.invoke(query)
        if not initial_docs:
            return []

        # تحضير الأزواج لحساب درجة التشابه الدقيقة (Re-ranking Score)
        pairs = [[query, doc.page_content] for doc in initial_docs]
        scores = self.reranker.predict(pairs)

        # ترتيب المستندات بناءً على أداء الـ Cross-Encoder
        scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
        
        # إرجاع أفضل Top-K مستندات
        return [doc for doc, score in scored_docs[:top_k]]