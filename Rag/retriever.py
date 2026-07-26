from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

class HybridReRankRetriever:
    def __init__(self, documents: list[Document]):
       
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        
        self.vectorstore = Chroma.from_documents(documents, self.embeddings)
        self.dense_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
    
        self.sparse_retriever = BM25Retriever.from_documents(documents)
        self.sparse_retriever.k = 10
         
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever if hasattr(self, 'bm25_retriever') else self.sparse_retriever, self.dense_retriever],
            weights=[0.5, 0.5]
        )
        
    
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def get_relevant_documents(self, query: str, top_k: int = 3) -> list[Document]:
           
        initial_docs = self.ensemble_retriever.invoke(query)
        if not initial_docs:
            return []

  
        pairs = [[query, doc.page_content] for doc in initial_docs]
        scores = self.reranker.predict(pairs)

      
        scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
        
           
        return [doc for doc, score in scored_docs[:top_k]]
