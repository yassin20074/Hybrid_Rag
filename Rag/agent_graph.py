from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from Rag.retriever import HybridReRankRetriever

# 1. تعريف الـ State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: str

class RAGAgentGraph:
    def __init__(self, retriever: HybridReRankRetriever, groq_api_key: str):
        self.retriever = retriever
        self.llm = ChatGroq(
            api_key=groq_api_key, 
            model_name="llama-3.3-70b-versatile",
            temperature=0.2
        )
        self.memory = MemorySaver() # Checkpointer لحفظ المحادثة
        self.graph = self._build_graph()

    def retrieve_node(self, state: AgentState):
        """عقدة جلب المعلومات المترابطة بالبحث الهجين"""
        latest_user_message = state["messages"][-1].content
        docs = self.retriever.get_relevant_documents(latest_user_message)
        context_text = "\n\n".join([d.page_content for d in docs])
        return {"context": context_text}

    def generate_node(self, state: AgentState):
        """عقدة توليد الرد باستعمال الـ LLM والـ Context والـ History"""
        context = state.get("context", "")
        system_prompt = (
            f"You are an expert AI assistant. Answer the user prompt using ONLY the provided context.\n"
            f"If you don't know, say so clearly.\n\nContext:\n{context}"
        )
        
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]
        response = self.llm.invoke(messages)
        return {"messages": [response]}

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("generate", self.generate_node)
        
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        # ربط الـ Graph بالـ MemorySaver (Checkpointer)
        return workflow.compile(checkpointer=self.memory)
