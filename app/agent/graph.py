"""
LangGraph RAG Agent
A stateful graph that orchestrates: retrieve → generate.

Current flow:
    START → retrieve → generate → END
"""

import logging
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document

from app.retrieval.retriever import hybrid_retrieve
from app.generation.llm import get_llm

logger = logging.getLogger(__name__)


# Graph State
class AgentState(TypedDict):
    question: str                        # user's raw question
    context: list[Document]              # retrieved chunks as LangChain Documents
    answer: str                          # final generated answer
    sources: list[str]                   # source filenames for citation


# Node functions 
def retrieve_node(state: AgentState) -> dict:
    """
    Retrieve relevant chunks using hybrid search.
    Converts them into LangChain Document objects for the LLM.
    """
    question = state["question"]
    logger.info("Retrieving context for: %s", question)

    chunks = hybrid_retrieve(question, top_k=5, dense_fetch=20)

    documents = [
        Document(
            page_content=chunk.text,
            metadata={
                "source_file": chunk.source_file,
                "chunk_index": chunk.chunk_index,
                "score": chunk.score,
            },
        )
        for chunk in chunks
    ]

    sources = list({chunk.source_file for chunk in chunks})

    return {"context": documents, "sources": sources}


def generate_node(state: AgentState) -> dict:
    """
    Generate an answer using the LLM with retrieved context.
    """
    question = state["question"]
    context_docs = state["context"]

    # Build context string from retrieved documents
    if context_docs:
        context_text = "\n\n---\n\n".join(
            f"[Source: {doc.metadata['source_file']}, "
            f"Chunk {doc.metadata['chunk_index']}]\n{doc.page_content}"
            for doc in context_docs
        )
    else:
        context_text = "No relevant context found."

    system_prompt = (
        "You are a helpful assistant that answers questions based on the "
        "provided context. Follow these rules:\n"
        "1. Answer ONLY from the context below. If the context doesn't "
        "   contain enough information, say so honestly.\n"
        "2. Cite which source file(s) your answer comes from.\n"
        "3. Be concise but thorough.\n"
        "4. If the question is ambiguous, state your interpretation.\n"
    )

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"Context:\n{context_text}\n\n"
            f"Question: {question}"
        )),
    ])

    return {"answer": response.content}


# Build the graph 
def build_rag_graph() -> StateGraph:
    """
    Construct and compile the RAG graph.
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    # Connect edges
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# Compiled graph instance (reusable) 
rag_graph = build_rag_graph()


async def run_agent(question: str) -> dict:
    """
    Run the RAG pipeline end-to-end.
    """
    result = await rag_graph.ainvoke({
        "question": question,
        "context": [],
        "answer": "",
        "sources": [],
    })

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "num_chunks_used": len(result["context"]),
    }