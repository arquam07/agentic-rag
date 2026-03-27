"""
Query Router — FastAPI endpoints for asking questions.
"""

import logging
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from app.agent.graph import run_agent

logger = logging.getLogger(__name__)

query_router = APIRouter()


#  Request / Response schemas 
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask")

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    num_chunks_used: int


# Endpoint 
@query_router.post("/", response_model=QueryResponse)
async def ask_question(req: QueryRequest):
    """
    Ask a question against your ingested documents.
    """
    try:
        result = await run_agent(req.question)
        return QueryResponse(**result)

    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))