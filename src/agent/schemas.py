"""Pydantic schemas for audited financial outputs and query interfaces."""

from typing import List, Optional
from pydantic import BaseModel, Field

class Citation(BaseModel):
    """Specific citation evidence from source documents."""
    document_name: str = Field(description="Name or file path of the source document")
    page_number: int = Field(description="Page number where evidence is found (1-indexed)")
    exact_quote: str = Field(description="Direct verbatim quote or markdown table excerpt from the document")
    relevance_explanation: Optional[str] = Field(default=None, description="Why this evidence supports the claim")

class FinancialMetric(BaseModel):
    """Quantitative metric extracted from financial statements."""
    metric_name: str = Field(description="e.g. Total Revenue, Net Income, Gross Margin")
    value: str = Field(description="Extracted value with unit (e.g., '$383,285 million', '28.5%')")
    fiscal_period: str = Field(description="Fiscal year or quarter, e.g. 'FY 2023', 'Q3 2023'")
    company: str = Field(description="Company associated with the metric, e.g. 'Apple Inc.'")

class AuditedFinancialResponse(BaseModel):
    """Enforces audited, verifiable output with required source citations."""
    executive_summary: str = Field(
        description="Concise, high-level answer to the analyst query."
    )
    key_metrics: List[FinancialMetric] = Field(
        default_factory=list,
        description="Key numerical figures and financial indicators extracted from the text/tables."
    )
    detailed_analysis: str = Field(
        description="Comprehensive breakdown and comparison, synthesizing the evidence."
    )
    citations: List[Citation] = Field(
        description="Strict, verifiable citations with page numbers and exact quotes supporting the statements."
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="Self-assessed confidence level based on retrieved context clarity and completeness."
    )

class QueryRequest(BaseModel):
    """User query request body for API."""
    query: str = Field(..., json_schema_extra={"example": "Compare Apple's net sales in 2022 vs 2023 and highlight key growth drivers."})
    top_k: Optional[int] = Field(default=4, description="Number of context chunks to use after reranking.")

class QueryResponse(BaseModel):
    """Wrapper response including audited result and processing time."""
    query: str
    result: AuditedFinancialResponse
    retrieved_chunks_count: int
    reranked_chunks_count: int
