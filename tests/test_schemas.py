"""Unit tests for Pydantic schema validation."""

import pytest
from pydantic import ValidationError
from src.agent.schemas import Citation, FinancialMetric, AuditedFinancialResponse


def test_valid_citation():
    cit = Citation(
        document_name="apple_10k_2023.pdf",
        page_number=12,
        exact_quote="Total net sales decreased 3% during 2023 compared to 2022.",
        relevance_explanation="Supports revenue decline claim."
    )
    assert cit.document_name == "apple_10k_2023.pdf"
    assert cit.page_number == 12


def test_confidence_score_range_validation():
    # Valid confidence
    res = AuditedFinancialResponse(
        executive_summary="Summary",
        detailed_analysis="Analysis",
        citations=[],
        confidence_score=0.95
    )
    assert res.confidence_score == 0.95

    # Out of bounds (> 1.0)
    with pytest.raises(ValidationError):
        AuditedFinancialResponse(
            executive_summary="Summary",
            detailed_analysis="Analysis",
            citations=[],
            confidence_score=1.5
        )
