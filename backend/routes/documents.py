"""
documents.py — Legal document generator API routes
Generates legal documents from templates using the LLM.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import logging

from services.llm import generate_response
from utils.guardrails import build_system_prompt, add_disclaimer
from config import settings

router = APIRouter(prefix="/api/documents", tags=["documents"])
logger = logging.getLogger(__name__)


SUPPORTED_DOCUMENTS = [
    "legal_notice",
    "rental_agreement",
    "nda",
    "complaint_letter",
    "affidavit",
    "demand_letter",
]


class DocumentRequest(BaseModel):
    document_type: str
    details: Dict[str, str]  # Variable fields for the document
    model: Optional[str] = "llama3"

    class Config:
        json_schema_extra = {
            "example": {
                "document_type": "legal_notice",
                "details": {
                    "sender_name": "Rahul Sharma",
                    "sender_address": "123 MG Road, Bengaluru",
                    "recipient_name": "Suresh Kumar",
                    "recipient_address": "456 Anna Salai, Chennai",
                    "subject": "Recovery of outstanding dues",
                    "amount": "₹50,000",
                    "due_date": "15th December 2024",
                },
            }
        }


DOCUMENT_PROMPTS = {
    "legal_notice": """Draft a formal legal notice under Indian law with these details:
{details}
Include: Date, sender details, recipient details, subject, facts, legal basis (cite relevant laws), 
demand, and deadline. Format professionally.""",

    "rental_agreement": """Draft a rental agreement under Indian law with:
{details}
Include all standard clauses: parties, property description, term, rent, security deposit,
maintenance, termination, dispute resolution. Format as a proper legal document.""",

    "nda": """Draft a Non-Disclosure Agreement (NDA) under Indian law with:
{details}
Include: parties, definition of confidential information, obligations, exclusions, term, 
governing law (India), dispute resolution. Format as a proper legal document.""",

    "complaint_letter": """Draft a formal complaint letter for Indian authorities with:
{details}
Include: subject, factual account, relief sought, supporting documents mentioned. 
Format professionally for submission to relevant authority.""",

    "affidavit": """Draft an affidavit for Indian courts with:
{details}
Include proper affidavit format with deponent details, sworn statement, verification clause.
Format as per Indian court requirements.""",

    "demand_letter": """Draft a formal demand letter under Indian law with:
{details}
Include: demand clearly stated, factual background, legal basis, deadline, consequences if ignored.
Format professionally.""",
}


@router.get("/types")
async def list_document_types():
    """List all supported document types."""
    return {
        "supported_types": SUPPORTED_DOCUMENTS,
        "descriptions": {
            "legal_notice": "Formal legal notice for disputes, recovery, etc.",
            "rental_agreement": "Residential or commercial rental agreement",
            "nda": "Non-Disclosure / Confidentiality Agreement",
            "complaint_letter": "Complaint to police, consumer forum, etc.",
            "affidavit": "Sworn affidavit for court or official use",
            "demand_letter": "Formal demand for payment or action",
        },
    }


@router.post("/generate")
async def generate_document(request: DocumentRequest):
    """
    Generate a legal document based on type and provided details.
    """
    if request.document_type not in SUPPORTED_DOCUMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported document type. Supported: {SUPPORTED_DOCUMENTS}",
        )

    if not request.details:
        raise HTTPException(status_code=400, detail="Document details cannot be empty.")

    prompt_template = DOCUMENT_PROMPTS.get(request.document_type)

    # Format details as readable key-value pairs
    details_str = "\n".join(
        f"- {k.replace('_', ' ').title()}: {v}" for k, v in request.details.items()
    )
    prompt = prompt_template.format(details=details_str)

    model = request.model if request.model in ["llama3", "mistral"] else settings.llm_model

    try:
        document_text = await generate_response(
            prompt=prompt,
            model=model,
            system_prompt=build_system_prompt(),
            ollama_url=settings.ollama_base_url,
            max_tokens=3000,
            temperature=0.2,  # Slightly higher for document creativity
        )

        disclaimer = (
            "\n\n---\n⚠️ IMPORTANT: This document is AI-generated and for reference only. "
            "Please review with a qualified lawyer before use. "
            "Advocate AI is not responsible for legal consequences of using this document."
        )

        return {
            "document_type": request.document_type,
            "document": document_text + disclaimer,
            "model_used": model,
        }

    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Document generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
