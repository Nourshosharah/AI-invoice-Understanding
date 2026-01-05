"""
Data Models and Schemas for Invoice Understanding System

This module defines all Pydantic models used for data validation,
serialization, and API contracts throughout the system.
"""

from pydantic import BaseModel, Field
from datetime import date, datetime
from typing import Optional, List
from enum import Enum


class LineItem(BaseModel):
    """
    Structured line item extracted from an invoice.
    
    Represents a single row in an invoice's line items table with
    all seven required fields plus metadata for tracking.
    """
    item_code: Optional[str] = Field(
        default=None,
        description="Unique identifier/code for the item"
    )
    description: Optional[str] = Field(
        default=None,
        description="Item description or name"
    )
    delivery_date: Optional[date] = Field(
        default=None,
        description="Date the item was delivered or is due"
    )
    quantity: Optional[float] = Field(
        default=None,
        description="Number of units purchased"
    )
    unit_price: Optional[float] = Field(
        default=None,
        description="Price per single unit"
    )
    total_amount: Optional[float] = Field(
        default=None,
        description="Total amount for this line item (qty * unit_price)"
    )
    
    # Metadata fields (not from invoice but for tracking)
    page_number: Optional[int] = Field(
        default=None,
        description="Page number where this item appears"
    )
    line_number: Optional[int] = Field(
        default=None,
        description="Line number on the page where this item appears"
    )
    
    class Config:
        json_encoders = {
            date: lambda v: v.isoformat() if v else None
        }
    
    def to_search_text(self) -> str:
        """Generate searchable text content for embedding."""
        parts = []
        if self.item_code:
            parts.append(f"Item Code: {self.item_code}")
        if self.description:
            parts.append(f"Description: {self.description}")
        if self.unit_price:
            parts.append(f"Unit Price: ${self.unit_price:.2f}")
        if self.quantity:
            parts.append(f"Quantity: {self.quantity}")
        if self.total_amount:
            parts.append(f"Total: ${self.total_amount:.2f}")
        if self.delivery_date:
            parts.append(f"Delivery Date: {self.delivery_date.isoformat()}")
        return "\n".join(parts)
    
    def to_metadata(self, invoice_id: str) -> dict:
        """Generate metadata dict for vector store filtering."""
        return {
            "invoice_id": invoice_id,
            "page_number": self.page_number,
            "line_number": self.line_number,
            "item_code": self.item_code,
            "delivery_date": self.delivery_date.isoformat() if self.delivery_date else None,
            "quantity": self.quantity,
            "unit_price": self.unit_price,
            "total_amount": self.total_amount
        }


class InvoiceMetadata(BaseModel):
    """
    Metadata extracted from the invoice header.
    
    Contains information from the invoice header block including
    company information, invoice numbers, and financial totals.
    """
    invoice_id: str = Field(
        description="Unique identifier for the invoice"
    )
    invoice_date: Optional[date] = Field(
        default=None,
        description="Date the invoice was issued"
    )
    due_date: Optional[date] = Field(
        default=None,
        description="Payment due date"
    )
    company_name: Optional[str] = Field(
        default=None,
        description="Seller/company name"
    )
    company_address: Optional[str] = Field(
        default=None,
        description="Company address"
    )
    company_email: Optional[str] = Field(
        default=None,
        description="Company email"
    )
    customer_name: Optional[str] = Field(
        default=None,
        description="Customer/client name"
    )
    customer_address: Optional[str] = Field(
        default=None,
        description="Customer address"
    )
    subtotal: Optional[float] = Field(
        default=None,
        description="Subtotal before tax"
    )
    tax: Optional[float] = Field(
        default=None,
        description="Tax amount"
    )
    total: Optional[float] = Field(
        default=None,
        description="Grand total"
    )
    currency: str = Field(
        default="USD",
        description="Currency code (USD, EUR, etc.)"
    )
    
    class Config:
        json_encoders = {
            date: lambda v: v.isoformat() if v else None
        }


class ExtractionResult(BaseModel):
    """
    Complete extraction result for a single invoice.
    
    Contains all extracted data including line items and metadata
    from processing an invoice PDF.
    """
    invoice_id: str = Field(description="Unique invoice identifier")
    file_path: str = Field(description="Path to source PDF")
    file_name: str = Field(description="Original file name")
    total_pages: int = Field(description="Number of pages in PDF")
    line_items: List[LineItem] = Field(
        default_factory=list,
        description="Extracted line items"
    )
    metadata: InvoiceMetadata = Field(
        description="Invoice header metadata"
    )
    extraction_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When extraction was performed"
    )


class QueryType(str, Enum):
    """
    Classification of query types for routing.
    """
    STRUCTURAL = "structural"  # "Show page X items"
    FILTERED = "filtered"      # "March 2024 items"
    SEMANTIC = "semantic"      # "maintenance support"
    HYBRID = "hybrid"          # "support on page 3"


class QueryRoutingResult(BaseModel):
    """
    Result of query classification and routing.
    """
    query_type: QueryType = Field(description="Type of query")
    query_text: str = Field(description="Search text for vector search")
    filters: dict = Field(
        default_factory=dict,
        description="Metadata filters to apply"
    )
    reasoning: str = Field(description="Explanation of routing decision")


class RetrievedItem(BaseModel):
    """
    A single retrieved line item from the vector store.
    """
    invoice_id: str
    page_number: Optional[int]
    line_number: Optional[int]
    item_code: Optional[str]
    description: Optional[str]
    unit_price: Optional[float]
    quantity: Optional[float]
    total_amount: Optional[float]
    delivery_date: Optional[date]
    distance: float = Field(description="Similarity score")


class AnswerResult(BaseModel):
    """
    Generated answer with grounding information.
    """
    answer: str = Field(description="Generated response text")
    sources: List[dict] = Field(
        default_factory=list,
        description="Source references for verification"
    )
    retrieved_items: List[RetrievedItem] = Field(
        default_factory=list,
        description="All items retrieved for context"
    )
    query_type: Optional[QueryType] = Field(
        default=None,
        description="Type of query answered"
    )


class InvoiceSummary(BaseModel):
    """
    Summary information for listing indexed invoices.
    """
    invoice_id: str
    file_name: str
    total_pages: int
    line_item_count: int
    invoice_date: Optional[date]
    company_name: Optional[str]
    total_amount: Optional[float]
