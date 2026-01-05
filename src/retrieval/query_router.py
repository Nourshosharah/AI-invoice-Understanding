"""
Query Router for Invoice Q&A System

This module handles query classification, filter extraction,
and routing queries to appropriate retrieval strategies.
"""

import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from enum import Enum


class QueryType(str, Enum):
    """Classification of query types."""
    STRUCTURAL = "structural"  # Page-based queries
    FILTERED = "filtered"      # Metadata-filtered queries
    SEMANTIC = "semantic"      # Natural language semantic search
    HYBRID = "hybrid"          # Combines semantic + filters


class QueryRouter:
    """
    Routes user queries to appropriate retrieval strategies.
    
    Analyzes query text to:
    1. Classify query type
    2. Extract metadata filters
    3. Determine semantic search text
    """
    
    # Patterns for structural queries (page-based)
    STRUCTURAL_PATTERNS = [
        r'page\s*(\d+)',
        r'pages?\s*(\d+)\s*(?:to|-)\s*(\d+)',
        r'on\s+page\s*(\d+)',
        r'from\s+page\s*(\d+)',
    ]
    
    # Patterns for filtered queries (metadata-based)
    FILTER_PATTERNS = {
        'invoice_id': [
            r'invoice\s*[#:]?\s*([A-Z0-9-]+)',
            r'([A-Z]{2,4}[-_]\d+)',  # Pattern like INV-001, RXC-101
        ],
        'date': [
            r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}',
            r'\d{1,2}[/-]\d{1,2}[/-]\d{4}',
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
        ],
        'item_code': [
            r'code\s*[#:]?\s*([A-Z0-9]+)',
            r'item\s*[#:]?\s*([A-Z0-9]+)',
        ],
    }
    
    # Keywords suggesting semantic content
    SEMANTIC_KEYWORDS = [
        'related to', 'about', 'concerning', 'regarding',
        'maintenance', 'support', 'service', 'license',
        'software', 'hardware', 'consulting', 'training',
        'items', 'products', 'what', 'which',
    ]
    
    def __init__(self):
        """Initialize the query router."""
        pass
    
    def classify(self, query: str) -> QueryType:
        """
        Classify the type of query.
        
        Args:
            query: User query string
            
        Returns:
            QueryType enum value
        """
        has_structural = self._has_structural_component(query)
        has_filter = self._has_filter_component(query)
        has_semantic = self._has_semantic_component(query)
        
        # Determine query type
        if has_structural and has_semantic:
            return QueryType.HYBRID
        elif has_structural:
            return QueryType.STRUCTURAL
        elif has_filter and has_semantic:
            return QueryType.HYBRID
        elif has_filter:
            return QueryType.FILTERED
        elif has_semantic:
            return QueryType.SEMANTIC
        else:
            # Default to semantic for general queries
            return QueryType.SEMANTIC
    
    def route(self, query: str) -> Dict:
        """
        Perform full routing analysis on a query.
        
        Args:
            query: User query string
            
        Returns:
            Dict with query_type, query_text, filters, and reasoning
        """
        query_type = self.classify(query)
        
        # Extract filters
        filters = self.extract_filters(query)
        
        # Extract semantic search text
        search_text = self.extract_semantic_text(query)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(query, query_type, filters, search_text)
        
        return {
            "query_type": query_type,
            "query_text": search_text,
            "filters": filters,
            "reasoning": reasoning,
        }
    
    def extract_filters(self, query: str) -> Dict:
        """
        Extract metadata filters from query.
        
        Args:
            query: User query string
            
        Returns:
            Dict of filter conditions
        """
        filters = {}
        query_lower = query.lower()
        
        # Extract page number filters
        page_filters = self._extract_page_filters(query)
        if page_filters:
            filters.update(page_filters)
        
        # Extract invoice ID
        invoice_id = self._extract_invoice_id(query)
        if invoice_id:
            filters["invoice_id"] = invoice_id
        
        # Extract date filters
        date_filter = self._extract_date_filter(query)
        if date_filter:
            filters.update(date_filter)
        
        # Extract item code
        item_code = self._extract_item_code(query)
        if item_code:
            filters["item_code"] = item_code
        
        # Extract numeric range filters
        numeric_filters = self._extract_numeric_filters(query)
        if numeric_filters:
            filters.update(numeric_filters)
        
        return filters
    
    def extract_semantic_text(self, query: str) -> str:
        """
        Extract the semantic search component from query.
        
        Removes structural and filter indicators to leave
        the natural language content for vector search.
        
        Args:
            query: User query string
            
        Returns:
            Cleaned text for semantic search
        """
        text = query
        
        # Remove page references
        for pattern in self.STRUCTURAL_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove invoice ID references
        for pattern in self.FILTER_PATTERNS['invoice_id']:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove date references
        for pattern in self.FILTER_PATTERNS['date']:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove common query prefixes
        prefixes = [
            r'^show\s+me\s+',
            r'^what\s+(?:are|is)\s+',
            r'^list\s+',
            r'^get\s+',
            r'^find\s+',
            r'^all\s+',
        ]
        for prefix in prefixes:
            text = re.sub(prefix, '', text, flags=re.IGNORECASE)
        
        # Remove trailing question marks
        text = text.rstrip('?')
        
        return text if text else ""
    
    def _has_structural_component(self, query: str) -> bool:
        """Check if query has structural (page-based) component."""
        for pattern in self.STRUCTURAL_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    def _has_filter_component(self, query: str) -> bool:
        """Check if query has filterable metadata components."""
        # Check for date references
        for pattern in self.FILTER_PATTERNS['date']:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        
        # Check for item code references
        for pattern in self.FILTER_PATTERNS['item_code']:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        
        # Check for invoice ID patterns
        for pattern in self.FILTER_PATTERNS['invoice_id']:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        
        return False
    
    def _has_semantic_component(self, query: str) -> bool:
        """
        Check if query has semantic (content) component.
        
        Returns True only if there's meaningful content beyond just
        structural/filter terms (e.g., "show me all items" is NOT semantic).
        """
        query_lower = query.lower()
        
        # FIRST: Check if the query is purely about retrieving items/filters
        # These patterns indicate just structural/filter queries (NOT semantic):
        # Check BOTH the original query and the cleaned version
        cleaned = self.extract_semantic_text(query)
        
        retrieval_patterns_original = [
            r'give\s+(?:me\s+)?all\s+items?\s+in',
            r'give\s+(?:me\s+)?all\s+items?\s+from',
            r'show\s+(?:me\s+)?all\s+items?\s+in',
            r'show\s+(?:me\s+)?all\s+items?\s+from',
            r'list\s+(?:all\s+)?items?\s+in',
            r'list\s+(?:all\s+)?items?\s+from',
            r'get\s+(?:all\s+)?items?\s+in',
            r'get\s+(?:all\s+)?items?\s+from',
        ]
        
        retrieval_patterns_cleaned = [
            r'^all\s+items?',
            r'^all\s+items?\s+in',
            r'^all\s+items?\s+from',
            r'^items?\s+in\s+[A-Z0-9-]+',
            r'^items?\s+from\s+[A-Z0-9-]+',
        ]
        
        # Check original query
        for pattern in retrieval_patterns_original:
            if re.search(pattern, query_lower):
                return False  # This is NOT a semantic query
        
        # Check cleaned query
        for pattern in retrieval_patterns_cleaned:
            if re.match(pattern, cleaned, re.IGNORECASE):
                return False  # This is NOT a semantic query
        
        if not cleaned or len(cleaned) < 3:
            return False
        
        # SECOND: Check for semantic keywords (real content words)
        # Only if we have meaningful cleaned content
        semantic_keywords = [
            'related to', 'about', 'concerning', 'regarding',
            'maintenance', 'support', 'service', 'license',
            'software', 'hardware', 'consulting', 'training',
        ]
        
        for keyword in semantic_keywords:
            if keyword in query_lower:
                return True
        
        # Check if the cleaned text contains actual semantic content
        # These patterns indicate just structural/filter queries:
        structural_patterns = [
            # Patterns for retrieval queries (action verb + optional "me/all/items")
            r'^(show|list|get|find|give|all)\s+(me\s+)?(all\s+)?(items?|records?|results?)?\s*(in|from|of)?\s*$',
            r'^(what\s+(is|are)|which)',
            r'^items?\s+from',
        ]
        
        for pattern in structural_patterns:
            if re.match(pattern, cleaned, re.IGNORECASE):
                return False
        
        # If cleaned text is substantial and not just structural, it's semantic
        # Use 15 chars as threshold (filters out most structural leftovers)
        return len(cleaned) > 15
    
    def _extract_page_filters(self, query: str) -> Dict:
        """Extract page number filters from query."""
        filters = {}
        
        # Single page: "page 3", "on page 5"
        single_match = re.search(
            r'(?:page|on\s+page|from\s+page)\s*(\d+)',
            query, re.IGNORECASE
        )
        if single_match:
            filters["page_number"] = int(single_match.group(1))
            return filters
        
        # Page range: "pages 10-15", "from page 5 to page 10"
        range_match = re.search(
            r'pages?\s*(\d+)\s*(?:to|-)\s*(\d+)',
            query, re.IGNORECASE
        )
        if range_match:
            filters["page_number"] = {
                "$gte": int(range_match.group(1)),
                "$lte": int(range_match.group(2)),
            }
        
        return filters
    
    def _extract_invoice_id(self, query: str) -> Optional[str]:
        """Extract invoice ID from query with strict pattern matching."""
        # Must have space before invoice keyword, and ID should NOT start with underscore
        # Patterns: "invoice INV-98448", "invoice_id INV-98448", "invoice# INV-98448"
        patterns = [
            r'(?:^|\s)invoice\s*[#:]?\s*(INV-[A-Z0-9-]+)',  # invoice INV-98448
            r'(?:^|\s)invoice_id\s+(INV-[A-Z0-9-]+)',        # invoice_id INV-98448 (space required!)
            r'(?:^|\s)(INV-[A-Z0-9]{3,})[?.!,\s]*',         # INV-XXX with trailing punctuation
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        return None
    
    def _extract_date_filter(self, query: str) -> Dict:
        """Extract date range filters from query."""
        filters = {}
        query_lower = query.lower()
        
        # Month + Year: "March 2024", "january 2023"
        month_year_match = re.search(
            r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})',
            query_lower
        )
        if month_year_match:
            month = month_year_match.group(1)
            year = int(month_year_match.group(2))
            
            # Convert to date range
            month_num = self._month_to_num(month)
            filters["delivery_date"] = {
                "$gte": f"{year}-{month_num:02d}-01",
                "$lte": f"{year}-{month_num:02d}-31",
            }
        
        # Year only: "in 2024", "during 2023"
        year_match = re.search(r'(?:in|during|of)\s+(\d{4})', query_lower)
        if year_match:
            year = int(year_match.group(1))
            filters["delivery_date"] = {
                "$gte": f"{year}-01-01",
                "$lte": f"{year}-12-31",
            }
        
        return filters
    
    def _extract_item_code(self, query: str) -> Optional[str]:
        """Extract item code from query using strict patterns."""
        # Item codes typically have patterns like: FS-1000, SKU-500, ABC-123
        # IMPORTANT: Must NOT match invoice IDs like INV-98448
        # Common prefixes: FS, SKU, PROD, ITEM, ABC, XYZ, etc. (but NOT INV)
        patterns = [
            r'item\s*(?:code)?\s*[#:]?\s*([A-Z]{2,4}-\d{3,})',  # FS-1000, SKU-500
            r'code\s*[#:]?\s*([A-Z]{2,4}-\d{3,})',               # code: FS-1000
            r'(?:^|\s)(?!INV-)([A-Z]{2,4}-\d{3,})(?:$|\s)',      # Standalone (exclude INV-)
            r'(?:^|\s)(?!INV-)([A-Z]{2,4}\d{3,})(?:$|\s)',       # ABC123 standalone (exclude INV)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                result = match.group(1).upper()
                # Double-check it's not an invoice ID pattern
                if not result.startswith('INV-'):
                    return result
        
        return None
    
    def _extract_numeric_filters(self, query: str) -> Dict:
        """Extract numeric range filters (price, quantity, etc.)."""
        filters = {}
        
        # Price greater than: "over $500", "more than 1000"
        price_gt_match = re.search(
            r'(?:over|more\s+than|above|greater\s+than|>\s*)\$?([\d,]+)',
            query, re.IGNORECASE
        )
        if price_gt_match:
            value = float(price_gt_match.group(1).replace(',', ''))
            filters["total_amount"] = {"$gte": value}
        
        # Price less than: "under $500", "less than 1000"
        price_lt_match = re.search(
            r'(?:under|less\s+than|below|less\s+than|<\s*)\$?([\d,]+)',
            query, re.IGNORECASE
        )
        if price_lt_match:
            value = float(price_lt_match.group(1).replace(',', ''))
            existing = filters.get("total_amount", {})
            existing["$lte"] = value
            filters["total_amount"] = existing
        
        return filters
    
    def _month_to_num(self, month: str) -> int:
        """Convert month abbreviation to number."""
        months = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
            'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
            'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
        }
        return months.get(month[:3], 1)
    
    def _generate_reasoning(
        self,
        query: str,
        query_type: QueryType,
        filters: Dict,
        search_text: str
    ) -> str:
        """Generate human-readable reasoning for routing decision."""
        parts = [f"Classified as {query_type.value} query"]
        
        if filters:
            filter_parts = []
            for key, value in filters.items():
                filter_parts.append(f"{key}={value}")
            parts.append(f"Filters: {', '.join(filter_parts)}")
        
        if search_text:
            parts.append(f"Semantic search: '{search_text}'")
        
        return ". ".join(parts)
