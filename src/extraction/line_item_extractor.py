"""
Line Item Extractor for Invoice Processing

This module handles the extraction of structured line item data from
raw table data extracted from invoice PDFs.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import pandas as pd


class LineItemExtractor:
    """
    Extracts structured line items from invoice table data.
    
    Uses a combination of header detection, column mapping, and
    pattern matching to convert raw table cells into LineItem objects.
    """
    
    # Column header mappings to field names
    HEADER_MAPPINGS = {
        'item_code': ['item code', 'item_code', 'code', 'product code', 'sku', 'item #', 'product code'],
        'description': ['description', 'desc', 'item description', 'particulars', 'item name', 'particular', 'details'],
        'delivery_date': ['delivery date', 'delivery_date', 'date', 'ship date', 'due date', 'service date'],
        'quantity': ['quantity', 'qty', 'qty.', 'units', 'count'],
        'unit_price': ['unit price', 'unit_price', 'unit price', 'price', 'rate', 'unit rate', 'unitcost'],
        'total_amount': ['amount', 'total', 'line total', 'line total', 'amount due', 'subtotal', 'total amount'],
    }
    
    # Patterns for parsing numeric values
    NUMBER_PATTERNS = {
        'decimal': r'[\d,]+\.?\d*',
        'currency': r'\$[\d,]+\.?\d*',
        'quantity': r'\d+\.?\d*',
    }
    
    def __init__(self):
        """Initialize the line item extractor."""
        pass
    
    def extract(self, tables: List[Dict], invoice_metadata: Dict) -> List[Dict]:
        """
        Extract all line items from tables.
        
        Args:
            tables: List of table dicts from invoice parser
            invoice_metadata: Invoice header metadata
            
        Returns:
            List of extracted line item dictionaries
        """
        all_items = []
        
        for table in tables:
            cells = table.get('cells', [])
            page_num = table.get('page_number', 1)
            
            if not cells or len(cells) < 2:
                continue
            
            # Detect column headers
            header_mapping = self._detect_column_headers(cells[0])
            
            if not header_mapping:
                continue  # Skip tables without recognizable headers
            
            # Extract items from data rows
            for row_idx, row in enumerate(cells[1:], start=2):
                if not row or all(not cell for cell in row):
                    continue  # Skip empty rows
                
                item = self._parse_row(row, header_mapping, page_num, row_idx, invoice_metadata)
                if item and self._is_valid_item(item):
                    all_items.append(item)
        
        return all_items
    
    def _detect_column_headers(self, header_row: List) -> Dict[int, str]:
        """
        Detect and map column headers to field names.
        
        Args:
            header_row: First row of table containing headers
            
        Returns:
            Dict mapping column index to field name
        """
        column_mapping = {}
        
        for col_idx, cell in enumerate(header_row):
            if not cell:
                continue
            
            cell_lower = str(cell).lower().strip()
            
            # Match against known header patterns
            for field_name, patterns in self.HEADER_MAPPINGS.items():
                for pattern in patterns:
                    if pattern in cell_lower:
                        column_mapping[col_idx] = field_name
                        break
        
        return column_mapping
    
    def _parse_row(
        self,
        row: List,
        column_mapping: Dict[int, str],
        page_num: int,
        line_num: int,
        invoice_metadata: Dict
    ) -> Optional[Dict]:
        """
        Parse a single row into a line item dictionary.
        
        Args:
            row: List of cell values from the row
            column_mapping: Mapping of column indices to field names
            page_num: Page number where this row appears
            line_num: Line number on the page
            invoice_metadata: Invoice header metadata for defaults
            
        Returns:
            Line item dictionary or None if parsing fails
        """
        item = {
            'page_number': page_num,
            'line_number': line_num,
        }
        
        for col_idx, cell in enumerate(row):
            if col_idx not in column_mapping:
                continue
            
            field_name = column_mapping[col_idx]
            cell_value = str(cell).strip() if cell else ''
            
            if not cell_value:
                continue
            
            # Parse based on field type
            if field_name == 'item_code':
                item['item_code'] = self._parse_item_code(cell_value)
            elif field_name == 'description':
                item['description'] = self._parse_description(cell_value)
            elif field_name == 'delivery_date':
                item['delivery_date'] = self._parse_date(cell_value)
            elif field_name == 'quantity':
                item['quantity'] = self._parse_quantity(cell_value)
            elif field_name == 'unit_price':
                item['unit_price'] = self._parse_price(cell_value)
            elif field_name == 'total_amount':
                item['total_amount'] = self._parse_price(cell_value)
        
        return item
    
    def _parse_item_code(self, value: str) -> str:
        """
        Parse item code from cell value.
        
        Args:
            value: Raw cell value
            
        Returns:
            Cleaned item code string
        """
        # Remove common prefixes/suffixes
        cleaned = re.sub(r'^[#\s-]+', '', value)
        cleaned = re.sub(r'[#\s-]+$', '', cleaned)
        return cleaned.strip()
    
    def _parse_description(self, value: str) -> str:
        """
        Parse description from cell value.
        
        Args:
            value: Raw cell value
            
        Returns:
            Cleaned description string
        """
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', value)
        return cleaned.strip()
    
    def _parse_date(self, value: str) -> Optional[str]:
        """
        Parse date from cell value.
        
        Args:
            value: Raw cell value
            
        Returns:
            Date string in YYYY-MM-DD format or None
        """
        date_formats = [
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%m-%d-%Y',
            '%m/%d/%Y',
            '%d-%m-%Y',
            '%d/%m/%Y',
            '%B %d, %Y',
            '%b %d, %Y',
            '%d %B %Y',
            '%d %b %Y',
            '%m/%d/%y',
            '%d-%m-%y',
        ]
        
        for fmt in date_formats:
            try:
                parsed = datetime.strptime(value.strip(), fmt)
                return parsed.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        return None
    
    def _parse_quantity(self, value: str) -> Optional[float]:
        """
        Parse quantity from cell value.
        
        Args:
            value: Raw cell value
            
        Returns:
            Quantity as float or None
        """
        try:
            # Remove any non-numeric characters except decimal point
            cleaned = re.sub(r'[^\d.]', '', value)
            if cleaned:
                return float(cleaned)
        except (ValueError, TypeError):
            pass
        
        return None
    
    def _parse_price(self, value: str) -> Optional[float]:
        """
        Parse price/amount from cell value.
        
        Args:
            value: Raw cell value
            
        Returns:
            Price as float or None
        """
        try:
            # Handle currency symbols and commas
            cleaned = value.replace('$', '').replace(',', '').replace('€', '').replace('£', '')
            cleaned = cleaned.strip()
            
            if cleaned:
                return float(cleaned)
        except (ValueError, TypeError):
            pass
        
        return None
    
    def _is_valid_item(self, item: Dict) -> bool:
        """
        Check if parsed item has minimum required fields.
        
        Args:
            item: Parsed line item dictionary
            
        Returns:
            True if item is valid, False otherwise
        """
        # Must have at least description or one numeric field
        has_description = item.get('description') and len(item['description']) > 0
        has_amount = item.get('total_amount') is not None
        has_quantity = item.get('quantity') is not None
        has_price = item.get('unit_price') is not None
        
        # Item is valid if it has description and at least one numeric field
        if has_description and (has_amount or has_quantity or has_price):
            return True
        
        # Also valid if it has item code
        if item.get('item_code') and has_amount:
            return True
        
        return False
