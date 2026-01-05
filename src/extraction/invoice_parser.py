"""
Invoice Parser for PDF and Image Documents

This module handles document parsing for both PDF and image formats.
- PDFs are processed using pdfplumber for text and table extraction
- Images are processed using pytesseract OCR for text extraction
- Multiple fallback strategies for invoices without visible table borders
- Spatial text analysis for extracting tabular data from layout
"""

import pdfplumber
import pytesseract
from PIL import Image
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TextWord:
    """Represents a word with its position on the page."""
    text: str
    x0: float  # Left edge
    x1: float  # Right edge
    top: float  # Top edge
    bottom: float  # Bottom edge
    page_width: float
    page_height: float


class InvoiceParser:
    """
    Unified invoice parser for PDF and image documents.
    
    Supports:
    - PDF documents (multi-page)
    - Image formats: PNG, JPG, JPEG, BMP, TIFF
    - Multiple table extraction strategies (standard + fallback)
    - Spatial text analysis for borderless tables
    """
    
    # Supported image extensions
    SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    # Supported PDF extensions
    SUPPORTED_PDF_EXTENSIONS = {'.pdf'}
    
    # Common invoice header patterns for metadata extraction
    INVOICE_ID_PATTERNS = [
        r'Invoice\s*#?\s*:?\s*(INV-[A-Z0-9-]+)',  # Invoice #: INV-98448
        r'Invoice\s*Number\s*:?\s*(INV-[A-Z0-9-]+)',  # Invoice Number: INV-98448
        r'(?:^|\s)(INV-[A-Z0-9-]+)',  # INV-XXXXX at word boundary
        r'Invoice\s*#?\s*:?\s*([A-Z0-9-]+)',  # Fallback to any alphanumeric
        r'#\s*([A-Z0-9-]+)',
        r'(?:INVOICE|INV)\s*[:#]?\s*([A-Z0-9-]+)',
    ]
    
    DATE_PATTERNS = [
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',
        r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})',
        r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})',
    ]
    
    # Standard column headers for line items
    COLUMN_HEADERS = [
        'item_code', 'item code', 'code', 'product code',
        'description', 'desc', 'item description', 'particulars',
        'delivery date', 'delivery_date', 'date', 'ship date',
        'quantity', 'qty', 'qty.',
        'unit price', 'unit_price', 'unit price', 'price', 'rate',
        'amount', 'total', 'amount due', 'line total',
    ]
    
    # Header patterns that indicate start of line items table
    TABLE_HEADER_PATTERNS = [
        r'^\s*(?:Item\s*Code|Code|SKU)\s+(?:Description|Desc|Item)\s+(?:Qty|Quantity|Qty\.?)\s+(?:Unit\s*Price|Unit\s*Price|Rate)\s+(?:Amount|Total)',
        r'^\s*(?:Item|Code|Qty|Description)\s+\d+\s+\d+',  # Numeric columns pattern
        r'^\s*(?:Description|Item\s*Description)\s+(?:Qty|Quantity)\s+(?:Price|Rate)',
        r'^\s*(?:Item|Product|Services?)\s+(?:Qty|Quantity)',
    ]
    
    # Patterns that indicate footer/summary (not line items)
    SUMMARY_PATTERNS = [
        r'^sub\s*total',
        r'^grand\s*total',
        r'^total\s*due',
        r'^balance\s*due',
        r'^tax\s*[:/]',
        r'^vat\s*[:/]',
        r'^shipping',
        r'^shipping\s*cost',
    ]
    
    # Patterns for line item recognition
    LINE_ITEM_PATTERNS = [
        # Pattern: description followed by numbers
        r'^(.+?)\s+(\d+(?:[.,]\d+)?)\s+\$?([\d,]+\.?\d*)\s+\$?([\d,]+\.?\d*)',
        # Pattern: item code, description, qty, price, total
        r'^([A-Z0-9-]+)\s+(.+?)\s+(\d+(?:[.,]\d+)?)\s+\$?([\d,]+\.?\d*)\s+\$?([\d,]+\.?\d*)',
    ]
    
    def __init__(
        self,
        tesseract_path: Optional[str] = None,
        use_ocr_fallback: bool = True,
        ocr_language: str = 'eng',
        spatial_tolerance: float = 3.0  # Vertical tolerance for grouping words
    ):
        """
        Initialize the invoice parser.
        
        Args:
            tesseract_path: Optional path to tesseract executable
            use_ocr_fallback: Whether to use OCR fallback for PDFs
            ocr_language: Language for OCR (default: English)
            spatial_tolerance: Vertical tolerance for spatial text grouping
        """
        # Set tesseract path if provided
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        self.use_ocr_fallback = use_ocr_fallback
        self.ocr_language = ocr_language
        self.spatial_tolerance = spatial_tolerance
        self.ocr_available = self._check_ocr_available()
    
    def _check_ocr_available(self) -> bool:
        """Check if Tesseract OCR is available."""
        try:
            # Check if pytesseract has the required functions
            if not hasattr(pytesseract, 'get_tesseract_version'):
                logger.warning("pytesseract.get_tesseract_version not available")
                return False
            
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR is available")
            return True
        except (AttributeError, Exception) as e:
            logger.warning(f"Tesseract OCR not available: {e}")
            return False
    
    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        Parse an invoice document (PDF or image) and extract all content.
        
        Args:
            file_path: Path to the document (PDF or image file)
            
        Returns:
            Dictionary containing extracted content
        """
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext in self.SUPPORTED_PDF_EXTENSIONS:
            return self._parse_pdf(file_path)
        elif file_ext in self.SUPPORTED_IMAGE_EXTENSIONS:
            return self._parse_image(file_path)
        else:
            raise ValueError(
                f"Unsupported file format: {file_ext}. "
                f"Supported formats: PDF and images"
            )
    
    def _parse_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Parse a PDF invoice document with multiple extraction strategies.
        
        Uses a hierarchical approach:
        1. Standard pdfplumber table extraction
        2. Spatial text analysis (for borderless tables)
        3. Pattern-based line extraction
        4. OCR fallback if enabled
        """
        result = {
            'text_content': '',
            'tables': [],
            'metadata': {},
            'page_count': 0,
            'file_name': Path(pdf_path).name,
            'file_type': 'pdf',
        }
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                result['page_count'] = len(pdf.pages)
                
                all_tables = []
                all_text_parts = []
                extraction_methods_used = []
                
                for page_num, page in enumerate(pdf.pages, start=1):
                    logger.info(f"Processing page {page_num}")
                    
                    # Strategy 1: Try standard table extraction
                    tables = self._extract_tables_from_page(page)
                    if tables:
                        extraction_methods_used.append('standard')
                        logger.info(f"  Found {len(tables)} tables using standard extraction")
                    
                    # Strategy 2: If no tables found, try spatial analysis
                    if not tables:
                        logger.info("  No tables found with standard extraction, trying spatial analysis...")
                        spatial_tables = self._extract_tables_spatially(page)
                        if spatial_tables:
                            tables = spatial_tables
                            extraction_methods_used.append('spatial')
                            logger.info(f"  Found {len(tables)} tables using spatial analysis")
                    
                    # Strategy 3: If still no tables, try pattern-based extraction
                    if not tables:
                        logger.info("  No tables found, trying pattern-based extraction...")
                        page_text = page.extract_text() or ""
                        pattern_tables = self._extract_tables_pattern_based(page_text)
                        if pattern_tables:
                            tables = pattern_tables
                            extraction_methods_used.append('pattern')
                            logger.info(f"  Found {len(tables)} tables using pattern extraction")
                    
                    # Strategy 4: OCR fallback for low quality text
                    page_text = page.extract_text() or ""
                    if self.use_ocr_fallback and not tables:
                        text_quality = self._assess_text_quality(page_text)
                        if text_quality < 0.3:  # Very low quality
                            logger.info(f"  Low text quality ({text_quality:.2f}), trying OCR...")
                            ocr_text = self._ocr_page(page)
                            
                            if ocr_text:
                                page_text = ocr_text
                                # Try pattern extraction on OCR text
                                ocr_tables = self._extract_tables_pattern_based(ocr_text)
                                if ocr_tables:
                                    tables = ocr_tables
                                    extraction_methods_used.append('ocr')
                                    logger.info(f"  Found {len(tables)} tables using OCR text")
                    
                    if page_text:
                        all_text_parts.append(f"[Page {page_num}]\n{page_text}")
                    
                    for table in tables:
                        table['page_number'] = page_num
                        all_tables.append(table)
                    
                    page_meta = self._extract_page_metadata(page_text, page_num)
                    result['metadata'].update(page_meta)
                
                result['text_content'] = '\n\n'.join(all_text_parts)
                result['tables'] = all_tables
                
                if extraction_methods_used:
                    result['metadata']['extraction_methods'] = list(set(extraction_methods_used))
                    logger.info(f"Extraction methods used: {extraction_methods_used}")
            
        except Exception as e:
            logger.error(f"Error parsing PDF: {e}")
            if self.ocr_available and self.use_ocr_fallback:
                logger.info("Attempting full document OCR as fallback")
                return self._ocr_fallback_pdf(pdf_path)
            raise
        
        result['metadata'].update(
            self._extract_invoice_metadata(result['text_content'])
        )
        
        return result
    
    def _parse_image(self, image_path: str) -> Dict[str, Any]:
        """Parse an image invoice document using OCR."""
        result = {
            'text_content': '',
            'tables': [],
            'metadata': {},
            'page_count': 1,
            'file_name': Path(image_path).name,
            'file_type': 'image',
        }
        
        if not self.ocr_available:
            raise RuntimeError("Tesseract OCR is not available for image processing")
        
        try:
            image = Image.open(image_path)
            
            logger.info(f"Performing OCR on image: {image_path}")
            text = pytesseract.image_to_string(image, lang=self.ocr_language)
            
            if text:
                result['text_content'] = f"[Image]\n{text}"
            else:
                result['text_content'] = "[Image]\n"
            
            # Extract tables using pattern-based method
            tables = self._extract_tables_pattern_based(text)
            for table in tables:
                table['page_number'] = 1
                result['tables'].append(table)
            
            result['metadata'] = self._extract_invoice_metadata(text)
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise
        
        return result
    
    def _extract_tables_from_page(self, page) -> List[Dict[str, Any]]:
        """
        Extract tables from a single page using pdfplumber.
        
        First tries basic table extraction, then falls back to text-based parsing
        for invoices with borderless tables.
        """
        tables = []
        
        # Try basic table extraction first
        try:
            extracted_tables = page.extract_tables()
            if extracted_tables:
                for table in extracted_tables:
                    if table and len(table) > 1:
                        tables.append({
                            'cells': table,
                            'settings': {'extraction_method': 'basic'},
                        })
                if tables:
                    logger.info(f"  Basic extraction: Found {len(tables)} table(s)")
                    return tables
        except Exception as e:
            logger.debug(f"Basic table extraction failed: {e}")
        
        # Fall back to text-based extraction for borderless tables
        text_tables = self._extract_tables_from_text(page)
        if text_tables:
            tables.extend(text_tables)
        
        return tables
    
    def _extract_tables_spatially(self, page) -> List[Dict[str, Any]]:
        """
        Extract tables using spatial text analysis.
        
        This method handles invoices with "invisible tables" where text is
        positioned in columns without visible borders.
        
        Strategy:
        1. Extract all words with positions
        2. Group words into lines based on vertical position
        3. Detect column boundaries from header row
        4. Parse each line into cells based on column positions
        """
        tables = []
        
        try:
            # Extract words with positions
            words = page.extract_words()
            if not words or len(words) < 3:
                return tables
            
            page_width = page.width
            page_height = page.height
            
            # Build TextWord objects
            word_objects = [
                TextWord(
                    text=w['text'],
                    x0=w['x0'],
                    x1=w['x1'],
                    top=w['top'],
                    bottom=w['bottom'],
                    page_width=page_width,
                    page_height=page_height
                )
                for w in words
            ]
            
            # Find potential table area
            table_area = self._find_table_area(word_objects)
            if not table_area:
                return tables
            
            # Filter words to table area
            table_words = [w for w in word_objects 
                          if table_area['top'] <= w.top <= table_area['bottom']]
            
            # Group words into lines
            lines = self._group_words_into_lines(table_words)
            
            # Detect column boundaries
            column_breaks = self._detect_column_breaks(lines)
            
            if not column_breaks or len(column_breaks) < 2:
                return tables
            
            # Parse lines into table cells
            table_cells = []
            for line in lines:
                cells = self._parse_line_into_cells(line, column_breaks)
                if cells and any(c.strip() for c in cells):
                    table_cells.append(cells)
            
            if table_cells and len(table_cells) >= 2:  # Header + at least one row
                tables.append({
                    'cells': table_cells,
                    'settings': {'extraction_method': 'spatial'},
                    'spatial_extracted': True,
                })
                logger.info(f"  Spatial extraction: {len(table_cells)} rows, {len(column_breaks)} columns")
        
        except Exception as e:
            logger.debug(f"Spatial extraction failed: {e}")
        
        return tables
    
    def _find_table_area(self, words: List[TextWord]) -> Optional[Dict[str, float]]:
        """
        Find the approximate area of the table on the page.
        
        Looks for regions with multiple numeric values and alignment patterns.
        """
        if not words:
            return None
        
        # Get page dimensions
        page_width = words[0].page_width
        page_height = words[0].page_height
        
        # Split page into horizontal zones
        zone_height = page_height / 20  # Divide into 20 zones
        zone_densities = []
        
        for i in range(20):
            zone_top = i * zone_height
            zone_bottom = (i + 1) * zone_height
            zone_words = [w for w in words if zone_top <= w.top < zone_bottom]
            
            # Calculate density score based on word count and numeric content
            numeric_count = sum(1 for w in zone_words if re.search(r'\d', w.text))
            density = len(zone_words) + numeric_count * 0.5
            zone_densities.append((zone_top, zone_bottom, density, zone_words))
        
        # Find zones with high density (likely table area)
        zone_densities.sort(key=lambda x: x[2], reverse=True)
        
        # Take top zones that are clustered together
        if zone_densities:
            best_zone = zone_densities[0]
            # Extend to include nearby zones
            table_top = best_zone[0]
            table_bottom = best_zone[1]
            
            # Include adjacent zones if they have significant content
            for zone in zone_densities[1:5]:
                if zone[2] > best_zone[2] * 0.5:  # At least 50% of best zone
                    if zone[0] > table_top - zone_height * 2 and zone[1] < table_bottom + zone_height * 3:
                        table_top = min(table_top, zone[0])
                        table_bottom = max(table_bottom, zone[1])
            
            # Add some margin
            margin = zone_height * 0.5
            return {
                'top': max(0, table_top - margin),
                'bottom': min(page_height, table_bottom + margin),
            }
        
        return None
    
    def _group_words_into_lines(self, words: List[TextWord]) -> List[List[TextWord]]:
        """
        Group words into lines based on vertical position.
        
        Words with similar y-positions (within tolerance) are considered on the same line.
        """
        if not words:
            return []
        
        # Sort by vertical position
        sorted_words = sorted(words, key=lambda w: w.top)
        
        lines = []
        current_line = [sorted_words[0]]
        current_y = sorted_words[0].top
        
        for word in sorted_words[1:]:
            # Check if word is on the same line (within tolerance)
            if abs(word.top - current_y) <= self.spatial_tolerance:
                current_line.append(word)
            else:
                # Start new line
                # Sort current line by horizontal position
                current_line.sort(key=lambda w: w.x0)
                lines.append(current_line)
                current_line = [word]
                current_y = word.top
        
        # Don't forget the last line
        if current_line:
            current_line.sort(key=lambda w: w.x0)
            lines.append(current_line)
        
        return lines
    
    def _detect_column_breaks(self, lines: List[List[TextWord]]) -> List[float]:
        """
        Detect column boundaries from header row.
        
        Looks for the header row and identifies column positions.
        """
        if not lines:
            return []
        
        # Find the header row (look for typical column headers)
        header_keywords = ['description', 'qty', 'quantity', 'price', 'amount', 'total', 'code', 'item']
        
        header_line = None
        for line in lines[:10]:  # Check first 10 lines
            line_text = ' '.join(w.text.lower() for w in line)
            matches = sum(1 for kw in header_keywords if kw in line_text)
            if matches >= 2:  # At least 2 header keywords
                header_line = line
                break
        
        if not header_line:
            # Fallback: use the first line with multiple words
            for line in lines:
                if len(line) >= 3:
                    header_line = line
                    break
        
        if not header_line:
            return []
        
        # Detect column breaks from header line
        column_breaks = [0.0]  # Start from left edge
        
        for i, word in enumerate(header_line[1:], 1):
            # Place break between this word and previous
            prev_word = header_line[i - 1]
            break_point = (prev_word.x1 + word.x0) / 2
            column_breaks.append(break_point)
        
        # Add right edge
        last_word = header_line[-1]
        column_breaks.append(last_word.page_width)
        
        return column_breaks
    
    def _parse_line_into_cells(self, line: List[TextWord], column_breaks: List[float]) -> List[str]:
        """
        Parse a line of words into table cells based on column positions.
        """
        if not line or not column_breaks:
            return []
        
        cells = []
        
        for i in range(len(column_breaks) - 1):
            left = column_breaks[i]
            right = column_breaks[i + 1]
            
            # Get words that fall in this column
            cell_words = [w.text for w in line if w.x0 >= left and w.x1 <= right + 5]
            
            # Also include words that overlap with the column
            if not cell_words:
                cell_words = [w.text for w in line if w.x0 < right and w.x1 > left]
            
            cells.append(' '.join(cell_words))
        
        return cells
    
    def _extract_tables_from_text(self, page) -> List[Dict[str, Any]]:
        """
        Extract tables from text-based tabular structure.
        
        For invoices where text extraction preserves column alignment,
        this method parses the text to reconstruct table structure.
        """
        tables = []
        
        try:
            # Get text with line information
            text_lines = page.extract_text_lines()
            if not text_lines:
                return tables
            
            # Find the header line (contains typical column headers)
            header_keywords = ['description', 'qty', 'quantity', 'price', 'amount', 'total', 'code', 'item', 'unit']
            header_idx = None
            
            for i, line in enumerate(text_lines):
                line_text = line.get('text', '').lower()
                matches = sum(1 for kw in header_keywords if kw in line_text)
                if matches >= 2:  # At least 2 header keywords found
                    header_idx = i
                    break
            
            if header_idx is None:
                return tables
            
            # Get header text and detect column positions
            header_line = text_lines[header_idx]
            header_text = header_line.get('text', '')
            
            # Detect column breaks from header using text positions
            column_breaks = self._detect_column_breaks_from_text(header_text)
            
            if not column_breaks or len(column_breaks) < 3:
                return tables
            
            # Parse data rows (after header, before summary lines)
            table_cells = []
            summary_keywords = ['subtotal', 'tax', 'total due', 'grand total', 'balance due']
            
            for i, line in enumerate(text_lines[header_idx + 1:], header_idx + 1):
                line_text = line.get('text', '').strip()
                
                # Skip empty lines
                if not line_text:
                    continue
                
                # Check if this is a summary line (not a data row)
                line_lower = line_text.lower()
                is_summary = any(kw in line_lower for kw in summary_keywords)
                
                # Also check for numeric patterns: item codes start with letters+numbers like FS-1000
                item_code_match = re.match(r'^[A-Z]{2}-\d+', line_text)
                
                if item_code_match and not is_summary:
                    # This is a data row - parse into columns
                    cells = self._parse_line_into_columns(line_text, column_breaks)
                    if cells and any(c.strip() for c in cells):
                        table_cells.append(cells)
            
            if table_cells and len(table_cells) >= 1:
                # Add header as first row
                header_cells = self._parse_line_into_columns(header_text, column_breaks)
                final_cells = [header_cells] + table_cells if header_cells else table_cells
                
                tables.append({
                    'cells': final_cells,
                    'settings': {'extraction_method': 'text_based'},
                    'header': header_text,
                })
                logger.info(f"  Text-based extraction: {len(table_cells)} line items")
        
        except Exception as e:
            logger.debug(f"Text-based extraction failed: {e}")
        
        return tables
    
    def _detect_column_breaks_from_text(self, header_text: str) -> List[int]:
        """
        Detect column boundaries from header text using whitespace patterns.
        
        Looks for multiple spaces that indicate column separations.
        """
        if not header_text:
            return []
        
        # Find positions of multiple consecutive spaces (column separators)
        column_breaks = [0]  # Start from beginning
        
        # Track positions of significant whitespace gaps
        in_whitespace = False
        current_start = 0
        
        for i, char in enumerate(header_text):
            if char == ' ':
                if not in_whitespace:
                    in_whitespace = True
                    current_start = i
            else:
                if in_whitespace:
                    # End of whitespace sequence
                    gap_size = i - current_start
                    if gap_size >= 2:  # Significant gap indicates column break
                        # Position is in the middle of the gap
                        column_breaks.append(current_start + gap_size // 2)
                    in_whitespace = False
        
        # Add end position
        column_breaks.append(len(header_text))
        
        # Remove duplicates and sort
        column_breaks = sorted(list(set(column_breaks)))
        
        return column_breaks
    
    def _parse_line_into_columns(self, line_text: str, column_breaks: List[int]) -> List[str]:
        """
        Parse a line of text into table cells based on column positions.
        """
        if not line_text or not column_breaks:
            return []
        
        cells = []
        
        for i in range(len(column_breaks) - 1):
            left = column_breaks[i]
            right = column_breaks[i + 1]
            
            # Extract text in this column range
            if left < len(line_text):
                cell_text = line_text[left:right].strip()
            else:
                cell_text = ''
            
            cells.append(cell_text)
        
        return cells
    
    def _extract_tables_pattern_based(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract tables using pattern-based line analysis.
        
        Useful for invoices where text flows continuously without clear table structure.
        """
        tables = []
        
        if not text:
            return tables
        
        lines = text.strip().split('\n')
        
        if len(lines) < 3:
            return tables
        
        # Find the header line
        header_keywords = ['description', 'qty', 'quantity', 'price', 'amount', 'total', 'code']
        header_idx = -1
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            matches = sum(1 for kw in header_keywords if kw in line_lower)
            if matches >= 2:
                header_idx = i
                break
        
        if header_idx < 0:
            # No clear header, try to infer structure
            header_idx = 0
        
        # Extract table rows
        table_rows = []
        
        for i, line in enumerate(lines[header_idx:], start=header_idx):
            line = line.strip()
            if not line:
                continue
            
            # Skip summary lines
            line_lower = line.lower()
            if any(re.match(pattern, line_lower) for pattern in self.SUMMARY_PATTERNS):
                continue
            
            # Try to parse as line item
            parsed_row = self._parse_line_item(line)
            
            if parsed_row:
                table_rows.append(parsed_row)
        
        if table_rows:
            # Add header row
            header_row = [w.strip() for w in lines[header_idx].split() if w.strip()]
            all_rows = [header_row] + table_rows
            
            tables.append({
                'cells': all_rows,
                'settings': {'extraction_method': 'pattern'},
                'pattern_extracted': True,
            })
            logger.info(f"  Pattern extraction: {len(table_rows)} line items found")
        
        return tables
    
    def _parse_line_item(self, line: str) -> Optional[List[str]]:
        """
        Parse a single line as a line item.
        
        Attempts to identify and extract: item_code, description, quantity, unit_price, total
        """
        # Try different parsing strategies
        
        # Strategy 1: Look for multiple numeric values
        numbers = re.findall(r'[\d,]+\.?\d*', line)
        
        if len(numbers) >= 2:
            # Clean numbers
            cleaned_numbers = [n.replace(',', '') for n in numbers]
            
            # Remove the line item line itself if found (common in invoices)
            line_item_text = line
            
            # Try to extract components
            parts = line.strip().split()
            
            if len(parts) >= 3:
                # Assume format: [description words...] [qty] [price] [total]
                # or: [code] [description words...] [qty] [price] [total]
                
                # Check if first part looks like an item code
                if re.match(r'^[A-Z0-9-]{3,}$', parts[0]):
                    item_code = parts[0]
                    description = ' '.join(parts[1:-3]) if len(parts) > 4 else parts[1] if len(parts) > 1 else ''
                    remaining = parts[-3:] if len(parts) > 4 else parts[-2:] if len(parts) > 2 else []
                else:
                    item_code = ''
                    description = ' '.join(parts[:-3]) if len(parts) > 3 else parts[0]
                    remaining = parts[-3:] if len(parts) > 3 else parts[-2:] if len(parts) > 2 else []
                
                # Ensure we have numeric values
                if len(remaining) >= 2:
                    try:
                        qty = float(remaining[0].replace(',', ''))
                        total = float(remaining[-1].replace(',', ''))
                        
                        # Calculate unit price if we have both
                        unit_price = total / qty if qty > 0 else 0
                        
                        return [
                            item_code or '',
                            description,
                            str(qty),
                            f'{unit_price:.2f}',
                            str(total)
                        ]
                    except (ValueError, IndexError):
                        pass
        
        # Strategy 2: Look for structured patterns
        pattern_match = re.match(
            r'^([A-Z0-9-]+)\s+(.+?)\s+(\d+(?:[.,]\d+)?)\s+\$?([\d,]+\.?\d*)\s+\$?([\d,]+\.?\d*)$',
            line
        )
        
        if pattern_match:
            return list(pattern_match.groups())
        
        # If nothing worked, return None
        return None
    
    def _ocr_fallback_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Full OCR fallback for problematic PDFs."""
        result = {
            'text_content': '',
            'tables': [],
            'metadata': {},
            'page_count': 0,
            'file_name': Path(pdf_path).name,
            'file_type': 'pdf',
            'extraction_method': 'ocr_fallback',
        }
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                result['page_count'] = len(pdf.pages)
                
                all_text_parts = []
                all_tables = []
                
                for page_num, page in enumerate(pdf.pages, start=1):
                    logger.info(f"OCR processing page {page_num}/{len(pdf.pages)}")
                    
                    page_image = page.to_image(resolution=300)
                    page_pil = page_image.original
                    
                    text = pytesseract.image_to_string(page_pil, lang=self.ocr_language)
                    
                    if text:
                        all_text_parts.append(f"[Page {page_num}]\n{text}")
                    
                    tables = self._extract_tables_pattern_based(text)
                    for table in tables:
                        table['page_number'] = page_num
                        all_tables.append(table)
                    
                    page_meta = self._extract_page_metadata(text, page_num)
                    result['metadata'].update(page_meta)
                
                result['text_content'] = '\n\n'.join(all_text_parts)
                result['tables'] = all_tables
                result['metadata']['ocr_fallback'] = True
        
        except Exception as e:
            logger.error(f"OCR fallback failed: {e}")
            raise
        
        result['metadata'].update(
            self._extract_invoice_metadata(result['text_content'])
        )
        
        return result
    
    def _ocr_page(self, page) -> str:
        """Perform OCR on a single PDF page."""
        try:
            page_image = page.to_image(resolution=300)
            page_pil = page_image.original
            
            text = pytesseract.image_to_string(page_pil, lang=self.ocr_language)
            return text if text else ''
        
        except Exception as e:
            logger.warning(f"OCR failed for page: {e}")
            return ''
    
    def _assess_text_quality(self, text: str) -> float:
        """Assess the quality of extracted text."""
        if not text:
            return 0.0
        
        score = 0.0
        text_lower = text.lower()
        
        invoice_indicators = ['invoice', 'total', 'amount', 'date', 'quantity', 'price', 'subtotal', 'tax']
        found_indicators = sum(1 for ind in invoice_indicators if ind in text_lower)
        score += (found_indicators / len(invoice_indicators)) * 0.5
        
        word_count = len(text.split())
        if 50 < word_count < 5000:
            score += 0.3
        elif word_count > 0:
            score += 0.1
        
        short_lines = sum(1 for line in text.split('\n') if len(line) < 5)
        total_lines = len(text.split('\n'))
        if total_lines > 0 and short_lines / total_lines < 0.5:
            score += 0.2
        
        return min(score, 1.0)
    
    def _ocr_extract_tables(self, text: str) -> List[Dict[str, Any]]:
        """Extract table-like structures from OCR text."""
        return self._extract_tables_pattern_based(text)
    
    def _split_ocr_row(self, line: str) -> List[str]:
        """Split an OCR text line into table cells."""
        cells = []
        
        split_strategies = [
            lambda s: s.split('  '),
            lambda s: s.split('\t'),
            lambda s: s.split(' | '),
            lambda s: s.split('|'),
        ]
        
        for strategy in split_strategies:
            parts = [p.strip() for p in strategy(line) if p.strip()]
            if len(parts) >= 3:
                return parts
        
        return [line] if line else []
    
    def _extract_invoice_metadata(self, text: str) -> Dict[str, Any]:
        """Extract invoice header metadata from text content."""
        metadata = {}
        
        # Extract invoice ID - look for pattern at start of line or after "Invoice #:"
        invoice_patterns = [
            r'Invoice\s*#?\s*:?\s*(INV-[A-Z0-9-]+)',  # Invoice #: INV-98448
            r'Invoice\s*Number\s*:?\s*(INV-[A-Z0-9-]+)',  # Invoice Number: INV-98448
            r'(?:^|\s)(INV-[A-Z0-9-]+)',  # INV-XXXXX pattern at word boundary
        ]
        
        for pattern in invoice_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                metadata['invoice_id'] = match.group(1).strip()
                break
        
        for pattern in self.DATE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                parsed_date = self._parse_date(date_str)
                if parsed_date:
                    metadata['invoice_date'] = parsed_date
                    break
        
        due_date_pattern = r'Due\s*Date\s*:?\s*' + self.DATE_PATTERNS[0].replace('\\d', '\\d')
        match = re.search(due_date_pattern, text, re.IGNORECASE)
        if match:
            date_str = match.group(1)
            parsed_date = self._parse_date(date_str)
            if parsed_date:
                metadata['due_date'] = parsed_date
        
        subtotal_match = re.search(r'Sub\s*[-]?total\s*:?\s*\$?([\d,]+\.?\d*)', text, re.IGNORECASE)
        if subtotal_match:
            metadata['subtotal'] = float(subtotal_match.group(1).replace(',', ''))
        
        tax_match = re.search(r'Tax\s*(?:\(\d+%\))?\s*:?\s*\$?([\d,]+\.?\d*)|VAT\s*:?\s*\$?([\d,]+\.?\d*)', text, re.IGNORECASE)
        if tax_match:
            tax_str = tax_match.group(1) or tax_match.group(2)
            metadata['tax'] = float(tax_str.replace(',', ''))
        
        # Extract total - look for Grand Total or Total at end, prefer larger value
        total_patterns = [
            r'Grand\s*Total\s*:?\s*\$?([\d,]+\.?\d*)',  # Grand Total first (most specific)
            r'Total\s*(?!.*Sub\s*Total)(?!.*Tax)[\s\$]*:?\s*\$?([\d,]+\.?\d*)',  # Total excluding subtotal/tax context
        ]
        
        totals_found = []
        for pattern in total_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    value = float(match.replace(',', ''))
                    totals_found.append(value)
                except ValueError:
                    continue
        
        # Use the largest total value found (usually the grand total)
        if totals_found:
            metadata['total'] = max(totals_found)
        
        if '$' in text or 'USD' in text:
            metadata['currency'] = 'USD'
        elif '€' in text or 'EUR' in text:
            metadata['currency'] = 'EUR'
        elif '£' in text or 'GBP' in text:
            metadata['currency'] = 'GBP'
        
        return metadata
    
    def _extract_page_metadata(self, text: str, page_num: int) -> Dict[str, Any]:
        """Extract metadata specific to a page."""
        return {}
    
    def _parse_date(self, date_str: str) -> Optional[str]:
        """Parse a date string into YYYY-MM-DD format."""
        date_formats = [
            '%Y-%m-%d', '%Y/%m/%d', '%m-%d-%Y', '%m/%d/%Y',
            '%d-%m-%Y', '%d/%m/%Y', '%B %d, %Y', '%b %d, %Y',
            '%d %B %Y', '%d %b %Y',
        ]
        
        for fmt in date_formats:
            try:
                parsed = datetime.strptime(date_str.strip(), fmt)
                return parsed.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        return None
    
    def detect_line_item_tables(self, tables: List[Dict]) -> List[Dict]:
        """Identify which tables contain line item data."""
        line_item_tables = []
        
        for table in tables:
            cells = table.get('cells', [])
            if not cells:
                continue
            
            header_row = cells[0] if cells else []
            header_text = ' '.join(str(cell) for cell in header_row if cell)
            header_lower = header_text.lower()
            
            has_quantity = any(ind in header_lower for ind in ['quantity', 'qty'])
            has_amount = any(ind in header_lower for ind in ['amount', 'total', 'price'])
            
            if has_quantity and has_amount:
                line_item_tables.append(table)
        
        return line_item_tables
    
    def is_image_file(self, file_path: str) -> bool:
        """Check if the file is a supported image format."""
        return Path(file_path).suffix.lower() in self.SUPPORTED_IMAGE_EXTENSIONS
    
    def is_pdf_file(self, file_path: str) -> bool:
        """Check if the file is a PDF."""
        return Path(file_path).suffix.lower() in self.SUPPORTED_PDF_EXTENSIONS


def check_tesseract_installation() -> Dict[str, Any]:
    """Check Tesseract OCR installation and configuration."""
    result = {
        'installed': False,
        'version': None,
        'path': None,
        'languages': [],
        'error': None,
    }
    
    try:
        version = pytesseract.get_tesseract_version()
        result['installed'] = True
        result['version'] = str(version)
        result['path'] = pytesseract.pytesseract.tesseract_cmd
        result['languages'] = pytesseract.get_languages()
    except pytesseract.TesseractNotFoundError:
        result['error'] = "Tesseract executable not found"
    except Exception as e:
        result['error'] = str(e)
    
    return result


def install_tesseract_instructions() -> str:
    """Return platform-specific Tesseract installation instructions."""
    instructions = """
Tesseract OCR Installation Instructions:

=== Windows ===
1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer
3. Default path: C:\\Program Files\\Tesseract-OCR\\tesseract.exe

=== macOS ===
brew install tesseract

=== Linux (Ubuntu/Debian) ===
sudo apt-get install tesseract-ocr

=== Verification ===
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
"""
    return instructions
