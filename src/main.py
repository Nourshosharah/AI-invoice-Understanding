#!/usr/bin/env python3
"""
Main Entry Point for Invoice Understanding System

CLI interface for invoice ingestion, querying, and management.
Supports both PDF documents and image files (PNG, JPG, etc.) with OCR.
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from extraction.invoice_parser import InvoiceParser, check_tesseract_installation
from extraction.line_item_extractor import LineItemExtractor
from extraction.llm_fixer import LLMFixer
from indexing.vector_store import VectorStore
from retrieval.query_router import QueryRouter
from retrieval.answer_generator import AnswerGenerator
from utils.models import ExtractionResult, InvoiceMetadata, LineItem


class InvoiceQASystem:
    """
    Main system orchestrator for invoice understanding.
    
    Coordinates:
    - Invoice ingestion (PDF and image files)
    - Line item extraction
    - Indexing in vector store
    - Query processing and answer generation
    """
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    def __init__(
        self,
        qdrant_dir: str = "data/qdrant_db",
        extracted_dir: str = "data/extracted",
        use_llm: bool = True,
        tesseract_path: str = None,
        use_ocr_fallback: bool = True
    ):
        """
        Initialize the invoice Q&A system.
        
        Args:
            qdrant_dir: Directory for Qdrant persistence
            extracted_dir: Directory for extracted JSON files
            use_llm: Whether to use LLM for extraction fixing
            tesseract_path: Optional path to Tesseract OCR executable
            use_ocr_fallback: Whether to use OCR fallback for PDFs
        """
        self.extracted_dir = Path(extracted_dir)
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.parser = InvoiceParser(
            tesseract_path=tesseract_path,
            use_ocr_fallback=use_ocr_fallback
        )
        self.extractor = LineItemExtractor()
        self.fixer = LLMFixer() if use_llm else None
        self.vector_store = VectorStore(persist_dir=qdrant_dir)
        self.router = QueryRouter()
        self.generator = AnswerGenerator()
    
    def ingest(
        self,
        file_path: str,
        invoice_id: str = None,
        save_json: bool = True
    ) -> ExtractionResult:
        """
        Ingest and process an invoice document (PDF or image).
        
        Args:
            file_path: Path to PDF or image file
            invoice_id: Optional custom invoice ID
            save_json: Whether to save extracted data as JSON
            
        Returns:
            ExtractionResult with all extracted data
        """
        file_ext = Path(file_path).suffix.lower()
        
        # Determine file type for display
        if file_ext in ['.pdf']:
            file_type = "PDF"
        elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
            file_type = "Image"
        else:
            file_type = "Document"
        
        print(f"\nProcessing {file_type}: {file_path}")
        
        # Check OCR availability for images
        if file_ext not in ['.pdf'] and not self.parser.ocr_available:
            print("  Warning: Tesseract OCR not available. Image processing may fail.")
            print("  Install Tesseract or set the path with --tesseract-path")
        
        # Step 1: Parse document
        print(f"  [1/4] Parsing document...")
        parsed = self.parser.parse(file_path)
        
        file_type_info = parsed.get('file_type', 'unknown')
        if parsed.get('metadata', {}).get('ocr_fallback'):
            print("    Note: Used OCR fallback extraction")
        if parsed.get('metadata', {}).get('ocr_used_pages'):
            print(f"    Note: OCR used on pages: {parsed['metadata']['ocr_used_pages']}")
        
        # Determine invoice ID
        inv_id = invoice_id or parsed['metadata'].get('invoice_id')
        if not inv_id:
            inv_id = Path(file_path).stem.replace(' ', '_').replace('-', '_').upper()
        
        # Step 2: Extract line items
        print(f"  [2/4] Extracting line items...")
        tables = self.parser.detect_line_item_tables(parsed['tables'])
        items = self.extractor.extract(tables, parsed['metadata'])
        
        # Step 3: LLM fixing (optional)
        if self.fixer and items:
            print(f"  [3/4] Applying LLM fixes...")
            page_text = parsed['text_content']
            invoice_date = parsed['metadata'].get('invoice_date', '')
            items = self.fixer.fix_extraction(items, page_text, invoice_date)
        else:
            print(f"  [3/4] Skipping LLM fixes (disabled)")
        
        # Step 4: Index in vector store
        print(f"  [4/4] Indexing in vector store...")
        
        # Create LineItem objects for chunking
        line_item_objects = []
        for item in items:
            line_item = LineItem(
                item_code=item.get('item_code'),
                description=item.get('description'),
                delivery_date=item.get('delivery_date'),
                quantity=item.get('quantity'),
                unit_price=item.get('unit_price'),
                total_amount=item.get('total_amount'),
                page_number=item.get('page_number'),
                line_number=item.get('line_number'),
            )
            line_item_objects.append(line_item)
        
        # Create chunks and add to vector store
        chunks = []
        for idx, item in enumerate(line_item_objects):
            chunks.append({
                "content": item.to_search_text(),
                "metadata": item.to_metadata(inv_id),
            })
        
        if chunks:
            self.vector_store.add_items(chunks, inv_id)
        
        # Save JSON
        if save_json:
            json_path = self.extracted_dir / f"{inv_id}.json"
            metadata = InvoiceMetadata(
                invoice_id=inv_id,
                invoice_date=parsed['metadata'].get('invoice_date'),
                due_date=parsed['metadata'].get('due_date'),
                company_name=parsed['metadata'].get('company_name'),
                company_address=parsed['metadata'].get('company_address'),
                company_email=parsed['metadata'].get('company_email'),
                customer_name=parsed['metadata'].get('customer_name'),
                customer_address=parsed['metadata'].get('customer_address'),
                subtotal=parsed['metadata'].get('subtotal'),
                tax=parsed['metadata'].get('tax'),
                total=parsed['metadata'].get('total'),
                currency=parsed['metadata'].get('currency', 'USD'),
            )
            
            result = ExtractionResult(
                invoice_id=inv_id,
                file_path=file_path,
                file_name=Path(file_path).name,
                total_pages=parsed['page_count'],
                line_items=line_item_objects,
                metadata=metadata,
            )
            
            with open(json_path, 'w') as f:
                f.write(result.model_dump_json(indent=2))
            print(f"  Saved extraction to: {json_path}")
        
        # Summary
        print(f"\n✓ Complete!")
        print(f"  Invoice ID: {inv_id}")
        print(f"  File Type: {file_type_info}")
        print(f"  Pages: {parsed['page_count']}")
        print(f"  Line Items: {len(items)}")
        
        return result
    
    def ingest_directory(self, directory: str) -> list:
        """
        Ingest all supported documents (PDF and images) in a directory.
        
        Args:
            directory: Directory path containing documents
            
        Returns:
            List of ExtractionResult objects
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise ValueError(f"Directory not found: {directory}")
        
        results = []
        
        # Find all supported files
        supported_files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            supported_files.extend(dir_path.glob(f"*{ext}"))
            supported_files.extend(dir_path.glob(f"*{ext.upper()}"))
        
        # Remove duplicates and sort
        supported_files = sorted(set(supported_files))
        
        print(f"\nFound {len(supported_files)} supported documents in {directory}")
        
        for file_path in supported_files:
            try:
                result = self.ingest(str(file_path))
                results.append(result)
            except Exception as e:
                print(f"  Error processing {file_path.name}: {e}")
        
        print(f"\n✓ Completed ingesting {len(results)} documents")
        return results
    
    def query(self, question: str, max_results: int = 20) -> dict:
        """
        Process a query and generate an answer.
        
        Args:
            question: User question
            max_results: Maximum items to retrieve
            
        Returns:
            Answer result dict
        """
        print(f"\nQuery: {question}")
        
        # Step 1: Route query
        print("  [1/3] Routing query...")
        routing = self.router.route(question)
        print(f"    Type: {routing['query_type'].value}")
        if routing['filters']:
            print(f"    Filters: {routing['filters']}")
        
        # Step 2: Retrieve items
        print("  [2/3] Retrieving items...")
        results = self.vector_store.query(
            query_text=routing['query_text'] or None,
            filters=routing['filters'],
            n_results=max_results,
        )
        print(f"    Found {len(results)} matching items")
        
        # Step 3: Generate answer
        print("  [3/3] Generating answer...")
        answer = self.generator.generate(
            question,
            results,
            routing['query_type'].value,
        )
        
        return answer
    
    def list_invoices(self) -> list:
        """
        List all indexed invoices.
        
        Returns:
            List of invoice summary dicts
        """
        return self.vector_store.list_invoices()
    
    def get_invoice(self, invoice_id: str, page: int = None) -> list:
        """
        Get all items for a specific invoice.
        
        Args:
            invoice_id: Invoice identifier
            page: Optional page number filter
            
        Returns:
            List of line items
        """
        return self.vector_store.get_invoice_items(invoice_id, page_number=page)


def check_ocr_status():
    """Check and display OCR status."""
    status = check_tesseract_installation()
    
    print("OCR Status:")
    print("-" * 40)
    if status['installed']:
        print(f"✓ Tesseract OCR is installed")
        print(f"  Version: {status['version']}")
        print(f"  Path: {status['path']}")
        print(f"  Languages: {', '.join(status.get('languages', []))}")
    else:
        print("✗ Tesseract OCR is not installed")
        if status['error']:
            print(f"  Error: {status['error']}")
        print("\nTo process image files, install Tesseract:")
        print("  Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        print("  macOS: brew install tesseract")
        print("  Linux: sudo apt-get install tesseract-ocr")
    print("-" * 40)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Invoice Understanding System - Ingest, Index, and Query Invoices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest a PDF invoice
  python -m src.main ingest invoice.pdf
  
  # Ingest an image invoice
  python -m src.main ingest invoice.png
  python -m src.main ingest invoice.jpg
  
  # Batch ingest all documents in a directory
  python -m src.main ingest ./invoices/ --batch
  
  # Query the system
  python -m src.main query "Show all items on page 3"
  python -m src.main query "What maintenance items were purchased?"
  
  # Interactive mode
  python -m src.main interactive

Supported formats:
  PDFs: .pdf
  Images: .png, .jpg, .jpeg, .bmp, .tiff, .tif
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest an invoice (PDF or image) or directory of invoices")
    ingest_parser.add_argument("file_path", help="Path to PDF/image file OR directory containing invoices")
    ingest_parser.add_argument(
        "--invoice-id",
        help="Custom invoice ID (only for single file)",
    )
    ingest_parser.add_argument(
        "--output-dir",
        default="data/extracted",
        help="Directory for JSON output",
    )
    ingest_parser.add_argument(
        "--tesseract-path",
        help="Path to Tesseract executable (for image processing)",
    )
    ingest_parser.add_argument(
        "--no-ocr-fallback",
        action="store_true",
        help="Disable OCR fallback for PDFs",
    )
    
    # List command
    list_parser = subparsers.add_parser("list", help="List indexed invoices")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument(
        "--max-results",
        type=int,
        default=3,
        help="Maximum results to retrieve",
    )
    
    # Interactive command
    interactive_parser = subparsers.add_parser(
        "interactive", help="Start interactive query session"
    )
    interactive_parser.add_argument(
        "--tesseract-path",
        help="Path to Tesseract executable",
    )
    
    # Get command
    get_parser = subparsers.add_parser("get", help="Get invoice items")
    get_parser.add_argument("invoice_id", help="Invoice ID")
    get_parser.add_argument("--page", type=int, help="Page number filter")
    
    # OCR status command
    ocr_parser = subparsers.add_parser(
        "ocr-status", help="Check Tesseract OCR installation status"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Handle OCR status command
    if args.command == "ocr-status":
        check_ocr_status()
        sys.exit(0)
    
    # Initialize system with Tesseract path if provided
    tesseract_path = getattr(args, 'tesseract_path', None)
    use_ocr_fallback = not getattr(args, 'no_ocr_fallback', False)
    
    system = InvoiceQASystem(
        tesseract_path=tesseract_path,
        use_ocr_fallback=use_ocr_fallback
    )
    
    try:
        if args.command == "ingest":
            import os
            path = args.file_path
            
            if os.path.isdir(path):
                # It's a directory - ingest all supported files
                print(f"[INFO] Detected directory: {path}")
                system.ingest_directory(path)
            elif os.path.isfile(path):
                # It's a single file - ingest it
                print(f"[INFO] Detected file: {path}")
                system.ingest(
                    path,
                    invoice_id=args.invoice_id,
                    save_json=True,
                )
            else:
                print(f"[ERROR] Path not found: {path}")
                sys.exit(1)
        
        elif args.command == "list":
            invoices = system.list_invoices()
            if not invoices:
                print("\nNo invoices indexed yet.")
                print("Use 'python -m src.main ingest <file_path>' to add invoices.")
                print("Supports PDF and image files (PNG, JPG, etc.)")
            else:
                print("\nIndexed Invoices:")
                print("-" * 60)
                for inv in invoices:
                    print(
                        f"{inv['invoice_id']}: "
                        f"{inv['line_item_count']} items, "
                        f"{inv['page_count']} pages, "
                        f"${inv['total_amount']:,.2f}"
                    )
                print("-" * 60)
                print(f"Total: {len(invoices)} invoices")
        
        elif args.command == "query":
            result = system.query(args.question, max_results=args.max_results)
            
            print("\n" + "=" * 60)
            print("ANSWER:")
            print("=" * 60)
            print(result['answer'])
            print("=" * 60)
            
            if result.get('sources'):
                print("\nSources:")
                for source in result['sources']:
                    inv = source.get('invoice_id', 'N/A')
                    page = source.get('page_number', 'N/A')
                    line = source.get('line_number', 'N/A')
                    print(f"  - {inv}, page {page}, line {line}")
        
        elif args.command == "interactive":
            print("\n" + "=" * 60)
            print("INVOICE UNDERSTANDING SYSTEM - Interactive Mode")
            print("=" * 60)
            print("Commands:")
            print("  'exit' or 'quit' - Exit")
            print("  'list' - List indexed invoices")
            print("  'status' - Show OCR status")
            print("  'help' - Show this help")
            print("=" * 60)
            
            while True:
                try:
                    question = input("\n> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break
                
                if not question:
                    continue
                
                if question.lower() in ['exit', 'quit']:
                    print("Goodbye!")
                    break
                
                if question.lower() == 'list':
                    invoices = system.list_invoices()
                    if not invoices:
                        print("No invoices indexed.")
                    else:
                        for inv in invoices:
                            print(f"  {inv['invoice_id']}: {inv['line_item_count']} items")
                    continue
                
                if question.lower() == 'status':
                    check_ocr_status()
                    continue
                
                if question.lower() == 'help':
                    print("Ask questions like:")
                    print("  - 'Show all items on page 3'")
                    print("  - 'What maintenance items were purchased?'")
                    print("  - 'Items from March 2024'")
                    print("  - 'Total amount for invoice INV-001'")
                    print("\nSupported file formats:")
                    print("  - PDF documents (.pdf)")
                    print("  - Images (.png, .jpg, .jpeg, .bmp, .tiff)")
                    continue
                
                try:
                    result = system.query(question)
                    
                    print("\n" + "-" * 40)
                    print(result['answer'])
                    print("-" * 40)
                    
                    if result.get('sources'):
                        print(f"\n({len(result['sources'])} sources)")
                except Exception as e:
                    print(f"Error: {e}")
        
        elif args.command == "get":
            items = system.get_invoice(args.invoice_id, page=args.page)
            
            if not items:
                print(f"No items found for invoice: {args.invoice_id}")
            else:
                print(f"\nItems for {args.invoice_id}:")
                for item in items:
                    meta = item['metadata']
                    print(
                        f"  Page {meta.get('page_number')}, "
                        f"Line {meta.get('line_number')}: "
                        f"{meta.get('description', 'N/A')[:50]}"
                    )
    
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
