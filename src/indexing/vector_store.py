"""
Vector Store for Invoice Line Items

This module provides Qdrant integration for storing and retrieving
invoice line items with semantic search and metadata filtering.

Advantages of Qdrant over ChromaDB:
- Better performance for large-scale deployments
- Native support for distributed deployments
- Rich filtering capabilities with must/should logic
- Built-in quantization for memory efficiency
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
)
from qdrant_client.http.exceptions import UnexpectedResponse
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from datetime import date
from dotenv import load_dotenv
import os
import uuid


class VectorStore:
    """
    Qdrant-based vector store for invoice line items.
    
    Provides:
    - Semantic search using embeddings
    - Metadata filtering for structured queries
    - Hybrid search combining both approaches
    - Efficient pagination and scrolling
    """
    
    COLLECTION_NAME = "invoice_line_items"
    VECTOR_SIZE = 384  # all-MiniLM-L6-v2 embedding dimension
    
    def __init__(
        self,
        persist_dir: str = "data/qdrant_db",
        embedding_function=None,
        use_in_memory: bool = False
    ):
        """
        Initialize the vector store.
        
        Args:
            persist_dir: Directory for persistent storage
            embedding_function: Function to generate embeddings
            use_in_memory: Use in-memory storage (for testing)
        """
        load_dotenv()
        
        self.persist_dir = Path(persist_dir)
        self.embedding_function = embedding_function
        
        if use_in_memory:
            # In-memory mode for testing
            self.client = QdrantClient(":memory:")
            print(f"  [DEBUG] Using in-memory Qdrant client")
        else:
            # Persistent mode
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(self.persist_dir))
            print(f"  [DEBUG] Using persistent Qdrant client at: {self.persist_dir}")
        
        print(f"  [DEBUG] QdrantClient type: {type(self.client)}")
        print(f"  [DEBUG] QdrantClient version: {getattr(self.client, '__version__', 'unknown')}")
        
        # Ensure collection exists
        self._ensure_collection()
    
    def _ensure_collection(self):
        """
        Create collection if it doesn't exist.
        Qdrant requires explicit collection creation with vector config.
        """
        try:
            self.client.get_collection(collection_name=self.COLLECTION_NAME)
            print(f"Collection '{self.COLLECTION_NAME}' already exists")
        except (UnexpectedResponse, ValueError) as e:
            # Collection doesn't exist, create it
            # Check if it's actually a "not found" error
            if "not found" in str(e).lower() or isinstance(e, ValueError):
                self.client.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=self.VECTOR_SIZE,
                        distance=Distance.COSINE,
                    ),
                )
                print(f"Created collection: {self.COLLECTION_NAME}")
            else:
                # Re-raise if it's a different UnexpectedResponse
                raise
    
    def _get_embedding_function(self, embedding_function=None):
        """
        Get the embedding function to use.
        
        Args:
            embedding_function: Optional override function
            
        Returns:
            Callable that generates embeddings
        """
        emb_func = embedding_function or self.embedding_function
        
        if emb_func is not None:
            return emb_func
        
        # Use sentence-transformers as default
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        def embed_texts(texts: List[str]) -> List[List[float]]:
            return model.encode(texts).tolist()
        
        return embed_texts
    
    def _build_qdrant_filter(
        self,
        filters: Dict = None,
        invoice_id: str = None
    ) -> Optional[Filter]:
        """
        Build Qdrant filter from filter parameters.
        
        Qdrant uses Filter with Must (AND) and Should (OR) conditions.
        
        Args:
            filters: Metadata filters dict
            invoice_id: Optional invoice ID filter
            
        Returns:
            Qdrant Filter object or None
        """
        conditions = []
        
        if filters:
            for key, value in filters.items():
                if isinstance(value, dict):
                    # Range filter (e.g., {"$gte": 100})
                    range_filter = self._build_range_filter(key, value)
                    if range_filter:
                        conditions.append(range_filter)
                else:
                    # Simple equality
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
        
        if invoice_id:
            conditions.append(
                FieldCondition(key="invoice_id", match=MatchValue(value=invoice_id))
            )
        
        if not conditions:
            return None
        
        return Filter(must=conditions)
    
    def _build_range_filter(
        self,
        field: str,
        conditions: Dict
    ) -> Optional[FieldCondition]:
        """
        Build Qdrant range filter from range conditions.
        
        Args:
            field: Field name
            conditions: Range conditions (e.g., {"$gte": 100, "$lte": 200})
            
        Returns:
            FieldCondition with Range or None
        """
        range_params = {}
        
        if "$gte" in conditions:
            range_params["gte"] = conditions["$gte"]
        if "$gt" in conditions:
            range_params["gt"] = conditions["$gt"]
        if "$lte" in conditions:
            range_params["lte"] = conditions["$lte"]
        if "$lt" in conditions:
            range_params["lt"] = conditions["$lt"]
        
        if range_params:
            return FieldCondition(key=field, range=Range(**range_params))
        
        return None
    
    def _get_score(self, point) -> float:
        """
        Safely get score from a Qdrant point object.
        Uses try-except to avoid triggering Qdrant's custom __getattr__ on immutable Record objects.

        Args:
            point: Qdrant point object

        Returns:
            Score value or 1.0 if not available
        """
        # Use try-except to avoid triggering Qdrant's custom __getattr__
        # This is necessary because Record objects in old Qdrant versions are immutable
        # and their __getattr__ raises AttributeError for any non-existent attribute
        try:
            return point.score
        except (AttributeError, TypeError):
            return 1.0
    
    def add_items(
        self,
        items: List[Dict],
        invoice_id: str,
        embedding_function=None
    ):
        """
        Add line item chunks to the vector store.
        
        Args:
            items: List of item dicts with 'content' and 'metadata'
            invoice_id: Invoice identifier for grouping
            embedding_function: Function to generate embeddings
        """
        if not items:
            return
        
        emb_func = self._get_embedding_function(embedding_function)
        
        # Prepare data - extract texts for embedding
        texts = [item['content'] for item in items]
        
        # Build payloads with metadata
        payloads = []
        for item in items:
            payload = {
                "content": item['content'],
                **item['metadata']
            }
            payloads.append(payload)
        
        # Generate embeddings
        print(f"Generating embeddings for {len(items)} items...")
        vectors = emb_func(texts)

        # Create points for Qdrant with UNIQUE IDs (UUID based on invoice_id + line_number)
        # This prevents overwriting when new invoices are ingested
        points = []
        for item, vector in zip(items, vectors):
            # Create unique ID using invoice_id and line_number
            inv_id = item.get('metadata', {}).get('invoice_id', invoice_id)
            line_num = item.get('metadata', {}).get('line_number', 0)
            # Generate deterministic UUID from the unique string ID
            unique_str_id = f"{inv_id}_{line_num}"
            unique_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, unique_str_id)
            
            points.append(
                PointStruct(
                    id=unique_uuid,  # Use UUID (compatible with Qdrant)
                    vector=vector,
                    payload={"content": item['content'], **item['metadata']}
                )
            )

        # Upload to Qdrant - only pass points, IDs are embedded in PointStruct
        self.client.upload_points(
            collection_name=self.COLLECTION_NAME,
            points=points,
            wait=True  # Wait for upload to complete
        )
        
        print(f"Added {len(items)} items to vector store")
    
    def query(
        self,
        query_text: str = None,
        filters: Dict = None,
        n_results: int = 10,
        invoice_id: str = None
    ) -> List[Dict]:
        """
        Query line items with semantic search and/or filters.
        
        Args:
            query_text: Text for semantic similarity search
            filters: Metadata filters (e.g., {"page_number": 3})
            n_results: Maximum number of results
            invoice_id: Filter by specific invoice
            
        Returns:
            List of matching items with metadata and scores
        """
        emb_func = self._get_embedding_function()
        
        # Build filter
        qdrant_filter = self._build_qdrant_filter(filters, invoice_id)
        
        # Generate query vector only if query_text is meaningful
        # For empty or very short queries, skip semantic search entirely
        is_meaningful_query = query_text and len(query_text.strip()) >= 3
        query_vector = emb_func([query_text])[0] if is_meaningful_query else None
        
        # If query_vector is None (no meaningful semantic search), use filter-only search
        if query_vector is None:
            print(f"  [DEBUG] No meaningful semantic query - using filter-only search")
            results = self._scroll_search(
                filter=qdrant_filter,
                limit=n_results
            )
        else:
            # Semantic search with optional filters
            # Try semantic search using query_points() method (Qdrant 1.16+)
            try:
                print(f"  [DEBUG] Attempting semantic search with query_points() method...")
                results = self.client.query_points(
                    collection_name=self.COLLECTION_NAME,
                    query=query_vector,
                    query_filter=qdrant_filter,
                    limit=n_results,
                    score_threshold=0.2,  # Only return items with >20% similarity
                    with_payload=True,
                    with_vectors=False,
                )
                # query_points returns a result object, extract the points
                if hasattr(results, 'result'):
                    results = results.result.points
                else:
                    results = getattr(results, 'points', results)
                print(f"  [DEBUG] Semantic search found {len(results)} results (threshold=0.2).")
                # Debug: show scores and all payload keys
                for i, point in enumerate(results[:5]):
                    score = getattr(point, 'score', 'N/A')
                    payload_keys = list(point.payload.keys())
                    desc = point.payload.get('description', point.payload.get('content', 'N/A')[:50])
                    print(f"    [DEBUG] Result {i+1}: score={score:.3f}, keys={payload_keys}")
                    print(f"             content/preview: '{desc}'")
            except (AttributeError, TypeError) as e:
                print(f"  [DEBUG] query_points() method failed: {type(e).__name__}: {e}")
                print(f"  [DEBUG] Falling back to filter-based scroll search...")
                results, _ = self.client.scroll(
                    collection_name=self.COLLECTION_NAME,
                    scroll_filter=qdrant_filter,
                    limit=n_results * 2,
                    with_payload=True,
                    with_vectors=False,
                )
                
                if results:
                    results = results[:n_results]
                else:
                    results = []
                print(f"  [DEBUG] Filter-based search found {len(results)} results.")
        
        return self._format_query_results(results)
    
    def _scroll_search(
        self,
        filter: Filter = None,
        limit: int = 10,
        offset: int = 0
    ) -> List:
        """
        Perform scroll search for filter-only queries.
        
        Args:
            filter: Qdrant filter
            limit: Number of results
            offset: Pagination offset
            
        Returns:
            List of scroll results
        """
        results, _ = self.client.scroll(
            collection_name=self.COLLECTION_NAME,
            scroll_filter=filter,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        return results
    
    def list_invoices(self) -> List[Dict]:
        """
        List all indexed invoices with summary information.
        
        Returns:
            List of invoice summaries
        """
        results, _ = self.client.scroll(
            collection_name=self.COLLECTION_NAME,
            scroll_filter=None,
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )
        
        invoice_summaries = {}
        
        for point in results:
            payload = point.payload
            inv_id = payload.get('invoice_id')
            
            if inv_id not in invoice_summaries:
                invoice_summaries[inv_id] = {
                    'invoice_id': inv_id,
                    'page_numbers': set(),
                    'item_count': 0,
                    'total_amount': 0.0,
                }
            
            invoice_summaries[inv_id]['page_numbers'].add(
                payload.get('page_number', 0)
            )
            invoice_summaries[inv_id]['item_count'] += 1
            
            # Sum total amounts
            total = payload.get('total_amount')
            if total is not None:
                invoice_summaries[inv_id]['total_amount'] += float(total)
        
        # Convert to list format
        summaries = []
        for inv_id, data in invoice_summaries.items():
            summaries.append({
                'invoice_id': inv_id,
                'page_count': len(data['page_numbers']),
                'line_item_count': data['item_count'],
                'total_amount': round(data['total_amount'], 2),
            })
        
        return summaries
    
    def get_invoice_items(
        self,
        invoice_id: str,
        page_number: int = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get all items for a specific invoice.
        
        Args:
            invoice_id: Invoice identifier
            page_number: Filter by page number
            limit: Maximum items to return
            
        Returns:
            List of line items
        """
        conditions = [
            FieldCondition(key="invoice_id", match=MatchValue(value=invoice_id))
        ]
        
        if page_number is not None:
            conditions.append(
                FieldCondition(key="page_number", match=MatchValue(value=page_number))
            )
        
        filter = Filter(must=conditions)
        
        results, _ = self.client.scroll(
            collection_name=self.COLLECTION_NAME,
            scroll_filter=filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        
        return self._format_query_results(results)
    
    def count_items(self, filters: Dict = None) -> int:
        """
        Count items in the store with optional filters.
        
        Args:
            filters: Optional metadata filters
            
        Returns:
            Number of matching items
        """
        qdrant_filter = self._build_qdrant_filter(filters)
        
        return self.client.count(
            collection_name=self.COLLECTION_NAME,
            count_filter=qdrant_filter,
        ).count
    
    def _format_query_results(self, results: List) -> List[Dict]:
        """
        Format Qdrant query results into a consistent structure.
        
        Args:
            results: Raw Qdrant query results (ScrolledPoint or Record)
            
        Returns:
            List of formatted item dictionaries
        """
        import re
        formatted = []
        
        if not results:
            return []
        
        for point in results:
            payload = point.payload
            
            # Get score using helper method for compatibility
            score = self._get_score(point)
            
            # Extract content from payload
            content = payload.get('content', '')
            metadata = {k: v for k, v in payload.items() if k != 'content'}
            
            # Extract description from content if not in metadata
            # Content format: "Item Code: FS-1017\nDescription: Canned Tuna in Sunflower Oil"
            if 'description' not in metadata and content:
                desc_match = re.search(r'Description:\s*(.+?)(?:\n|$)', content)
                if desc_match:
                    metadata['description'] = desc_match.group(1).strip()
            
            # Convert distance to similarity score (Qdrant uses cosine distance)
            similarity = max(0.0, 1.0 - score) if score >= 0 else 0.0
            
            formatted.append({
                'id': f"{self.COLLECTION_NAME}_{point.id}",
                'content': content,
                'metadata': metadata,
                'distance': score,
                'similarity': similarity,
            })
        
        return formatted
