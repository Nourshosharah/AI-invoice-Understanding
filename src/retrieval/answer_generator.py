"""
Answer Generator for Invoice Q&A System

This module generates grounded responses from retrieved invoice data
using Qwen2.5-14B-Instruct for answer generation with strict source citation.
Uses transformers library for local inference (no API calls).
"""

import json
from typing import List, Dict, Any, Optional
from datetime import date
from dotenv import load_dotenv
import os


class AnswerGenerator:
    """
    Generates grounded answers from retrieved invoice line items.
    
    Uses Qwen2.5-14B-Instruct via transformers to:
    1. Format retrieved items as context
    2. Generate natural language answers
    3. Maintain strict grounding in source data
    """
    
    # System prompt for answer generation
    SYSTEM_PROMPT = """You are a helpful assistant that answers questions about invoice line items.

CRITICAL RULES:
1. ONLY use information from the provided context
2. If the answer is not in the context, say "I don't have that information"
3. Do NOT make up or infer information not present in the context
4. Always cite your sources using the provided format
5. Be concise but complete in your responses

When citing sources, use this format:
- Include the invoice ID, page number, and line number for each claim
- Example: "Items from invoice INV-001, page 3, line 5"

Structure your response with:
1. Direct answer to the question
2. Formatted table or list of relevant items (when appropriate)
3. Summary of key findings (when relevant)
"""
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-14B-Instruct",
        device: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 1024
    ):
        """
        Initialize the answer generator with Qwen model.
        
        Args:
            model_path: Path or name of the Qwen model
            device: Device to run on ("auto", "cuda", "cpu")
            torch_dtype: PyTorch dtype ("auto", "bfloat16", "float16")
            max_new_tokens: Maximum tokens to generate
        """
        load_dotenv()
        
        print(f"Loading Qwen model: {model_path}")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map=device,
            )
            
            self.model.eval()
            
            self.max_new_tokens = max_new_tokens
            print(f"✓ Qwen model loaded successfully on device: {device}")
            
        except Exception as e:
            print(f"Error loading Qwen model: {e}")
            print("Falling back to template-based responses only")
            self.model = None
            self.tokenizer = None
    
    def generate(
        self,
        query: str,
        retrieved_items: List[Dict],
        query_type: str = None
    ) -> Dict:
        """
        Generate an answer from retrieved items.
        
        Args:
            query: User's question
            retrieved_items: List of retrieved line items
            query_type: Type of query (structural, semantic, etc.)
            
        Returns:
            Dict with answer text, sources, and retrieved items
        """
        if not retrieved_items:
            return {
                "answer": "I don't have any information that matches your query.",
                "sources": [],
                "retrieved_items": [],
                "query_type": query_type,
            }
        
        # Format context from retrieved items
        context = self._format_context(retrieved_items)
        
        # Build prompt
        prompt = self._build_prompt(query, context)
        
        if self.model is None:
            # Fallback to template-based response
            return self._fallback_response(query, retrieved_items, query_type)
        
        try:
            # Generate response using Qwen
            answer = self._generate_with_qwen(prompt)
            
            # Extract sources from retrieved items
            sources = self._extract_sources(retrieved_items)
            
            return {
                "answer": answer,
                "sources": sources,
                "retrieved_items": retrieved_items,
                "query_type": query_type,
            }
            
        except Exception as e:
            print(f"Qwen generation failed: {e}")
            return self._fallback_response(query, retrieved_items, query_type)
    
    def _generate_with_qwen(self, prompt: str) -> str:
        """
        Generate response using Qwen model.
        
        Args:
            prompt: Formatted prompt string
            
        Returns:
            Generated response text
        """
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt"
        ).to(self.model.device)
        
        with self.model.disable_kv_cache():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.0,  # Deterministic
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Extract only the new tokens (response)
        response_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        response = self.tokenizer.decode(
            response_ids,
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def generate_structured(
        self,
        query: str,
        retrieved_items: List[Dict],
        query_type: str = None
    ) -> Dict:
        """
        Generate a structured response with formatted table.
        
        Args:
            query: User's question
            retrieved_items: List of retrieved line items
            query_type: Type of query
            
        Returns:
            Dict with table-formatted response
        """
        if not retrieved_items:
            return {
                "answer": "No matching items found.",
                "table": None,
                "summary": None,
                "sources": [],
            }
        
        # Generate answer
        result = self.generate(query, retrieved_items, query_type)
        
        # Add table format
        result["table"] = self._format_as_table(retrieved_items)
        
        # Add summary
        result["summary"] = self._generate_summary(retrieved_items)
        
        return result
    
    def _format_context(self, items: List[Dict]) -> str:
        """
        Format retrieved items as context string.
        
        Args:
            items: List of retrieved line items
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for idx, item in enumerate(items, start=1):
            metadata = item.get('metadata', {})
            content = item.get('content', '')
            
            # Build structured context
            context_parts.append(f"[Item {idx}]")
            context_parts.append(f"Invoice: {metadata.get('invoice_id', 'N/A')}")
            context_parts.append(f"Page: {metadata.get('page_number', 'N/A')}")
            context_parts.append(f"Line: {metadata.get('line_number', 'N/A')}")
            context_parts.append(content)
            context_parts.append("")  # Empty line between items
        
        return '\n'.join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """
        Build the full prompt for LLM generation.
        
        Args:
            query: User's question
            context: Formatted context from retrieved items
            
        Returns:
            Complete prompt string
        """
        prompt = f"""Based on the following retrieved invoice line items, answer the question.

Question: {query}

Retrieved Line Items:
{context}

Please provide a helpful answer based only on the items above.
"""
        return prompt
    
    def _extract_sources(self, items: List[Dict]) -> List[Dict]:
        """
        Extract source citations from retrieved items.
        
        Args:
            items: Retrieved line items
            
        Returns:
            List of source references
        """
        sources = []
        
        for item in items:
            metadata = item.get('metadata', {})
            sources.append({
                "invoice_id": metadata.get('invoice_id'),
                "page_number": metadata.get('page_number'),
                "line_number": metadata.get('line_number'),
                "item_code": metadata.get('item_code'),
            })
        
        return sources
    
    def _format_as_table(self, items: List[Dict]) -> str:
        """
        Format items as a markdown table.
        
        Args:
            items: List of line items
            
        Returns:
            Markdown formatted table string
        """
        if not items:
            return ""
        
        # Build table header
        header = "| Code | Description | Qty | Unit Price | Total |"
        separator = "|------|-------------|-----|------------|-------|"
        
        rows = []
        for item in items:
            metadata = item.get('metadata', {})
            
            code = metadata.get('item_code', 'N/A') or 'N/A'
            desc = metadata.get('description', 'N/A') or 'N/A'
            # Truncate long descriptions
            if len(desc) > 50:
                desc = desc[:47] + '...'
            qty = metadata.get('quantity')
            qty_str = f"{qty:.2f}" if qty else 'N/A'
            price = metadata.get('unit_price')
            price_str = f"${price:.2f}" if price else 'N/A'
            total = metadata.get('total_amount')
            total_str = f"${total:.2f}" if total else 'N/A'
            
            rows.append(f"| {code} | {desc} | {qty_str} | {price_str} | {total_str} |")
        
        table = '\n'.join([header, separator] + rows)
        return table
    
    def _generate_summary(self, items: List[Dict]) -> Dict:
        """
        Generate summary statistics for retrieved items.
        
        Args:
            items: Retrieved line items
            
        Returns:
            Dict with summary statistics
        """
        if not items:
            return {"count": 0}
        
        total_amount = 0.0
        total_quantity = 0.0
        invoices = set()
        pages = set()
        descriptions = []
        
        for item in items:
            metadata = item.get('metadata', {})
            
            # Sum totals
            total = metadata.get('total_amount')
            if total is not None:
                total_amount += float(total)
            
            qty = metadata.get('quantity')
            if qty is not None:
                total_quantity += float(qty)
            
            # Collect metadata
            inv_id = metadata.get('invoice_id')
            if inv_id:
                invoices.add(inv_id)
            
            page = metadata.get('page_number')
            if page:
                pages.add(page)
            
            desc = metadata.get('description')
            if desc:
                descriptions.append(desc)
        
        return {
            "count": len(items),
            "total_amount": round(total_amount, 2),
            "total_quantity": round(total_quantity, 2),
            "invoice_count": len(invoices),
            "invoice_ids": list(invoices),
            "pages": sorted(list(pages)),
        }
    
    def _fallback_response(
        self,
        query: str,
        items: List[Dict],
        query_type: str
    ) -> Dict:
        """
        Generate response without LLM (fallback for errors).
        
        Args:
            query: User's question
            items: Retrieved items
            query_type: Type of query
            
        Returns:
            Basic formatted response
        """
        # Build simple response
        lines = [f"Found {len(items)} matching items:"]
        
        # Format as table
        table = self._format_as_table(items)
        if table:
            lines.append("")
            lines.append(table)
        
        # Add summary
        summary = self._generate_summary(items)
        lines.append("")
        lines.append(f"Total: ${summary.get('total_amount', 0):.2f}")
        
        # Add sources
        sources = self._extract_sources(items)
        
        return {
            "answer": '\n'.join(lines),
            "sources": sources,
            "retrieved_items": items,
            "query_type": query_type,
        }
