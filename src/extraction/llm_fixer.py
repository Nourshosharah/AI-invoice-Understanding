"""
LLM Post-Processing for Invoice Extraction

This module uses Qwen2.5-14B-Instruct to fix extraction errors,
clean up data, and improve the quality of extracted line items.
Uses transformers library for local inference (no API calls).
"""

import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
import os


class LLMFixer:
    """
    Uses Qwen2.5-14B-Instruct to post-process and fix extraction errors.
    
    Handles common issues like:
    - OCR typos in numbers (e.g., "1.0O0" → "100.00")
    - Malformed currency values
    - Missing dates (infer from context)
    - Description cleanup
    - Field validation
    """
    
    # System prompt for extraction fixing
    SYSTEM_PROMPT = """You are an expert invoice data processor. Your task is to
correct and validate extracted line item data from invoices.

IMPORTANT RULES:
1. Fix OCR errors in numbers (e.g., "1.0O0" → 100.00, "l00" → 100)
2. Convert all prices and amounts to numeric values (not strings)
3. Parse dates to YYYY-MM-DD format
4. Clean up descriptions (remove extra whitespace, fix capitalization)
5. Set missing fields to null rather than guessing
6. Only fix obvious errors - don't invent data

Return your response as a valid JSON object with corrected line items.
"""
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-14B-Instruct",
        device: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 512
    ):
        """
        Initialize the LLM fixer with Qwen model.
        
        Args:
            model_path: Path or name of the Qwen model
            device: Device to run on ("auto", "cuda", "cpu")
            torch_dtype: PyTorch dtype ("auto", "bfloat16", "float16")
            max_new_tokens: Maximum tokens to generate
        """
        load_dotenv()
        
        print(f"Loading Qwen model for LLM fixer: {model_path}")
        
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
            print(f"✓ Qwen model loaded successfully for LLM fixer")
            
        except Exception as e:
            print(f"Error loading Qwen model: {e}")
            print("LLM fixing will be disabled")
            self.model = None
            self.tokenizer = None
    
    def fix_extraction(
        self,
        items: List[Dict],
        page_text: str = "",
        invoice_date: str = ""
    ) -> List[Dict]:
        """
        Fix extraction errors in a list of line items.
        
        Args:
            items: List of extracted line item dicts
            page_text: Context text from the page
            invoice_date: Invoice date for date inference
            
        Returns:
            List of corrected line item dicts
        """
        if not items:
            return []
        
        # Only process items that need fixing
        items_to_fix = [item for item in items if self._needs_fixing(item)]
        
        if not items_to_fix:
            return items
        
        if self.model is None:
            print("LLM fixing disabled, returning original items")
            return items
        
        # Build prompt with items needing fixes
        prompt = self._build_fix_prompt(
            items_to_fix, page_text, invoice_date
        )
        
        try:
            # Generate fixed items using Qwen
            response_text = self._generate_with_qwen(prompt)
            
            # Parse JSON response
            result = json.loads(response_text)
            fixed_items = result.get("line_items", [])
            
            # Merge fixed items back with original items
            return self._merge_results(items, fixed_items)
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON from LLM response: {e}")
            return items
        except Exception as e:
            print(f"LLM fixing failed: {e}")
            return items
    
    def _generate_with_qwen(self, prompt: str) -> str:
        """
        Generate JSON response using Qwen model.
        
        Args:
            prompt: Formatted prompt string
            
        Returns:
            Generated response text (should be JSON)
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
                temperature=0.0,
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
    
    def fix_single_item(self, item: Dict, context: str = "") -> Dict:
        """
        Fix a single line item.
        
        Args:
            item: Line item dictionary
            context: Additional context text
            
        Returns:
            Corrected line item dictionary
        """
        if self.model is None:
            return item
        
        prompt = self._build_fix_prompt([item], context, "")
        
        try:
            response_text = self._generate_with_qwen(prompt)
            result = json.loads(response_text)
            fixed_items = result.get("line_items", [])
            
            return fixed_items[0] if fixed_items else item
            
        except Exception as e:
            print(f"LLM fixing failed for single item: {e}")
            return item
    
    def _needs_fixing(self, item: Dict) -> bool:
        """
        Check if an item needs LLM fixing.
        
        Args:
            item: Line item dictionary
            
        Returns:
            True if item likely has errors
        """
        # Check for common error patterns
        
        # Numeric fields as strings
        for field in ['quantity', 'unit_price', 'total_amount']:
            value = item.get(field)
            if isinstance(value, str):
                # Check for non-numeric characters
                if not value.replace('.', '').replace('-', '').isdigit():
                    return True
                # Check for OCR-like patterns
                if re.search(r'[OIl]', value, re.IGNORECASE):
                    return True
        
        # Malformed descriptions
        desc = item.get('description', '')
        if desc and (len(desc) > 200 or '\n' in desc):
            return True
        
        # Missing critical fields (may need inference)
        if not item.get('delivery_date') and item.get('description'):
            return True
        
        return False
    
    def _build_fix_prompt(
        self,
        items: List[Dict],
        page_text: str,
        invoice_date: str
    ) -> str:
        """
        Build the prompt for LLM fixing.
        
        Args:
            items: List of line items needing fixes
            page_text: Page context text
            invoice_date: Invoice date for inference
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            "Fix the following extracted invoice line items.",
            "",
        ]
        
        if invoice_date:
            prompt_parts.extend([
                f"Invoice Date: {invoice_date}",
                "Use this date to infer delivery dates when not specified.",
                "",
            ])
        
        if page_text:
            prompt_parts.extend([
                "Page Context (for reference):",
                page_text[:2000],  # Limit context length
                "",
            ])
        
        prompt_parts.extend([
            "Line Items to Fix:",
            json.dumps(items, indent=2),
            "",
            "Return a JSON object with structure:",
            '{"line_items": [{"item_code": ..., "description": ..., "delivery_date": ..., "quantity": ..., "unit_price": ..., "total_amount": ...}]}',
            "",
            "Rules:",
            "- Fix OCR errors: '1.0O0' → 100.00, 'l00' → 100, 'O' → 0",
            "- Convert all numbers to native JSON numbers (not strings)",
            "- Clean descriptions (fix whitespace, proper case)",
            "- If delivery date unknown, set to null",
            "- If quantity unknown, calculate from unit_price and total_amount if possible",
        ])
        
        return '\n'.join(prompt_parts)
    
    def _merge_results(
        self,
        original_items: List[Dict],
        fixed_items: List[Dict]
    ) -> List[Dict]:
        """
        Merge fixed items back with original list.
        
        Args:
            original_items: Full list of original items
            fixed_items: List of items that were fixed
            
        Returns:
            Merged list with corrections applied
        """
        # Create a map of fixed items by line number
        fixed_map = {
            item.get('line_number'): item
            for item in fixed_items
        }
        
        # Merge: keep original position, update with fixes
        result = []
        for item in original_items:
            line_num = item.get('line_number')
            if line_num in fixed_map:
                # Apply fixes while preserving original metadata
                fixed = fixed_map[line_num]
                # Update fields from fixed version
                for field in ['item_code', 'description', 'delivery_date',
                              'quantity', 'unit_price', 'total_amount']:
                    if field in fixed and fixed[field] is not None:
                        item[field] = fixed[field]
            result.append(item)
        
        return result
    
    def clean_description(self, description: str) -> str:
        """
        Clean up a description string.
        
        Args:
            description: Raw description text
            
        Returns:
            Cleaned description
        """
        # Common cleanup operations
        cleaned = description
        
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Fix common OCR errors
        cleaned = cleaned.replace('l1', '11').replace('lI', 'II')
        cleaned = cleaned.replace('0O', '00').replace('O0', '00')
        cleaned = re.sub(r'\s*,\s*', ', ', cleaned)
        
        # Strip punctuation
        cleaned = cleaned.strip('.,; ')
        
        return cleaned.strip()
    
    def validate_and_fix_numbers(self, value: Any) -> Optional[float]:
        """
        Validate and fix numeric values.
        
        Args:
            value: Raw value (string, number, or None)
            
        Returns:
            Validated float or None
        """
        if value is None:
            return None
        
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            # Remove common issues
            fixed = value
            
            # Fix OCR-like characters
            fixed = fixed.replace('O', '0').replace('l', '1').replace('I', '1')
            fixed = fixed.replace('$', '').replace(',', '').replace('€', '').replace('£', '')
            
            # Handle ranges (take first value)
            if ' - ' in fixed or ' to ' in fixed:
                fixed = re.split(r'\s*[-to]+\s*', fixed)[0]
            
            try:
                return float(fixed)
            except ValueError:
                return None
        
        return None


class LLMClient:
    """
    Unified LLM client for the invoice system.
    
    Provides embedding generation using sentence-transformers
    and text completion using Qwen2.5-14B-Instruct.
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "Qwen/Qwen2.5-14B-Instruct",
        device: str = "auto"
    ):
        """
        Initialize the unified LLM client.
        
        Args:
            embedding_model: Path or name of the embedding model
            llm_model: Path or name of the LLM model
            device: Device to run on ("auto", "cuda", "cpu")
        """
        print("Initializing LLM Client...")
        
        # Initialize embedding model
        print(f"  Loading embedding model: {embedding_model}")
        from sentence_transformers import SentenceTransformer
        
        self.embedding_model = SentenceTransformer(
            embedding_model,
            device=device
        )
        print(f"  ✓ Embedding model loaded")
        
        # Initialize LLM
        print(f"  Loading LLM model: {llm_model}")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                llm_model,
                trust_remote_code=True
            )
            
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                llm_model,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map=device,
            )
            
            self.llm_model.eval()
            print(f"  ✓ LLM model loaded")
            
        except Exception as e:
            print(f"  ⚠ LLM model failed to load: {e}")
            print("  LLM features will be disabled")
            self.llm_model = None
            self.llm_tokenizer = None
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a text string.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding dimensions
        """
        return self.embedding_model.encode(text).tolist()
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        return self.embedding_model.encode(texts).tolist()
    
    def complete(self, prompt: str, system_prompt: str = "") -> str:
        """
        Get text completion for a prompt.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            
        Returns:
            Generated text response
        """
        if self.llm_model is None:
            raise ValueError("LLM model not loaded")
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        text = self.llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt"
        ).to(self.llm_model.device)
        
        with self.llm_model.disable_kv_cache():
            generated_ids = self.llm_model.generate(
                **model_inputs,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.llm_tokenizer.eos_token_id,
            )
        
        response_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        response = self.llm_tokenizer.decode(
            response_ids,
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def complete_json(self, prompt: str, system_prompt: str = "") -> Dict:
        """
        Get JSON completion for a prompt.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            
        Returns:
            Parsed JSON response as dict
        """
        response_text = self.complete(prompt, system_prompt)
        return json.loads(response_text)
    
    def unload(self):
        """
        Unload all models from memory.
        """
        if self.embedding_model:
            del self.embedding_model
        
        if self.llm_model:
            del self.llm_model
        
        print("All models unloaded from memory")
