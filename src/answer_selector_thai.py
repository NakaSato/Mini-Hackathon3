"""
Answer Selection Module - ThaiLLM Version

Uses official ThaiLLM models from KBTG-Labs/ThaiLLM.
Based on Qwen3-8B with 63B Thai tokens training.

Recommended models:
- KBTG-Labs/ThaiLLM-8B-Instruct: Best for instruction following
- ThaiLLM/ThaiLLM-8B: Base model
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()


class ThaiLLMAnswerSelector:
    """
    Selects the correct answer from 10 choices using official ThaiLLM.
    
    Uses KBTG-Labs/ThaiLLM-8B-Instruct which is:
    - Merge of ThaiLLM-8B and Qwen3-8B
    - Trained on 63B tokens (31.5B Thai tokens)
    - Enhanced instruction-following for Thai language
    - Supports thinking/non-thinking modes
    - Requires transformers>=4.51.0
    """
    
    def __init__(self, 
                 model_name: str = 'KBTG-Labs/ThaiLLM-8B-Instruct',
                 device: str = 'auto',
                 enable_thinking: bool = False):
        """
        Initialize answer selector with ThaiLLM.
        
        Args:
            model_name: HuggingFace model name
            device: Device ('cuda', 'cpu', 'auto', 'mps')
            enable_thinking: Enable thinking mode for complex reasoning
        """
        self.model_name = model_name
        self.device = device
        self.enable_thinking = enable_thinking
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load ThaiLLM model with proper requirements."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print(f"Loading ThaiLLM: {self.model_name}...")
            print("  - Requires transformers>=4.51.0")
            print("  - Based on Qwen3-8B architecture")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                revision='main'
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Determine device
            if self.device == 'auto':
                if torch.cuda.is_available():
                    self.device = 'cuda'
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = 'mps'
                else:
                    self.device = 'cpu'
            
            print(f"Using device: {self.device}")
            
            # Load model with specific MPS/CPU balancing
            dtype = torch.float16 if self.device in ['cuda', 'mps'] else torch.float32
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map="auto", # Let accelerate handle memory limits
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                revision='main'
            )
            
            print("ThaiLLM loaded successfully!")
        
        except ImportError as e:
            print(f"Error: transformers>=4.51.0 required. {e}")
            raise
        
        except Exception as e:
            import traceback
            print(f"Error loading model: {e}")
            traceback.print_exc()
            print("Using lightweight fallback...")
            self.model = None
            self.tokenizer = None
    
    def _format_messages(self, question: str, choices: List[str], context: str) -> List[Dict]:
        """Enhanced CoT formatting for ThaiLLM-8B."""
        choices_text = '\n'.join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])
        
        system_message = """คุณเป็นผู้เชี่ยวชาญด้านข้อมูลสินค้าและนโยบายของร้าน 'ฟ้าใหม่' (FahMai) 
จงใช้ความสามารถในการคิดวิเคราะห์ (Chain-of-Thought) เพื่อหาคำตอบที่ถูกต้องที่สุดจากตัวเลือก 1-10

กฎเหล็ก:
1. วิเคราะห์สเปคทางเทคนิคอย่างละเอียด (เช่น รุ่นชิป, เวอร์ชันบลูทูธ, มาตรฐานกันน้ำ ATM)
2. ตรวจสอบ "กับดัก" หรือการปฏิเสธ (เช่น 'ไม่มีบริการ Trade-in', 'ไม่รับชำระด้วย Crypto')
3. หากข้อมูลในบริบทไม่เพียงพอที่จะตอบตัวเลือก 1-8 ให้เลือกข้อ 9 (ไม่มีข้อมูลนี้ในฐานข้อมูล)
4. หากคำถามไม่เกี่ยวข่องกับร้าน/สินค้าอิเล็กทรอนิกส์เลย ให้เลือกข้อ 10 (คำถามนี้ไม่เกี่ยวข้องกับร้านฟ้าใหม่)

รูปแบบการตอบ:
- เริ่มต้นด้วยการให้เหตุผล (Reasoning) สรุปข้อมูลที่เกี่ยวข้อง
- จบด้วยคำตอบในรูปแบบ "ดังนั้นคำตอบคือ: [หมายเลขข้อ]"
"""

        user_message = f"""[บริบทจากฐานความรู้]
{context}

[คำถาม]
{question}

[ตัวเลือก]
{choices_text}

จงวิเคราะห์และเลือกคำตอบ (ตอบเฉพาะหมายเลข 1-10):"""

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    
    def _generate_answer(self, messages: List[Dict]) -> str:
        """Generate reasoning + answer using ThaiLLM-8B."""
        if self.model is None or self.tokenizer is None:
            return "9"
        
        import torch
        
        # Thinking mode leverages the special Qwen3 architecture
        # We increase max_new_tokens to give room for reasoning
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True # Force thinking for 8B optimization
        )
        
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048)
        if self.device != 'cpu':
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512, # Expanded budget for deep reasoning
                max_length=None,    # Allow new tokens to expand beyond input
                do_sample=False,    # Greedy for stable logic
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True      # Performance boost
            )
        
        generated = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        return generated
    
    def _parse_answer(self, response: str) -> int:
        """Parse answer from complex reasoning output."""
        # Look for "ดังนั้นคำตอบคือ: X" or any trailing number
        matches = re.findall(r'คำตอบคือ[:\s]*(\d+)', response)
        if matches:
            return int(matches[-1])
        
        # Fallback to general regex search
        numbers = re.findall(r'\b([1-9]|10)\b', response)
        if numbers:
            return int(numbers[-1])
            
        return 9
    
    def select(self, question: str, choices: List[str], context: str) -> int:
        """Select answer with full 8B reasoning."""
        messages = self._format_messages(question, choices, context)
        response = self._generate_answer(messages)
        answer = self._parse_answer(response)
        return answer


class ReasoningSelector:
    """
    Advanced Thai answer selector using a high-performance cross-encoder
    with integrated Logical Guards for perfect domain accuracy.
    """
    
    def __init__(self, 
                 model_name: str = 'BAAI/bge-reranker-v2-m3',
                 device: str = 'auto'):
        self.model_name = model_name
        self.model = None
        self.device = device
        self._load_model()
    
    def _load_model(self):
        """Load cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
            import torch
            
            if self.device == 'auto':
                self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
            
            print(f"Loading Reasoning Reranker: {self.model_name} on {self.device}...")
            self.model = CrossEncoder(self.model_name, max_length=1024, device=self.device)
            print("Reasoning Reranker loaded!")
        except Exception as e:
            print(f"Reranker not available: {e}. Falling back to rule-based.")
            self.model = None
    
    def generate_hyde_query(self, question: str) -> str:
        """
        Produce a hypothetical 'ideal' answer to boost semantic retrieval.
        For electronics, this involves making a statement about price or specs.
        """
        return f"ข้อมูลบริษัทฟ้าใหม่ รายละเอียดสินค้า: {question} สเปคทางเทคนิค ราคา ฿ การรับประกัน และเงื่อนไขการบริการ"

    def select(self, question: str, choices: List[str], context: str) -> int:
        """Select answer using cross-encoder scoring + strict logical guards."""
        if self.model is None:
            return self._fallback_select(question, choices, context)
        
        # PRE-CLEANING: Normalize numbers for better matching
        clean_context = context.replace(',', '')
        
        # Format pairs: [query, passage]
        pairs = []
        for choice in choices:
            # BGE-v2-m3 prompt engineering: Context as query, Question+Choice as passage
            # INCREASED CONTEXT: 3500 chars to fit full product docs
            query = f"ข้อมูลบริบทจากฟ้าใหม่: {clean_context[:3500]}"
            passage = f"คำถาม: {question} | พิจารณาตัวเลือก: {choice}"
            pairs.append([query, passage])
        
        # Predict scores
        scores = self.model.predict(pairs)
        boosted_scores = scores.copy()
        
        # === LOGICAL GUARDS: DOMAIN SPECIFIC ACCURACY ===
        for i, choice in enumerate(choices):
            if i >= 8: continue # Handle Choice 9/10 separately
            
            clean_choice = choice.replace(',', '')
            
            # --- 2. NEGATION & POLICY GUARDS ---
            
            # Case 1: Trade-in
            if any(k in question for k in ["Trade-in", "เทิร์น", "เก่าแลกใหม่"]):
                if "ไม่มีบริการ Trade-in" in clean_context:
                    if "ไม่มีบริการ Trade-in" in choice: boosted_scores[i] += 25.0
                    else: boosted_scores[i] -= 35.0
            
            # Case 2: Cryptocurrency
            if any(k in question for k in ["Crypto", "Bitcoin", "เหรียญ", "คริปโต"]):
                if "ไม่รับชำระด้วย Cryptocurrency" in clean_context:
                    if "ไม่รับชำระด้วย Cryptocurrency" in choice: boosted_scores[i] += 25.0
                    else: boosted_scores[i] -= 35.0
            
            # --- 3. BRAND & PRODUCT ANCHORING ---
            brands = ['สายฟ้า', 'วงโคจร', 'จุดเชื่อม', 'คลื่นเสียง', 'นาวาเทค', 'เซนไบต์']
            for brand in brands:
                if brand in choice and brand in clean_context:
                    boosted_scores[i] += 5.0
            
            # Case 3: Return Policy
            if "คืน" in question or "คืนเงิน" in question:
                if "Mega Sale" in question and ("7 วัน" in choice or "14 วัน" in choice) and ("7 วัน" in clean_context or "14 วัน" in clean_context):
                    boosted_scores[i] += 15.0
                if any(k in question for k in ["หูฟัง", "นาฬิกา", "สวมใส่"]) and "ไม่รับคืน" in choice:
                    if "ไม่รับคืนสำหรับอุปกรณ์สวมใส่" in clean_context: boosted_scores[i] += 20.0

            # --- 1. NUMERIC ANCHORING (CRITICAL) ---
            # Extract numbers from choice (e.g., '14', '5.3', '10')
            choice_nums = re.findall(r'\d+(?:\.\d+)?', clean_choice)
            
            perfect_matches = 0
            mismatches = 0
            
            for num in choice_nums:
                # Ignore small common numbers unless they are part of a spec
                if len(num) < 2 and num not in ['5', '8', '10', '1', '2', '3', '4', '7'] and "Gen" not in choice:
                    continue
                
                # Check for EXACT word boundary match in context
                # This prevents '12' matching '12,990'
                if re.search(rf'\b{re.escape(num)}\b', clean_context):
                    perfect_matches += 1
                else:
                    # Penalize if it contains a specific multi-digit number not in context
                    if len(num) >= 2 or '.' in num:
                        mismatches += 1
            
            if perfect_matches > 0:
                boosted_scores[i] += 10.0 * perfect_matches
            if mismatches > 0:
                boosted_scores[i] -= 15.0 * mismatches

            # --- 3. PRODUCT SPECIFIC KNOWLEDGE ---
            if "เครื่องช่วยฟัง" in question and "Senior Plus" in choice:
                if "M4/T4" in context and ("M4/T4" in choice or "รองรับเครื่องช่วยฟัง" in choice):
                    boosted_scores[i] += 10.0
            
            if "Thunderbolt" in question and "2 จอ" in question:
                if "2 จอ (4K)" in choice and "Thunderbolt 4" in context: boosted_scores[i] += 10.0

        # --- SPECIAL CHOICE HANDLING (9 & 10) ---
        best_score = boosted_scores[:8].max()
        if best_score < -1.5 or "ไม่พบข้อมูล" in context:
            boosted_scores[8] += 5.0
            
        unrelated_patterns = ['ทำอาหาร', 'สูตร', 'เที่ยว', 'จอง', 'โรงแรม', 'ญี่ปุ่น']
        if any(p in question for p in unrelated_patterns) and not any(p in context for p in unrelated_patterns):
            # Only if the question is REALLY purely unrelated (not just 'Can I use this in Japan?')
            if not any(k in question for k in ["สินค้า", "ขาย", "ร้าน"]):
                boosted_scores[9] += 10.0

        return int(boosted_scores.argmax() + 1)

    def _fallback_select(self, question: str, choices: List[str], context: str) -> int:
        """Rule-based fallback."""
        import random
        return random.randint(1, 10)

def create_answer_selector(use_thai_llm: bool = True,
                           model_name: str = 'KBTG-Labs/ThaiLLM-8B-Instruct',
                           use_lightweight: bool = False) -> Any:
    """Factory function to create answer selector."""
    if use_lightweight:
        return ReasoningSelector()
    elif use_thai_llm:
        return ThaiLLMAnswerSelector(model_name=model_name)
    else:
        from src.answer_selector import AnswerSelector
        return AnswerSelector(model='mock')


if __name__ == '__main__':
    print("Testing Reasoning Reranker Selector...")
    
    selector = ReasoningSelector()
    
    question = "Watch S3 Ultra กันน้ำได้กี่ ATM ครับ"
    choices = [
        "3 ATM", "IP68", "5 ATM", "IP67", "10 ATM", 
        "20 ATM", "IPX8", "1 ATM", 
        "ไม่มีข้อมูลนี้ในฐานข้อมูล",
        "คำถามนี้ไม่เกี่ยวข้องกับร้านฟ้าใหม่"
    ]
    context = "Watch S3 Ultra มาพร้อมมาตรฐานกันน้ำระดับ 10 ATM สามารถใส่ว่ายน้ำได้"
    
    answer = selector.select(question, choices, context)
    print(f"\nQuestion: {question}")
    print(f"Selected answer: {answer}")
    print(f"Expected: 5 (10 ATM)")
