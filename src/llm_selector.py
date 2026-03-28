import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import List, Any
import re

class LLMSelector:
    """Select answer using ThaiLLM-8B-Instruct for logical reasoning."""
    
    def __init__(self, model_name: str = 'Qwen/Qwen2.5-1.5B-Instruct'):
        print(f"Loading Thai-capable reasoning engine: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        # Qwen2.5-1.5B fits entirely in memory, no offloading needed.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != 'cpu' else torch.float32,
            device_map="auto"
        )
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=20, # We only need the choice number
            temperature=0.1,   # Fixed reasoning
            do_sample=False
        )
        print(f"Thai LLM loaded on {device}.")

    def select(self, question: str, choices: List[str], context: str) -> int:
        """Select answer using LLM reasoning."""
        
        # Construct the choice list
        choices_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
        
        # Thai Reasoning Prompt
        prompt = f"""ท่านเป็นพนักงานช่วยเหลือลูกค้าของร้าน 'ฟ้าใหม่' (FahMai) หน้าที่ของท่านคืออ่านบริบทที่กำหนดให้และเลือกตัวเลือกที่ 'ถูกต้องที่สุด' เพียงตัวเลือกเดียวเท่านั้น

### บริบท:
{context}

### คำถาม:
{question}

### ตัวเลือก:
{choices_str}

### คำสั่งพิเศษ:
- หากในบริบทระบุชัดเจนว่า 'ไม่มี' หรือ 'ไม่รองรับ' ให้ระวังอย่าเลือกตัวเลือกที่บอกว่า 'ทำได้'
- หากหาคำตอบในบริบทไม่ได้เลย ให้เลือกตัวเลือก 9
- หากคำถามไม่เกี่ยวกับสินค้าหรือนโยบายร้าน ให้เลือกตัวเลือก 10
- ตอบเพียงตัวเลข 1, 2, 3, 4, 5, 6, 7, 8, 9 หรือ 10 เท่านั้น

คำตอบคือตัวเลข: """

        # Generate response
        try:
            # We use chat format if supported, or just the raw prompt
            messages = [{"role": "user", "content": prompt}]
            
            # Use the tokenizer's chat template
            if hasattr(self.tokenizer, "apply_chat_template"):
                formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                formatted_prompt = prompt
                
            outputs = self.pipe(formatted_prompt)
            response = outputs[0]["generated_text"]
            
            # Extract only the last part (the answer)
            if "<|im_start|>assistant" in response:
                response = response.split("<|im_start|>assistant")[-1].strip()
            elif formatted_prompt in response:
                response = response.replace(formatted_prompt, "").strip()
            
            # Find the first number in the response
            match = re.search(r'\b(10|[1-9])\b', response)
            if match:
                return int(match.group(1))
            else:
                print(f"LLM Reasoning failed to find a number. Output: {response}")
                return 9 # Default to No Data
                
        except Exception as e:
            print(f"Error during LLM reasoning: {e}")
            return 9 # Default to No Data
