"""
Answer Selection Module

Handles:
- Formatting prompts for LLM
- Answer selection from 10 choices
- Special case handling (Choice 9: no data, Choice 10: out-of-scope)
"""

import os
import re
import random
from pythainlp.tokenize import word_tokenize
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()


class AnswerSelector:
    """
    Selects the correct answer from 10 choices using LLM.
    """
    
    def __init__(self, 
                 model: str = 'gpt-4o-mini',
                 temperature: float = 0.1,
                 use_few_shot: bool = True):
        """
        Initialize answer selector.
        
        Args:
            model: LLM model name
            temperature: Sampling temperature (low for consistency)
            use_few_shot: Whether to use few-shot examples
        """
        self.model = model
        self.temperature = temperature
        self.use_few_shot = use_few_shot
        
        # Check for API key
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            print("Warning: OPENAI_API_KEY not set. Using mock mode.")
            self.api_key = None
        
        # Few-shot examples
        self.few_shot_examples = self._load_few_shot_examples()
    
    def _load_few_shot_examples(self) -> List[Dict[str, Any]]:
        """Load few-shot examples for prompting."""
        return [
            {
                'question': 'Watch S3 Ultra กันน้ำได้กี่ ATM ครับ',
                'context': 'Watch S3 Ultra มาพร้อมมาตรฐานกันน้ำระดับ 10 ATM สามารถใส่ว่ายน้ำและดำน้ำตื้นได้',
                'choices': ['3 ATM', 'IP68', '5 ATM', 'IP67', '10 ATM', '20 ATM', 'IPX8', '1 ATM'],
                'answer': 5,
                'reasoning': 'จากเอกสารระบุชัดเจนว่า Watch S3 Ultra กันน้ำระดับ 10 ATM'
            },
            {
                'question': 'ร้านฟ้าใหม่มีสาขาที่ประเทศญี่ปุ่นไหม',
                'context': 'ฟ้าใหม่มีศูนย์บริการ 5 แห่ง: กรุงเทพฯ 3 แห่ง, เชียงใหม่ 1 แห่ง, ภูเก็ต 1 แห่ง',
                'choices': ['มี 1 สาขา', 'มี 3 สาขา', 'มี 5 สาขา', 'มี 10 สาขา', 'ไม่มีสาขาในญี่ปุ่น', 'กำลังจะเปิด', 'มีเฉพาะออนไลน์', 'มี 2 สาขา'],
                'answer': 5,
                'reasoning': 'เอกสารระบุสาขาเฉพาะในประเทศไทย ไม่มีข้อมูลเกี่ยวกับสาขาในญี่ปุ่น'
            },
            {
                'question': 'วิธีทำอาหารผัดไทยให้อร่อย',
                'context': 'ฟ้าใหม่เป็นร้านขายอุปกรณ์อิเล็กทรอนิกส์ ไม่ขายอาหารหรือสูตรทำอาหาร',
                'choices': ['ใส่ไข่เพิ่ม', 'ใช้ไฟแรง', 'ใส่น้ำมะขาม', 'ผัดเร็วๆ', 'ใส่กุ้งสด', 'ใส่เต้าหู้', 'ใส่หัวไชโป๊', 'ใส่ใบกุยช่าย'],
                'answer': 8,
                'reasoning': 'คำถามเกี่ยวกับสูตรอาหาร ไม่เกี่ยวข้องกับร้านฟ้าใหม่ (Choice 10 = ตัวเลือกที่ 8 ในที่นี้)'
            }
        ]
    
    def _format_choices(self, choices: List[str]) -> str:
        """Format choices for prompt."""
        formatted = []
        for i, choice in enumerate(choices, 1):
            formatted.append(f"{i}. {choice}")
        return '\n'.join(formatted)
    
    def _build_prompt(self, 
                      question: str,
                      choices: List[str],
                      context: str) -> str:
        """
        Build prompt for LLM.
        
        Args:
            question: The question to answer
            choices: List of 10 answer choices
            context: Retrieved context from knowledge base
        
        Returns:
            Formatted prompt string
        """
        # System prompt
        system_prompt = """คุณเป็นผู้ช่วยตอบคำถามเกี่ยวกับร้านฟ้าใหม่ (FahMai) 
หน้าที่ของคุณคือเลือกคำตอบที่ถูกต้องจาก 10 ตัวเลือก โดยพิจารณาจากข้อมูลที่กำหนดให้

คำแนะนำ:
1. อ่านข้อมูลอย่างละเอียด
2. เปรียบเทียบแต่ละตัวเลือกกับข้อมูลที่มี
3. เลือกตัวเลือกที่ตรงกับข้อมูลมากที่สุด
4. หากข้อมูลไม่มีในเอกสาร ให้เลือกข้อ 9 (ไม่มีข้อมูลนี้ในฐานข้อมูล)
5. หากคำถามไม่เกี่ยวข้องกับร้านฟ้าใหม่ ให้เลือกข้อ 10 (คำถามนี้ไม่เกี่ยวข้องกับร้านฟ้าใหม่)

ตอบเฉพาะตัวเลข 1-10 เท่านั้น"""

        # Build user prompt
        user_prompt = f"""คำถาม: {question}

ตัวเลือก:
{self._format_choices(choices)}

ข้อมูลจากฐานความรู้:
{context}

จงเลือกคำตอบที่ถูกต้อง (ตอบเฉพาะตัวเลข 1-10):"""

        return system_prompt, user_prompt
    
    def _call_llm(self, 
                  system_prompt: str,
                  user_prompt: str) -> str:
        """
        Call LLM API.
        
        Returns:
            LLM response text
        """
        if not self.api_key:
            # Mock mode - return random answer for testing
            import random
            return str(random.randint(1, 10))
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=10
            )
            
            return response.choices[0].message.content.strip()
        
        except ImportError:
            print("Warning: openai not installed. Using mock mode.")
            import random
            return str(random.randint(1, 10))
        
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return "9"  # Default to "no data" on error
    
    def _parse_answer(self, response: str) -> int:
        """
        Parse LLM response to extract answer number.
        
        Returns:
            Integer 1-10
        """
        # Extract first number from response
        numbers = re.findall(r'\b([1-9]|10)\b', response)
        
        if numbers:
            answer = int(numbers[0])
            if 1 <= answer <= 10:
                return answer
        
        # Default to 9 (no data) if parsing fails
        return 9
    
    def select(self, 
               question: str,
               choices: List[str],
               context: str,
               return_reasoning: bool = False) -> int:
        """
        Select the correct answer.
        
        Args:
            question: The question to answer
            choices: List of 10 answer choices
            context: Retrieved context
            return_reasoning: Whether to return reasoning (for debugging)
        
        Returns:
            Selected answer (1-10)
        """
        system_prompt, user_prompt = self._build_prompt(
            question, choices, context
        )
        
        response = self._call_llm(system_prompt, user_prompt)
        answer = self._parse_answer(response)
        
        if return_reasoning:
            return answer, response
        
        return answer
    
    def select_batch(self,
                     questions_data: List[Dict[str, Any]],
                     verbose: bool = True) -> List[int]:
        """
        Select answers for multiple questions.
        
        Args:
            questions_data: List of dicts with 'question', 'choices', 'context'
            verbose: Show progress
        
        Returns:
            List of selected answers
        """
        answers = []
        
        for i, data in enumerate(questions_data):
            if verbose:
                print(f"Processing question {i+1}/{len(questions_data)}...")
            
            answer = self.select(
                question=data['question'],
                choices=data['choices'],
                context=data.get('context', '')
            )
            answers.append(answer)
        
        return answers


class SpecialCaseDetector:
    """
    Detects special cases:
    - Choice 9: No data in knowledge base
    - Choice 10: Question not related to FahMai
    """
    
    def __init__(self, 
                 embedding_model,
                 no_data_threshold: float = 0.35,
                 unrelated_threshold: float = 0.30):
        """
        Initialize special case detector.
        
        Args:
            embedding_model: EmbeddingModel for similarity check
            no_data_threshold: Threshold for "no data" detection
            unrelated_threshold: Threshold for "unrelated" detection
        """
        self.embedding_model = embedding_model
        self.no_data_threshold = no_data_threshold
        self.unrelated_threshold = unrelated_threshold
        
        # Keywords for unrelated questions
        self.unrelated_topics = [
            'อาหาร', 'สูตร', 'ทำอาหาร', 'ร้านอาหาร',
            'ท่องเที่ยว', 'เที่ยว', 'โรงแรม', 'ที่พัก',
            'การเมือง', 'ข่าว', 'กีฬา',
            'สุขภาพ', 'หมอ', 'โรงพยาบาล',
            'โรงเรียน', 'มหาวิทยาลัย', 'การศึกษา',
            'การเงิน', 'ธนาคาร', 'หุ้น',
            'รถยนต์', 'มอเตอร์ไซค์',
        ]
    
    def is_unrelated(self, question: str) -> bool:
        """
        Check if question is unrelated to FahMai.
        
        FahMai sells electronics. Questions about other topics
        should be flagged as unrelated.
        """
        # Check for unrelated keywords
        for topic in self.unrelated_topics:
            if topic in question:
                # Additional check: is there any electronics mention?
                electronics_keywords = [
                    'มือถือ', 'โทรศัพท์', 'คอมพิวเตอร์', 'โน้ตบุ๊ค',
                    'หูฟัง', 'ลำโพง', 'นาฬิกา', 'แท็บเล็ต',
                    'ชาร์จ', 'แบตเตอรี่', 'หน้าจอ', 'กล้อง',
                    'ฟ้าใหม่', 'FahMai', 'สายฟ้า', 'ดาวเหนือ'
                ]
                
                has_electronics = any(
                    kw in question for kw in electronics_keywords
                )
                
                if not has_electronics:
                    return True
        
        return False
    
    def is_no_data(self, 
                   question: str,
                   retrieval_results: List[Dict[str, Any]],
                   min_score: float = 0.01) -> bool:
        """
        Check if there's no relevant data in retrieval results.
        
        Args:
            question: The question
            retrieval_results: Results from retrieval system
            min_score: Minimum combined score threshold (adjusted for RRF scale)
        """
        if not retrieval_results:
            return True
        
        # Check best score
        # Note: Hybrid Search (RRF) scores are small (e.g. 1/60 + 1/61 = 0.033)
        # We check if we have any result at all from retrieval.
        best_score = max(r.get('combined_score', 0) for r in retrieval_results)
        
        # If score is very low, it might be just random noise
        # But for RRF, even 0.01 is a valid rank (e.g. rank 40)
        if best_score < min_score:
            return True
        
        # Check if any result has meaningful content match
        # Use pythainlp for Thai-aware word overlap
        question_tokens = set(word_tokenize(question.lower(), engine='newmm'))
        # Remove common Thai stops and short chars
        question_tokens = {t.strip() for t in question_tokens if len(t.strip()) > 1}
        
        for result in retrieval_results:
            content = result.get('content', '')
            content_tokens = set(word_tokenize(content.lower(), engine='newmm'))
            
            overlap = len(question_tokens & content_tokens)
            # If we have significant word overlap, it's not "No Data"
            if overlap >= 2:
                return False
        
        # If we got here and the score is decent, we trust the retrieval
        if best_score > 0.02: # Equivalent to being in top 20 for at least one search
            return False
            
        return True
    
    def detect(self,
               question: str,
               retrieval_results: List[Dict[str, Any]]) -> Optional[int]:
        """
        Detect special cases.
        
        Returns:
            9 if no data, 10 if unrelated, None otherwise
        """
        # Check unrelated first
        if self.is_unrelated(question):
            return 10
        
        # Check no data
        if self.is_no_data(question, retrieval_results):
            return 9
        
        return None


def create_answer_selector(model: str = 'gpt-4o-mini',
                           temperature: float = 0.1) -> AnswerSelector:
    """
    Factory function to create answer selector.
    """
    return AnswerSelector(
        model=model,
        temperature=temperature
    )


if __name__ == '__main__':
    # Test answer selector
    selector = create_answer_selector()
    
    # Test case
    question = "Watch S3 Ultra กันน้ำได้กี่ ATM ครับ"
    choices = [
        "3 ATM", "IP68", "5 ATM", "IP67", "10 ATM", 
        "20 ATM", "IPX8", "1 ATM", 
        "ไม่มีข้อมูลนี้ในฐานข้อมูล",
        "คำถามนี้ไม่เกี่ยวข้องกับร้านฟ้าใหม่"
    ]
    context = """Watch S3 Ultra มาพร้อมมาตรฐานกันน้ำระดับ 10 ATM 
    สามารถใส่ว่ายน้ำและดำน้ำตื้นได้ เหมาะสำหรับการใช้งานทางน้ำ"""
    
    answer = selector.select(question, choices, context)
    print(f"Question: {question}")
    print(f"Selected answer: {answer}")
    print(f"Expected: 5 (10 ATM)")
