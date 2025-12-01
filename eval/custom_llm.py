import re 
import sys
import json
from pathlib import Path
from typing import Optional
from deepeval.models import DeepEvalBaseLLM
from openai import OpenAI

class LocalEvalLlama(DeepEvalBaseLLM):
    """
    Wrapper for using local LLM model with DeepEval.
    """

    def __init__(self):
        self.client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="sk-xxx")
        self.model_name = "Local LLM" # TODO: implement options for selecting eval model

    def load_model(self):
        return self.client
    
    def _clean_json_response(self, response: str) -> str:
        """Clean response of any LLM generated artifacts"""
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)

        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json_match.group(0).strip()

        return response.strip()

    def generate(self, prompt: str) -> str:
        client = self.load_model()
        needs_json = "json" in prompt.lower() or '"' in prompt or "{" in prompt

        if needs_json:
            messages = [
                {
                    "role": "system",
                    "content": "You are a precise evaluator. Always respond with valid JSON only, no explanations."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        else:
            messages = [
                {
                    "role": "user", 
                    "content": prompt
                }
            ]

        try:
            response = client.chat.completions.create(
                model="localmodel",
                messages=messages,
                max_tokens=2000,
                temperature=0.1,
                top_p=0.95,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )

            result = response.choices[0].message.content

            if needs_json:
                result = self._clean_json_response(result)
            
            return result
        
        except Exception as e:
            raise RuntimeError(f"Error generatign respones: {e}")
    
    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name
    
def get_local_llama():
    return LocalEvalLlama()