import os
import torch
from typing import Any, List, Optional, Iterator
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline


class LocalHuggingFaceLLM(LLM):
    """
    Local LLM wrapper for running models directly on HPC GPUs
    """
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"  # Can be changed
    model: Any = None
    tokenizer: Any = None
    pipeline: Any = None
    device: str = None
    
    def __init__(self, model_name: str = None, device: str = None, **kwargs):
        super().__init__(**kwargs)
        
        if model_name:
            self.model_name = model_name
            
        # Detect device
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
            print(f"CUDA available. Using GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = "cpu"
            print("No GPU available, using CPU")
        
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer"""
        print(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Load model with appropriate settings for GPU/CPU
        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,  # Use fp16 for GPU efficiency
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True
            )
        
        # Create pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            max_new_tokens=1000,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
        )
        
        print(f"Model loaded successfully on {self.device}")
    
    @property
    def _llm_type(self) -> str:
        return "local_huggingface"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        """Generate text from the prompt"""
        try:
            # Generate response
            response = self.pipeline(
                prompt,
                max_new_tokens=kwargs.get("max_tokens", 1000),
                temperature=kwargs.get("temperature", 0.7),
                do_sample=True,
                stop_sequence=stop,
            )
            
            # Extract generated text
            generated_text = response[0]['generated_text']
            
            # Remove the input prompt from the response
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            print(f"Error generating text: {e}")
            return f"Error: {str(e)}"


class LocalLlamaCpp(LLM):
    """
    Alternative: Use llama.cpp for efficient CPU/GPU inference
    """
    model_path: str
    n_gpu_layers: int = -1  # -1 means use all available GPU layers
    n_ctx: int = 2048
    model: Any = None
    
    def __init__(self, model_path: str, **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.n_gpu_layers = kwargs.get("n_gpu_layers", -1)
        self.n_ctx = kwargs.get("n_ctx", 2048)
        self._load_model()
    
    def _load_model(self):
        """Load llama.cpp model"""
        try:
            from llama_cpp import Llama
            
            self.model = Llama(
                model_path=self.model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                verbose=True
            )
            print(f"Llama.cpp model loaded from {self.model_path}")
            
        except Exception as e:
            print(f"Error loading llama.cpp model: {e}")
            raise
    
    @property
    def _llm_type(self) -> str:
        return "llama_cpp"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        """Generate text using llama.cpp"""
        try:
            response = self.model(
                prompt,
                max_tokens=kwargs.get("max_tokens", 1000),
                temperature=kwargs.get("temperature", 0.7),
                stop=stop,
                echo=False
            )
            
            return response['choices'][0]['text']
            
        except Exception as e:
            print(f"Error generating text with llama.cpp: {e}")
            return f"Error: {str(e)}"


# Factory function to create appropriate LLM based on available resources
def create_local_llm(model_type: str = "auto", **kwargs):
    """
    Create an appropriate local LLM based on available resources
    
    Args:
        model_type: "huggingface", "llama_cpp", or "auto"
        **kwargs: Additional arguments for the LLM
    
    Returns:
        An LLM instance
    """
    if model_type == "auto":
        # Auto-detect best option
        if torch.cuda.is_available():
            print("GPU detected, using HuggingFace model")
            return LocalHuggingFaceLLM(**kwargs)
        else:
            print("No GPU, checking for llama.cpp model...")
            model_path = kwargs.get("model_path", "/models/llama-2-7b.gguf")
            if os.path.exists(model_path):
                return LocalLlamaCpp(model_path=model_path, **kwargs)
            else:
                print("No local model found, using HuggingFace on CPU")
                return LocalHuggingFaceLLM(device="cpu", **kwargs)
    
    elif model_type == "huggingface":
        return LocalHuggingFaceLLM(**kwargs)
    
    elif model_type == "llama_cpp":
        return LocalLlamaCpp(**kwargs)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")