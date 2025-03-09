import time
import os
import sys
from typing import Any, Dict, List, Optional

# Add the current directory to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import GPTCache components
from gptcache.embedding import SBERT
from gptcache.similarity_evaluation import SbertCrossencoderEvaluation, SearchDistanceEvaluation
from gptcache.processor.pre import get_prompt
from gptcache.adapter.langchain_models import LangChainLLMs

# Import custom components
from components import custom_llm
from components.monitored_cache import MonitoredCache

# Import cache logger and analyzer (for reporting)
from components.cache_logger import CachePerformanceLogger
from components.cache_analyzer import generate_latest_run_report

def embedding_func(prompt: str, **kwargs) -> tuple:
    """Embedding function for the cache."""
    encoder = SBERT('all-MiniLM-L6-v2')
    return tuple(encoder.to_embeddings(prompt))

def run_cache_test(cache_dir: str = "cache_logs", similarity_threshold: float = 0.7,
                  test_questions: Optional[List[str]] = None):
    """
    Run a test of the cache system with performance monitoring.
    
    Args:
        cache_dir: Directory for cache logs
        similarity_threshold: Threshold for similarity evaluation
        test_questions: List of questions to test
    """
    print(f"Starting cache test with similarity threshold: {similarity_threshold}")
    
    # Create monitored cache
    cache = MonitoredCache(log_dir=cache_dir)
    
    # Initialize cache with our specific configuration
    evaluation = SbertCrossencoderEvaluation()
    
    cache.init(
        pre_embedding_func=get_prompt,
        embedding_func=embedding_func, 
        similarity_evaluation=evaluation
    )
    
    # Initialize the LLM and cached LLM
    llm = custom_llm.localLlama()
    cached_llm = LangChainLLMs(llm=llm)
    
    # Default questions if none provided
    if test_questions is None:
        test_questions = [
            "What is GitHub? Explain briefly.",
            "Can you explain what GitHub is? Keep it brief.",
            "Tell me about GitHub in a few sentences.",
            "What is the purpose of GitHub? Short answer please.",
            "What is Python? Explain briefly."
        ]
    
    results = []
    
    # Process each test question
    for i, question in enumerate(test_questions):
        print(f"\nProcessing question {i+1}/{len(test_questions)}: '{question}'")
        
        # Track time
        start_time = time.time()
        
        # Every third question, use a higher temperature to test cache bypass
        temperature = 0.7 if i % 3 == 0 else 0.0
        
        # Process the question
        try:
            answer = cached_llm(prompt=question, cache_obj=cache, temperature=temperature)
            
            # Track completion time
            elapsed_time = time.time() - start_time
            
            # Print results
            print(f"Temperature: {temperature}")
            print(f"Time taken: {elapsed_time:.4f} seconds")
            print(f"Answer: {answer[:100]}...")  # Show first 100 chars
            
            # Store result
            results.append({
                "question": question,
                "temperature": temperature,
                "time": elapsed_time,
                "answer": answer[:100] + "..."
            })
            
        except Exception as e:
            print(f"Error processing question: {e}")
    
    # Close cache to finalize logs
    cache.close()
    
    # Generate a performance report
    try:
        report_path = generate_latest_run_report(log_dir=cache_dir)
        print(f"\nPerformance report generated: {report_path}")
    except Exception as e:
        print(f"Error generating report: {e}")
    
    return results

def main():
    """Main entry point for the test script."""
    # Create output directory if it doesn't exist
    os.makedirs("cache_logs", exist_ok=True)
    
    # Test questions with variations to test cache behavior
    questions = [
        "What is GitHub?",
        "Can you explain GitHub to me?",
        "What is the purpose of GitHub?",
        "Tell me about Git version control",
        "What is Python programming language?",
        "Explain Python to a beginner",
        "What are some popular Python libraries?",
        "What is TensorFlow?",
        "How does GPT-3 work?",
        "What is the difference between GPT-3 and GPT-4?"
    ]
    
    # Run the test
    run_cache_test(
        cache_dir="cache_logs",
        similarity_threshold=0.75,
        #test_questions=questions
    )

if __name__ == "__main__":
    main()