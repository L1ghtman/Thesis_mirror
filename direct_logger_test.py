import time
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Import the simple logger
from simple_cache_logger import SimpleGPTCacheLogger

# Import GPTCache components
from gptcache.embedding import SBERT
from gptcache.similarity_evaluation import SbertCrossencoderEvaluation
from gptcache.processor.pre import get_prompt
from gptcache.adapter.langchain_models import LangChainLLMs
from gptcache.core import Cache

# Import your project components
from components import custom_llm

# Import for analysis and reporting
from components.cache_analyzer import generate_latest_run_report

def embedding_func(prompt: str, **kwargs) -> tuple:
    """Embedding function for the cache."""
    encoder = SBERT('all-MiniLM-L6-v2')
    return tuple(encoder.to_embeddings(prompt))

def run_cache_test(log_dir: str = "cache_logs", similarity_threshold: float = 0.7,
                  test_questions: Optional[List[str]] = None) -> None:
    """
    Run a test of the cache system with direct performance logging.
    
    Args:
        log_dir: Directory for cache logs
        similarity_threshold: Threshold for similarity evaluation
        test_questions: List of questions to test
    """
    print(f"Starting cache test with similarity threshold: {similarity_threshold}")
    
    # Create the logger
    logger = SimpleGPTCacheLogger(log_dir=log_dir)
    
    # Create and initialize a standard GPTCache
    cache = Cache()
    
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
    
    # Process each test question
    for i, question in enumerate(test_questions):
        print(f"\nProcessing question {i+1}/{len(test_questions)}: '{question}'")
        
        # Start tracking request with logger
        context = logger.log_start_request(question, metadata={
            "question_number": i+1,
            "similarity_threshold": similarity_threshold
        })
        
        # Determine if we should use a temperature for this question
        temperature = 0.7 if i % 3 == 0 else 0.0
        
        try:
            # If using temperature, log direct LLM usage
            if temperature > 0:
                context = logger.log_llm_direct(context, temperature)
                
                # Call LLM directly
                start_time = time.time()
                answer = llm.invoke(question)
                elapsed_time = time.time() - start_time
                
                # Log completion with timing information
                context["response_time"] = elapsed_time
                logger.log_completion(context, answer)
                
                print(f"Direct LLM call (temp={temperature}):")
                print(f"Time: {elapsed_time:.4f}s")
                print(f"Answer: {answer[:100]}...")
            
            else:
                # Simply use the cached LLM and infer whether it was a hit or miss
                # from the response time
                start_time = time.time()
                answer = cached_llm(prompt=question, cache_obj=cache)
                elapsed_time = time.time() - start_time
                
                # Use a simple heuristic to guess if it was a cache hit:
                # If response was very fast, it was likely a cache hit
                estimated_cache_hit = elapsed_time < 5.0  # Using 1 second as a threshold
                
                if estimated_cache_hit:
                    # Log as a positive cache hit (we don't know similarity score)
                    context = logger.log_cache_check(
                        context, 
                        cache_hit=True,
                        positive_hit=True
                    )
                    print(f"Estimated Cache HIT:")
                else:
                    # Log as a cache miss
                    context = logger.log_cache_check(context, cache_hit=False)
                    print(f"Estimated Cache MISS:")
                
                # Log completion
                logger.log_completion(context, answer)
                
                print(f"Time: {elapsed_time:.4f}s")
                print(f"Answer: {answer[:100]}...")
        
        except Exception as e:
            # Log any errors
            logger.log_error(context, e)
            print(f"Error processing question: {e}")
    
    # Close logger to finalize logs
    logger.close()
    
    # Generate a performance report
    try:
        report_path = generate_latest_run_report(log_dir=log_dir)
        print(f"\nPerformance report generated: {report_path}")
    except Exception as e:
        print(f"Error generating report: {e}")

def main():
    """Main entry point for the test script."""
    # Create output directories if they don't exist
    os.makedirs("cache_logs", exist_ok=True)
    os.makedirs("cache_reports", exist_ok=True)
    
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
        log_dir="cache_logs",
        similarity_threshold=0.75,
        #test_questions=questions
    )

if __name__ == "__main__":
    main()