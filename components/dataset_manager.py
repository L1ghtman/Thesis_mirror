#!/usr/bin/env python
# dataset_manager.py - A module to manage datasets for LLM cache benchmarking

import os
import json
import random
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from datasets import load_dataset, Dataset

class DatasetManager:
    """Manager for loading and processing datasets for LLM cache benchmarking."""
    
    def __init__(self, cache_dir: str = "./dataset_cache"):
        """
        Initialize the dataset manager.
        
        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.cache_dir = cache_dir
        self.datasets = {}
        self.active_dataset = None
        self.active_dataset_name = None
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    
    def load_msmarco(self, split: str = "train", max_samples: Optional[int] = 1000) -> str:
        """
        Load the MS MARCO dataset.
        
        Args:
            split: Dataset split to load ('train', 'validation', 'test')
            max_samples: Maximum number of samples to load (None for all)
            
        Returns:
            Name of the loaded dataset
        """
        dataset_name = f"msmarco_{split}"
        
        try:
            # Load the dataset
            ds = load_dataset("microsoft/ms_marco", "v1.1", split=split, cache_dir=self.cache_dir)
            
            # Subsample if needed
            if max_samples and max_samples < len(ds):
                ds = ds.select(range(max_samples))
            
            # Store the dataset
            self.datasets[dataset_name] = {
                "data": ds,
                "type": "qa",
                "description": f"MS MARCO {split} dataset (max {max_samples} samples)"
            }
            
            print(f"Loaded {len(ds)} samples from MS MARCO {split} dataset")
            return dataset_name
        
        except Exception as e:
            print(f"Error loading MS MARCO dataset: {e}")
            return None
    
    def load_squad(self, split: str = "train", max_samples: Optional[int] = 1000) -> str:
        """
        Load the SQuAD dataset.
        
        Args:
            split: Dataset split to load ('train', 'validation')
            max_samples: Maximum number of samples to load (None for all)
            
        Returns:
            Name of the loaded dataset
        """
        dataset_name = f"squad_{split}"
        
        try:
            # Load the dataset
            ds = load_dataset("squad", split=split, cache_dir=self.cache_dir)
            
            # Subsample if needed
            if max_samples and max_samples < len(ds):
                ds = ds.select(range(max_samples))
            
            # Store the dataset
            self.datasets[dataset_name] = {
                "data": ds,
                "type": "qa",
                "description": f"SQuAD {split} dataset (max {max_samples} samples)"
            }
            
            print(f"Loaded {len(ds)} samples from SQuAD {split} dataset")
            return dataset_name
        
        except Exception as e:
            print(f"Error loading SQuAD dataset: {e}")
            return None
    
    def load_natural_questions(self, split: str = "train", max_samples: Optional[int] = 1000) -> str:
        """
        Load the Natural Questions dataset.
        
        Args:
            split: Dataset split to load ('train', 'validation')
            max_samples: Maximum number of samples to load (None for all)
            
        Returns:
            Name of the loaded dataset
        """
        dataset_name = f"natural_questions_{split}"
        
        try:
            # Load the dataset
            ds = load_dataset("natural_questions", split=split, cache_dir=self.cache_dir)
            
            # Subsample if needed
            if max_samples and max_samples < len(ds):
                ds = ds.select(range(max_samples))
            
            # Store the dataset
            self.datasets[dataset_name] = {
                "data": ds,
                "type": "qa",
                "description": f"Natural Questions {split} dataset (max {max_samples} samples)"
            }
            
            print(f"Loaded {len(ds)} samples from Natural Questions {split} dataset")
            return dataset_name
        
        except Exception as e:
            print(f"Error loading Natural Questions dataset: {e}")
            return None
    
    def load_yahoo_answers(self, split: str = "train", max_samples: Optional[int] = 1000) -> str:
        """
        Load the Yahoo Answers dataset.
        
        Args:
            split: Dataset split to load ('train', 'test')
            max_samples: Maximum number of samples to load (None for all)
            
        Returns:
            Name of the loaded dataset
        """
        dataset_name = f"yahoo_answers_{split}"
        
        try:
            # Load the dataset
            ds = load_dataset("yahoo_answers_topics", split=split, cache_dir=self.cache_dir)
            
            # Subsample if needed
            if max_samples and max_samples < len(ds):
                ds = ds.select(range(max_samples))
            
            # Store the dataset
            self.datasets[dataset_name] = {
                "data": ds,
                "type": "qa",
                "description": f"Yahoo Answers {split} dataset (max {max_samples} samples)"
            }
            
            print(f"Loaded {len(ds)} samples from Yahoo Answers {split} dataset")
            return dataset_name
        
        except Exception as e:
            print(f"Error loading Yahoo Answers dataset: {e}")
            return None
    
    def load_quora_question_pairs(self, split: str = "train", max_samples: Optional[int] = 1000) -> str:
        """
        Load the Quora question-pairs dataset
        
        Args:
            max_samples: Maximum number of samples to load (None for all)
        Returns:
            Name of the loaded dataset"""
        
        dataset_name = f"quora_question_pairs_{split}"
        
        try:
            ds = load_dataset("AlekseyKorshuk/quora-question-pairs", split=split, cache_dir=self.cache_dir)
            print(type(ds))
            print(ds)

            if max_samples and max_samples < len(ds):
                ds = ds.select(range(max_samples))
            
            self.datasets[dataset_name] = {
                "data": ds,
                "type": "qp",
                "description": f"Quora question-pairs {split} dataset (max {max_samples} samples)"
            }

            print(f"Loaded {len(ds)} samples from Quora question-pairs {split} dataset")
            return dataset_name

        except Exception as e:
            print(f"Error loading Quora question-pairs: {e}")
            return None
    
    def load_custom(self, dataset_name: str, questions: List[str], answers: Optional[List[str]] = None) -> str:
        """
        Load a custom dataset from provided questions and answers.
        
        Args:
            dataset_name: Name to identify the dataset
            questions: List of questions
            answers: Optional list of answers (if None, will be empty strings)
            
        Returns:
            Name of the loaded dataset
        """
        if answers is None:
            answers = [""] * len(questions)
        
        # Ensure equal lengths
        if len(questions) != len(answers):
            raise ValueError("Questions and answers must have the same length")
        
        # Create a dataset dict
        data_dict = {
            "question": questions,
            "answer": answers
        }
        
        # Convert to HuggingFace Dataset
        ds = Dataset.from_dict(data_dict)
        
        # Store the dataset
        self.datasets[dataset_name] = {
            "data": ds,
            "type": "qa",
            "description": f"Custom dataset with {len(ds)} samples"
        }
        
        print(f"Loaded custom dataset '{dataset_name}' with {len(ds)} samples")
        return dataset_name
    
    def load_from_file(self, file_path: str, dataset_name: Optional[str] = None) -> str:
        """
        Load a dataset from a JSON or text file.
        
        Args:
            file_path: Path to the file
            dataset_name: Name to identify the dataset (defaults to filename)
            
        Returns:
            Name of the loaded dataset
        """
        if dataset_name is None:
            dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        
        try:
            # Check file extension
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == '.json':
                # Load JSON file
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle different formats
                if isinstance(data, list):
                    # Assume list of dictionaries with 'question' and optionally 'answer' keys
                    questions = [item.get('question', '') for item in data]
                    answers = [item.get('answer', '') for item in data]
                elif isinstance(data, dict):
                    # Assume dict with 'questions' and optionally 'answers' keys
                    questions = data.get('questions', [])
                    answers = data.get('answers', [''] * len(questions))
                else:
                    raise ValueError("Unsupported JSON format")
                
                return self.load_custom(dataset_name, questions, answers)
            
            elif ext == '.txt':
                # Load text file - assume one question per line
                with open(file_path, 'r', encoding='utf-8') as f:
                    questions = [line.strip() for line in f if line.strip()]
                
                return self.load_custom(dataset_name, questions)
            
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
        
        except Exception as e:
            print(f"Error loading dataset from file: {e}")
            return None
    
    def set_active_dataset(self, dataset_name: str) -> bool:
        """
        Set the active dataset for benchmarking.
        
        Args:
            dataset_name: Name of the dataset to set as active
            
        Returns:
            True if successful, False otherwise
        """
        if dataset_name not in self.datasets:
            print(f"Dataset '{dataset_name}' not found")
            return False
        
        self.active_dataset = self.datasets[dataset_name]["data"]
        self.active_dataset_name = dataset_name
        print(f"Active dataset set to '{dataset_name}'")
        return True
    
    def get_questions(self, dataset_name: Optional[str] = None, count: Optional[int] = None, 
                     shuffle: bool = False) -> List[str]:
        """
        Get questions from a dataset.
        
        Args:
            dataset_name: Name of the dataset (uses active dataset if None)
            count: Number of questions to return (all if None)
            shuffle: Whether to shuffle the questions
            
        Returns:
            List of questions
        """
        # Use active dataset if none specified
        if dataset_name is None:
            if self.active_dataset is None:
                raise ValueError("No active dataset. Use set_active_dataset() first.")
            dataset = self.active_dataset
            dataset_info = self.datasets[self.active_dataset_name]
        else:
            if dataset_name not in self.datasets:
                raise ValueError(f"Dataset '{dataset_name}' not found")
            dataset = self.datasets[dataset_name]["data"]
            dataset_info = self.datasets[dataset_name]
        
        # Extract questions based on dataset type
        if dataset_info["type"] == "qa":
            # Find the question field
            question_field = self._guess_question_field(dataset)
            questions = [item[question_field] for item in dataset]
            print(f"[DEBUG] questions type: {type(questions[0])}")
            if type(questions[0]) == dict:
                questions = [q['question'] for q in questions]
        elif dataset_info["type"] == "qp":
            questions = []
            for item in dataset:
                q1 = item['question1']
                q2 = item['question2']
                questions.append(q1)
                questions.append(q2)

        else:
            # Fallback: get the first column
            first_col = list(dataset.features.keys())[0]
            questions = [item[first_col] for item in dataset]
        
        # Shuffle if requested
        if shuffle:
            # Create a copy before shuffling to avoid modifying the original
            questions = questions.copy()
            random.shuffle(questions)
        
        # Limit count if specified
        if count is not None:
            questions = questions[:count]
        
        return questions
    
    def get_question_answer_pairs(self, dataset_name: Optional[str] = None, 
                                  count: Optional[int] = None,
                                  shuffle: bool = False) -> List[Tuple[str, str]]:
        """
        Get question-answer pairs from a dataset.
        
        Args:
            dataset_name: Name of the dataset (uses active dataset if None)
            count: Number of pairs to return (all if None)
            shuffle: Whether to shuffle the pairs
            
        Returns:
            List of (question, answer) tuples
        """
        # Use active dataset if none specified
        if dataset_name is None:
            if self.active_dataset is None:
                raise ValueError("No active dataset. Use set_active_dataset() first.")
            dataset = self.active_dataset
            dataset_info = self.datasets[self.active_dataset_name]
        else:
            if dataset_name not in self.datasets:
                raise ValueError(f"Dataset '{dataset_name}' not found")
            dataset = self.datasets[dataset_name]["data"]
            dataset_info = self.datasets[dataset_name]
        
        # Extract question-answer pairs based on dataset type
        if dataset_info["type"] == "qa":
            # Find the question and answer fields
            question_field = self._guess_question_field(dataset)
            answer_field = self._guess_answer_field(dataset)
            
            if answer_field:
                pairs = [(item[question_field], item[answer_field]) for item in dataset]
            else:
                pairs = [(item[question_field], "") for item in dataset]
        else:
            # Fallback: get the first two columns
            cols = list(dataset.features.keys())
            if len(cols) >= 2:
                pairs = [(item[cols[0]], item[cols[1]]) for item in dataset]
            else:
                pairs = [(item[cols[0]], "") for item in dataset]
        
        # Shuffle if requested
        if shuffle:
            # Create a copy before shuffling to avoid modifying the original
            pairs = pairs.copy()
            random.shuffle(pairs)
        
        # Limit count if specified
        if count is not None:
            pairs = pairs[:count]
        
        return pairs
    
    def _guess_question_field(self, dataset: Dataset) -> str:
        """
        Guess which field contains the questions in a dataset.
        
        Args:
            dataset: The dataset to analyze
            
        Returns:
            Name of the field likely containing questions
        """
        # Common field names for questions
        question_fields = ['question', 'query', 'context', 'text', 'content']

        # Check if any of these fields exist
        for field in question_fields:
            if field in dataset.features:
                return field
        
        # If none found, use the first field
        return list(dataset.features.keys())[0]
    
    def _guess_answer_field(self, dataset: Dataset) -> Optional[str]:
        """
        Guess which field contains the answers in a dataset.
        
        Args:
            dataset: The dataset to analyze
            
        Returns:
            Name of the field likely containing answers, or None if not found
        """
        # Common field names for answers
        answer_fields = ['answer', 'answers', 'response', 'title', 'summary']
        
        # Check if any of these fields exist
        for field in answer_fields:
            if field in dataset.features:
                return field
        
        # If none found but there are at least two fields, use the second field
        keys = list(dataset.features.keys())
        if len(keys) >= 2:
            return keys[1]
        
        return None
    
    def list_datasets(self) -> Dict[str, Dict[str, Any]]:
        """
        List all loaded datasets with their information.
        
        Returns:
            Dictionary of dataset names to their info
        """
        dataset_info = {}
        
        for name, info in self.datasets.items():
            dataset_info[name] = {
                "description": info["description"],
                "type": info["type"],
                "size": len(info["data"]),
                "active": (name == self.active_dataset_name)
            }
        
        return dataset_info
    
    def print_dataset_stats(self, dataset_name: Optional[str] = None) -> None:
        """
        Print statistics about a dataset.
        
        Args:
            dataset_name: Name of the dataset (uses active dataset if None)
        """
        if dataset_name is None:
            if self.active_dataset is None:
                print("No active dataset. Use set_active_dataset() first.")
                return
            dataset_name = self.active_dataset_name
        
        if dataset_name not in self.datasets:
            print(f"Dataset '{dataset_name}' not found")
            return
        
        dataset_info = self.datasets[dataset_name]
        dataset = dataset_info["data"]
        
        # Print basic info
        print(f"Dataset: {dataset_name}")
        print(f"Description: {dataset_info['description']}")
        print(f"Type: {dataset_info['type']}")
        print(f"Size: {len(dataset)} samples")
        
        # Print column info
        print("\nColumns:")
        for key in dataset.features.keys():
            print(f"  - {key}: {dataset.features[key]}")
        
        # Print sample data
        print("\nSample data (first 3 items):")
        for i, item in enumerate(dataset[:3]):
            print(f"\nItem {i+1}:")
            for key, value in item.items():
                # Truncate long values
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                print(f"  {key}: {value}")
                
    def sample_questions(self, dataset_name: Optional[str] = None, 
                         count: int = 5, seed: Optional[int] = None) -> List[str]:
        """
        Sample random questions from a dataset.
        
        Args:
            dataset_name: Name of the dataset (uses active dataset if None)
            count: Number of questions to sample
            seed: Random seed for reproducibility
            
        Returns:
            List of sampled questions
        """
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            
        # Get all questions
        questions = self.get_questions(dataset_name, shuffle=True)
        
        # Sample 'count' questions
        return questions[:count]

    def load(self, dataset: str, **kwargs) -> Optional[str]:
        dataset = dataset.lower().replace('-', '_').replace(' ', '_')

        loader_map = {
            'msmarco': self.load_msmarco,
            'ms_marco': self.load_msmarco,
            'squad': self.load_squad,
            'natural_questions': self.load_natural_questions,
            'nq': self.load_natural_questions,
            'yahoo_answers': self.load_yahoo_answers,
            'yahoo': self.load_yahoo_answers,
            'quora_question_pairs': self.load_quora_question_pairs,
            'quora': self.load_quora_question_pairs,
            'custom': self.load_custom,
            'from_file': self.load_from_file,
            'file': self.load_from_file,
        }

        if dataset not in loader_map:
            print(f"Error: Unknown dataset name '{dataset}'")
            print(f"Supported names: {', '.join(sorted(set(loader_map.keys())))}")
            return None

        loader_func = loader_map[dataset]

        try:
            return loader_func(**kwargs)
        except TypeError as e:
            print(f"Error: Invalid parameters for dataset name '{dataset}': {e}")
            return None
        except Exception as e:
            print(f"Error loading dataset '{dataset}': {e}")
            return None

def create_default_manager():
    """Create a dataset manager with some default datasets."""
    manager = DatasetManager()
    
    # Load some default datasets
    manager.load_custom("github_questions", [
        "What is github? Explain briefly.",
        "can you explain what GitHub is? Explain briefly.",
        "can you tell me more about GitHub? Explain briefly.",
        "what is the purpose of GitHub? Explain briefly.",
        "How do I create a repository on GitHub?",
        "What's the difference between Git and GitHub?",
        "How do I fork a repository?",
        "What is a pull request in GitHub?",
        "How do GitHub Actions work?",
    ])
    
    manager.load_custom("python_questions", [
        "What is Python? Explain briefly.",
        "What are the key features of Python?",
        "How do I install Python on Windows?",
        "What is a Python virtual environment?",
        "What's the difference between Python 2 and Python 3?",
        "How do I handle exceptions in Python?",
        "What are Python decorators?",
        "Explain Python's GIL (Global Interpreter Lock).",
        "How do you manage dependencies in Python?",
        "What is PEP 8 in Python?",
    ])
    
    # Set a default active dataset
    manager.set_active_dataset("github_questions")
    
    return manager


if __name__ == "__main__":
    # Example usage
    manager = create_default_manager()
    
    # Print available datasets
    print("\nAvailable datasets:")
    for name, info in manager.list_datasets().items():
        print(f"- {name}: {info['description']} ({info['size']} samples)")
    
    # Get questions from the active dataset
    questions = manager.get_questions(count=3)
    print("\nSample questions:")
    for q in questions:
        print(f"- {q}")