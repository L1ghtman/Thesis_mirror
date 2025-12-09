import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import networkx as nx
from scipy import stats
from sklearn.decomposition import PCA
from collections import defaultdict
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from config_manager import get_config

pd.set_option('future.no_silent_downcasting', True)

class CachePerformanceAnalyzer:
    """
    Analyzer class for visualizing and reporting on cache performance metrics.
    """
    
    def __init__(self, log_dir: str = "cache_logs", output_dir: str = "cache_reports"):
        """
        Initialize the analyzer with directories for log files and output reports.
        
        Args:
            log_dir: Directory where log files are stored
            output_dir: Directory where reports will be saved
        """
        self.config = get_config()
        self.run_id = self.config.experiment["run_id"]
        self.log_dir = log_dir
        self.output_dir = output_dir
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def load_run_data(self, run_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Load data for a specific run or the latest run if no run_id is provided.
        
        Args:
            run_id: The specific run ID to load, or None for the latest run
            
        Returns:
            The loaded run data as a dictionary
        """
        if self.run_id:
            run_id = self.run_id

        if run_id is not None:
            log_file = os.path.join(self.log_dir, f"cache_run_{run_id}.json")
            if not os.path.exists(log_file):
                raise ValueError(f"Run {run_id} not found")
        else:
            log_files = glob.glob(os.path.join(self.log_dir, "cache_run_*.json"))
            if not log_files:
                raise ValueError("No run data found")
            
            # Get the latest run
            log_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            log_file = log_files[-1]
            run_id = int(log_file.split("_")[-1].split(".")[0])
        
        with open(log_file, 'r') as f:
            try:
                run_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {log_file}")
                # Return minimal valid structure
                return {
                    "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "total_requests": 0,
                    "requests": [],
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "positive_hits": 0,
                    "negative_hits": 0,
                    "llm_direct_calls": 0
                }
            
        # Validate and sanitize run data
        self._ensure_valid_structure(run_data)

        #print("Checking run data right after loading:")
        #print(f"cache analyzer Debug info: {run_data.get('lsh_debug_info', 'No LSH debug info')}")
            
        return run_data
    
    def _ensure_valid_structure(self, run_data: Dict[str, Any]) -> None:
        """
        Ensure the run data has a valid structure with all required fields.
        
        Args:
            run_data: The run data to validate and sanitize
        """
        # Ensure requests array exists
        if "requests" not in run_data:
            run_data["requests"] = []
        
        # Ensure required fields exist
        required_fields = [
            "start_time", "total_requests", "cache_hits", "cache_misses",
            "positive_hits", "negative_hits", "llm_direct_calls", "temperature", "temperature_times", "lsh_debug_info"
        ]
        
        for field in required_fields:
            if field not in run_data:
                if field == "start_time":
                    run_data[field] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                else:
                    run_data[field] = 0
        
        # Validate each request entry
        for i, request in enumerate(run_data["requests"]):
            # Ensure timestamp exists
            if "timestamp" not in request:
                request["timestamp"] = datetime.now().isoformat()
            
            # Ensure query exists
            if "query" not in request:
                request["query"] = f"Unknown query {i}"
            
            # Ensure other required fields exist
            for field in ["event_type", "used_cache", "cache_hit", "positive_hit", "response_time"]:
                if field not in request:
                    if field in ["used_cache", "cache_hit", "positive_hit"]:
                        request[field] = None
                    elif field == "response_time":
                        request[field] = 0.0
                    else:
                        request[field] = "UNKNOWN"
    
    def load_multiple_runs(self, run_ids: Optional[List[int]] = None, n_latest: int = 5) -> List[Dict[str, Any]]:
        """
        Load data for multiple runs.
        
        Args:
            run_ids: Specific run IDs to load, or None to load the latest runs
            n_latest: Number of latest runs to load if run_ids is None
            
        Returns:
            List of loaded run data dictionaries
        """
        if run_ids is not None:
            return [self.load_run_data(run_id) for run_id in run_ids]
        
        log_files = glob.glob(os.path.join(self.log_dir, "cache_run_*.json"))
        if not log_files:
            raise ValueError("No run data found")
        
        # Sort by run ID
        log_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        
        # Take the latest n runs
        latest_files = log_files[-n_latest:]
        
        runs_data = []
        for log_file in latest_files:
            try:
                with open(log_file, 'r') as f:
                    run_data = json.load(f)
                    # Validate and sanitize run data
                    self._ensure_valid_structure(run_data)
                    runs_data.append(run_data)
            except Exception as e:
                print(f"Error loading run data from {log_file}: {e}")
        
        return runs_data
    
    def _prepare_run_dataframe(self, run_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert run requests data to a pandas DataFrame for analysis.
        
        Args:
            run_data: The run data dictionary
            
        Returns:
            DataFrame of the run's request data
        """
        # Ensure we have a valid structure with requests
        self._ensure_valid_structure(run_data)
        
        if not run_data["requests"]:
            # Create a minimal empty dataframe with the expected columns
            return pd.DataFrame(columns=[
                "timestamp", "query", "event_type", "used_cache", 
                "cache_hit", "positive_hit", "response_time", "temperature_times",
                "similarity_score", "temperature", "request_num"
            ])
        
        try:
            df = pd.DataFrame(run_data["requests"])
            
            # Convert timestamps to datetime
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            except:
                # If timestamp conversion fails, create a new timestamp column
                df["timestamp"] = pd.date_range(
                    start=datetime.now(), 
                    periods=len(df), 
                    freq='s'
                )
            
            # Add sequential request number
            df["request_num"] = range(1, len(df) + 1)
            
            return df
        except Exception as e:
            print(f"Error preparing DataFrame: {e}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                "timestamp", "query", "event_type", "used_cache", 
                "cache_hit", "positive_hit", "response_time", "temperature_times",
                "similarity_score", "temperature", "request_num"
            ])
    
    def generate_performance_summary(self, run_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary of performance metrics for a run.
        
        Args:
            run_data: The run data dictionary
            
        Returns:
            Dictionary of summary metrics
        """
        # If summary already exists in the data, return it
        if "summary" in run_data:
            return run_data["summary"]
        
        # Ensure we have a valid structure
        self._ensure_valid_structure(run_data)
        
        # If no requests, return a minimal summary
        if not run_data["requests"]:
            return {
                "run_id": run_data.get("run_id", 1),
                "total_requests": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "positive_hits": 0,
                "negative_hits": 0,
                "llm_direct_calls": 0,
                "avg_cache_time": 0,
                "avg_llm_time": 0,
                #"avg_clustering_time": 0,
                "avg_temperature_time": 0,
                "cache_hit_rate": 0,
                "positive_hit_rate": 0,
                "time_saved": 0
            }
        
        try:
            # Create DataFrame for analysis
            df = self._prepare_run_dataframe(run_data)
            
            # Calculate metrics from DataFrame
            total_requests = len(df)
            cache_hits = len(df[df["cache_hit"] == True])
            cache_misses = len(df[df["cache_hit"] == False])
            positive_hits = len(df[(df["cache_hit"] == True) & (df["positive_hit"] == True)])
            negative_hits = len(df[(df["cache_hit"] == True) & (df["positive_hit"] == False)])
            llm_direct_calls = len(df[df["used_cache"] == False])

            
            # Get response times
            cache_response_times = df[(df["cache_hit"] == True)]["response_time"].tolist()
            llm_response_times = df[(df["cache_hit"] == False) | (df["used_cache"] == False)]["response_time"].tolist()
            #clustering_times = df["clustering_time"].tolist()
            temperature_times = df["temperature_time"].tolist()
            
            # Calculate averages safely
            avg_cache_time = sum(cache_response_times) / len(cache_response_times) if cache_response_times else 0
            avg_llm_time = sum(llm_response_times) / len(llm_response_times) if llm_response_times else 0
            #avg_clustering_time = sum(clustering_times) / len(clustering_times) if clustering_times else 0
            avg_temperature_time = sum(temperature_times) / len(temperature_times) if temperature_times else 0
            
            # Calculate rates
            cache_hit_rate = cache_hits / total_requests if total_requests > 0 else 0
            positive_hit_rate = positive_hits / cache_hits if cache_hits > 0 else 0
            
            # Calculate time saved
            time_saved = (avg_llm_time - avg_cache_time) * positive_hits if avg_llm_time and positive_hits else 0
            
            return {
                "run_id": run_data.get("run_id", 1),
                "total_requests": total_requests,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "positive_hits": positive_hits,
                "negative_hits": negative_hits,
                "llm_direct_calls": llm_direct_calls,
                "avg_cache_time": avg_cache_time,
                "avg_llm_time": avg_llm_time,
                #"avg_clustering_time": avg_clustering_time,
                "avg_temperature_time": avg_temperature_time,
                "cache_hit_rate": cache_hit_rate,
                "positive_hit_rate": positive_hit_rate,
                "time_saved": time_saved
            }
        except Exception as e:
            # Log the error and return a minimal summary
            print(f"Error generating performance summary: {e}")
            traceback.print_exc()
            
            return {
                "run_id": run_data.get("run_id", 1),
                "total_requests": len(run_data.get("requests", [])),
                "cache_hits": run_data.get("cache_hits", 0),
                "cache_misses": run_data.get("cache_misses", 0),
                "positive_hits": run_data.get("positive_hits", 0),
                "negative_hits": run_data.get("negative_hits", 0),
                "llm_direct_calls": run_data.get("llm_direct_calls", 0),
                "avg_cache_time": 0,
                "avg_llm_time": 0,
                #"avg_clustering_time": 0,
                "avg_temperature_time": 0,
                "cache_hit_rate": 0,
                "positive_hit_rate": 0,
                "time_saved": 0
            }
    
    def plot_cache_hit_distribution(self, run_data: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a pie chart showing the distribution of cache hits, misses, and direct LLM calls.
        
        Args:
            run_data: The run data dictionary
            save_path: If provided, the plot will be saved to this path
            
        Returns:
            The matplotlib figure
        """
        try:
            summary = self.generate_performance_summary(run_data)
            
            # Create data for the pie chart
            labels = ['Positive Hits', 'Negative Hits', 'Cache Misses', 'Direct LLM Calls']
            sizes = [
                summary["positive_hits"],
                summary["negative_hits"],
                summary["cache_misses"],
                summary["llm_direct_calls"]
            ]
            
            # Filter out zero values
            non_zero_indices = [i for i, size in enumerate(sizes) if size > 0]
            
            # If all values are zero, create a dummy chart
            if not non_zero_indices:
                fig, ax = plt.subplots(figsize=(10, 7))
                ax.text(0.5, 0.5, "No data available for distribution chart", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=14)
                
                if save_path:
                    plt.savefig(save_path, bbox_inches='tight')
                
                return fig
            
            filtered_labels = [labels[i] for i in non_zero_indices]
            filtered_sizes = [sizes[i] for i in non_zero_indices]
            
            # Set colors
            colors = ['#2ecc71', '#f39c12', '#e74c3c', '#3498db']
            filtered_colors = [colors[i] for i in non_zero_indices]
            
            # Create the pie chart
            fig, ax = plt.subplots(figsize=(10, 7))
            wedges, texts, autotexts = ax.pie(
                filtered_sizes, 
                labels=filtered_labels, 
                colors=filtered_colors,
                autopct='%1.1f%%', 
                startangle=90,
                shadow=False
            )
            
            # Equal aspect ratio ensures that pie is drawn as a circle
            ax.axis('equal')
            
            # Set title
            run_id = run_data.get("summary", {}).get("run_id", summary.get("run_id", "Unknown"))
            plt.title(f'Cache Performance Distribution - Run #{run_id}', size=16)
            
            # Save if a save path is provided
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
            
            return fig
        except Exception as e:
            print(f"Error creating distribution chart: {e}")
            traceback.print_exc()
            
            # Create a simple error figure
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.text(0.5, 0.5, f"Error creating chart: {e}", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14)
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
            
            return fig
    
    def plot_response_time_comparison(self, run_data: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a box plot comparing response times between cache hits and LLM calls.
        
        Args:
            run_data: The run data dictionary
            save_path: If provided, the plot will be saved to this path
            
        Returns:
            The matplotlib figure
        """
        try:
            df = self._prepare_run_dataframe(run_data)
            
            # If no data or only one row, create a dummy chart
            if len(df) <= 1:
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.text(0.5, 0.5, "Not enough data for response time comparison", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=14)
                
                if save_path:
                    plt.savefig(save_path, bbox_inches='tight')
                
                return fig
            
            # Create categories for the plot
            df["response_category"] = "Unknown"
            
            # Handle None values properly by using boolean masking with isnull/notnull
            if "used_cache" in df.columns and "cache_hit" in df.columns and "positive_hit" in df.columns:
                # Positive cache hits
                mask_positive = (
                    df["used_cache"].fillna(False).infer_objects(copy=False) & 
                    df["cache_hit"].fillna(False).infer_objects(copy=False) & 
                    df["positive_hit"].fillna(False).infer_objects(copy=False)
                )
                df.loc[mask_positive, "response_category"] = "Positive Cache Hit"
                
                # Negative cache hits - handle None values in positive_hit
                mask_negative = (
                    df["used_cache"].fillna(False).infer_objects(copy=False) & 
                    df["cache_hit"].fillna(False).infer_objects(copy=False) & 
                    (df["positive_hit"] == False).infer_objects(copy=False)  # Only explicitly False values
                )
                df.loc[mask_negative, "response_category"] = "Negative Cache Hit"
                
                # Cache misses
                mask_miss = (
                    df["used_cache"].fillna(False) & 
                    (df["cache_hit"] == False)  # Only explicitly False values
                )
                df.loc[mask_miss, "response_category"] = "Cache Miss (LLM)"
                
                # Direct LLM calls
                mask_direct = ~df["used_cache"].fillna(False)
                df.loc[mask_direct, "response_category"] = "Direct LLM Call"
            
            
            # Create the plot
            plt.figure(figsize=(12, 8))
            
            # Use seaborn for a nicer looking plot
            sns.set_style("whitegrid")
            
            # Box plot
            ax = sns.boxplot(x="response_category", y="response_time", hue="response_category", data=df, palette="Set3", legend=False)
            
            # Add individual points
            sns.stripplot(x="response_category", y="response_time", data=df, size=4, color=".3", linewidth=0)
            
            # Rotate x-axis labels
            plt.xticks(rotation=45, ha='right')
            
            # Add labels and title
            plt.xlabel("Response Category")
            plt.ylabel("Response Time (seconds)")
            
            run_id = run_data.get("summary", {}).get("run_id", "Unknown")
            plt.title(f"Response Time Comparison - Run #{run_id}", size=16)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save if a save path is provided
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
            
            return plt.gcf()
        except Exception as e:
            print(f"Error creating response time comparison chart: {e}")
            traceback.print_exc()
            
            # Create a simple error figure
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, f"Error creating chart: {e}", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14)
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
            
            return fig
    
    def plot_similarity_score_distribution(self, run_data: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a histogram of similarity scores for cache hits.
        
        Args:
            run_data: The run data dictionary
            save_path: If provided, the plot will be saved to this path
            
        Returns:
            The matplotlib figure
        """
        try:
            df = self._prepare_run_dataframe(run_data)
            
            # Filter to only include rows with similarity scores
            similarity_df = df[df["similarity_score"].notna()].copy()
            
            if len(similarity_df) == 0:
                # No similarity scores available
                fig, ax = plt.subplots(figsize=(10, 7))
                ax.text(0.5, 0.5, "No similarity score data available", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=14)
                if save_path:
                    plt.savefig(save_path, bbox_inches='tight')
                return fig
            
            # Create the plot
            plt.figure(figsize=(12, 8))
            
            # Use seaborn for a nicer looking plot
            sns.set_style("whitegrid")
            
            # Check if positive_hit column exists and has valid values
            if 'positive_hit' in similarity_df.columns and not similarity_df['positive_hit'].isna().all():
                # Create a histogram with positive/negative hit coloring
                ax = sns.histplot(
                    data=similarity_df,
                    x="similarity_score",
                    hue="positive_hit",
                    palette=["#e74c3c", "#2ecc71"],
                    element="step",
                    bins=20
                )
                plt.legend(title="Hit Type")
            else:
                # If no positive_hit information, plot without hue
                ax = sns.histplot(
                    data=similarity_df,
                    x="similarity_score",
                    color="#3498db",  # Blue color
                    element="step",
                    bins=20
                )
            
            # Add a vertical line at the typical threshold
            threshold = 0.7  # This should be the same threshold used in your system
            plt.axvline(x=threshold, color='purple', linestyle='--', label=f'Threshold ({threshold})')
            
            # Add labels and title
            plt.xlabel("Similarity Score")
            plt.ylabel("Count")
            
            run_id = run_data.get("summary", {}).get("run_id", "Unknown")
            plt.title(f"Similarity Score Distribution - Run #{run_id}", size=16)
            
            # Add the threshold to the legend if it's not already there
            handles, labels = plt.gca().get_legend_handles_labels()
            if f'Threshold ({threshold})' not in labels:
                plt.legend()
            
            # Adjust layout
            plt.tight_layout()
            
            # Save if a save path is provided
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
            
            return plt.gcf()
        except Exception as e:
            print(f"Error creating similarity score distribution chart: {e}")
            traceback.print_exc()
            
            # Create a simple error figure
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, f"Error creating chart: {e}", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14)
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
            
            return fig

    def plot_performance_comparison(self, runs_data: List[Dict[str, Any]], save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a bar chart comparing performance metrics across multiple runs.
        
        Args:
            runs_data: List of run data dictionaries
            save_path: If provided, the plot will be saved to this path
            
        Returns:
            The matplotlib figure
        """
        try:
            # Extract run IDs and performance metrics
            run_ids = []
            cache_hit_rates = []
            positive_hit_rates = []
            avg_cache_times = []
            avg_llm_times = []
            
            for run_data in runs_data:
                summary = self.generate_performance_summary(run_data)
                run_id = run_data.get("summary", {}).get("run_id", summary.get("run_id", "Unknown"))
                
                run_ids.append(str(run_id))
                cache_hit_rates.append(summary["cache_hit_rate"] * 100)  # Convert to percentage
                positive_hit_rates.append(summary["positive_hit_rate"] * 100)  # Convert to percentage
                avg_cache_times.append(summary["avg_cache_time"])
                avg_llm_times.append(summary["avg_llm_time"])
            
            # If no runs, create a dummy chart
            if not run_ids:
                fig, ax = plt.subplots(figsize=(14, 12))
                ax.text(0.5, 0.5, "No run data available for comparison", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=14)
                
                if save_path:
                    plt.savefig(save_path, bbox_inches='tight')
                
                return fig
            
            # Create the plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
            
            # First subplot: Hit rates
            x = range(len(run_ids))
            width = 0.35
            
            bar1 = ax1.bar(x, cache_hit_rates, width, label='Cache Hit Rate', color='#3498db')
            bar2 = ax1.bar([i + width for i in x], positive_hit_rates, width, label='Positive Hit Rate', color='#2ecc71')
            
            ax1.set_xlabel('Run ID')
            ax1.set_ylabel('Percentage (%)')
            ax1.set_title('Hit Rates Comparison Across Runs')
            ax1.set_xticks([i + width/2 for i in x])
            ax1.set_xticklabels(run_ids)
            ax1.legend()
            
            # Add data labels
            for bar in [bar1, bar2]:
                for rect in bar:
                    height = rect.get_height()
                    ax1.annotate(f'{height:.1f}%',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
            
            # Second subplot: Response times
            bar3 = ax2.bar(x, avg_cache_times, width, label='Avg Cache Response Time', color='#9b59b6')
            bar4 = ax2.bar([i + width for i in x], avg_llm_times, width, label='Avg LLM Response Time', color='#e74c3c')
            
            ax2.set_xlabel('Run ID')
            ax2.set_ylabel('Response Time (seconds)')
            ax2.set_title('Response Times Comparison Across Runs')
            ax2.set_xticks([i + width/2 for i in x])
            ax2.set_xticklabels(run_ids)
            ax2.legend()
            
            # Add data labels
            for bar in [bar3, bar4]:
                for rect in bar:
                    height = rect.get_height()
                    ax2.annotate(f'{height:.3f}s',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save if a save path is provided
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
            
            return fig
        except Exception as e:
            print(f"Error creating performance comparison chart: {e}")
            traceback.print_exc()
            
            # Create a simple error figure
            fig, ax = plt.subplots(figsize=(14, 12))
            ax.text(0.5, 0.5, f"Error creating chart: {e}", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14)
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
            
            return fig
    
    def generate_html_report(self, run_data: Dict[str, Any], include_plots: bool = True) -> str:
        """
        Generate an HTML report for a run with enhanced styling.

        Args:
            run_data: The run data dictionary
            include_plots: Whether to include plots in the report

        Returns:
            HTML report as a string
        """
        try:
            # Ensure we have a valid structure
            self._ensure_valid_structure(run_data)

            summary = self.generate_performance_summary(run_data)
            run_id = summary.get("run_id", "Unknown")

            #print("####################")
            #print(run_data["lsh_debug_info"])
            #print("####################")

            # Prepare plots if needed
            plot_paths = {}
            if include_plots:
                os.makedirs(os.path.join(self.output_dir, f"run_{run_id}"), exist_ok=True)

                # Generate and save plots
                dist_path = os.path.join(self.output_dir, f"run_{run_id}", "distribution.png")
                self.plot_cache_hit_distribution(run_data, dist_path)
                plot_paths["distribution"] = os.path.basename(dist_path)

                time_path = os.path.join(self.output_dir, f"run_{run_id}", "response_times.png")
                self.plot_response_time_comparison(run_data, time_path)
                plot_paths["response_times"] = os.path.basename(time_path)

                similarity_path = os.path.join(self.output_dir, f"run_{run_id}", "similarity.png")
                self.plot_similarity_score_distribution(run_data, similarity_path)
                plot_paths["similarity"] = os.path.basename(similarity_path)

                temperature_path = os.path.join(self.output_dir, f"run_{run_id}", "temperature.png")
                self.plot_temperature_analysis(run_data, temperature_path)
                plot_paths["temperature"] = os.path.basename(temperature_path)
                
                lsh_plot_paths = {}

                # Temperature dynamics
                temp_dynamics_path = os.path.join(self.output_dir, f"run_{run_id}", "lsh_temperature_dynamics.png")
                self.plot_temperature_dynamics(run_data, temp_dynamics_path)
                lsh_plot_paths["temperature_dynamics"] = os.path.basename(temp_dynamics_path)

                # Performance metrics
                perf_metrics_path = os.path.join(self.output_dir, f"run_{run_id}", "lsh_performance_metrics.png")
                self.plot_lsh_performance_metrics(run_data, perf_metrics_path)
                lsh_plot_paths["performance_metrics"] = os.path.basename(perf_metrics_path)

                # System behavior
                system_behavior_path = os.path.join(self.output_dir, f"run_{run_id}", "lsh_system_behavior.png")
                self.plot_lsh_system_behavior(run_data, system_behavior_path)
                lsh_plot_paths["system_behavior"] = os.path.basename(system_behavior_path)

                # Adaptive behavior
                adaptive_path = os.path.join(self.output_dir, f"run_{run_id}", "lsh_adaptive_behavior.png")
                self.plot_lsh_adaptive_behavior(run_data, adaptive_path)
                lsh_plot_paths["adaptive_behavior"] = os.path.basename(adaptive_path)

            # Calculate performance stats for dashboard
            total_requests = summary["total_requests"]
            cache_hits = summary["cache_hits"]
            positive_hits = summary["positive_hits"]
            time_saved = summary["time_saved"]
            avg_cache_time = summary["avg_cache_time"]
            avg_llm_time = summary["avg_llm_time"]
            total_time = summary["total_time"]
            start_time = run_data["start_time"]
            #avg_cluster_time = summary["avg_cluster_time"]
            avg_temperature_time = summary["avg_temperature_time"]

            # Speed improvement calculation
            speed_improvement = avg_llm_time / avg_cache_time if avg_cache_time > 0 else 0

            # Build HTML content with enhanced styling
            html = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>GPTCache Performance Report - Run #{run_id}</title>
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
                <style>
                    :root {{
                        --primary: #3a86ff;
                        --primary-light: #e9f0ff;
                        --success: #38b000;
                        --warning: #ffb703;
                        --danger: #ff5a5f;
                        --dark: #252525;
                        --text: #333333;
                        --light-text: #6c757d;
                        --light-bg: #f8f9fa;
                        --lighter-bg: #ffffff;
                        --border: #e9ecef;
                        --shadow: rgba(0, 0, 0, 0.05);
                    }}

                    * {{
                        margin: 0;
                        padding: 0;
                        box-sizing: border-box;
                    }}

                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        line-height: 1.6;
                        color: var(--text);
                        background-color: var(--light-bg);
                        padding: 0;
                        margin: 0;
                    }}

                    .container {{
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 0 20px;
                    }}

                    .header {{
                        background-color: var(--primary);
                        color: white;
                        padding: 20px 0;
                        margin-bottom: 30px;
                        box-shadow: 0 4px 12px var(--shadow);
                    }}

                    .header h1 {{
                        font-size: 2.2rem;
                        margin-bottom: 5px;
                    }}

                    .header p {{
                        opacity: 0.9;
                        font-size: 1.1rem;
                    }}

                    .dashboard {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                        gap: 20px;
                        margin-bottom: 30px;
                    }}

                    .metric-card {{
                        background-color: var(--lighter-bg);
                        border-radius: 12px;
                        padding: 24px;
                        box-shadow: 0 4px 12px var(--shadow);
                        transition: transform 0.3s ease, box-shadow 0.3s ease;
                        position: relative;
                        overflow: hidden;
                    }}

                    .metric-card:hover {{
                        transform: translateY(-5px);
                        box-shadow: 0 8px 24px var(--shadow);
                    }}

                    .metric-card i {{
                        position: absolute;
                        top: 20px;
                        right: 20px;
                        font-size: 2rem;
                        opacity: 0.15;
                    }}

                    .metric-label {{
                        font-size: 1rem;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                        font-weight: 600;
                        color: var(--light-text);
                        margin-bottom: 8px;
                    }}

                    .metric-value {{
                        font-size: 2.2rem;
                        font-weight: 700;
                        margin-bottom: 5px;
                    }}

                    .metric-context {{
                        font-size: 0.95rem;
                        color: var(--light-text);
                    }}

                    .primary {{ color: var(--primary); }}
                    .success {{ color: var(--success); }}
                    .warning {{ color: var(--warning); }}
                    .danger {{ color: var(--danger); }}

                    .section {{
                        background-color: var(--lighter-bg);
                        border-radius: 12px;
                        padding: 30px;
                        margin-bottom: 30px;
                        box-shadow: 0 4px 12px var(--shadow);
                    }}

                    .section-title {{
                        font-size: 1.6rem;
                        margin-bottom: 20px;
                        color: var(--dark);
                        padding-bottom: 10px;
                        border-bottom: 2px solid var(--primary-light);
                    }}

                    .sub-title {{
                        font-size: 1.3rem;
                        margin: 25px 0 15px 0;
                        color: var(--dark);
                    }}

                    .metrics-grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                        gap: 20px;
                        margin-bottom: 20px;
                    }}

                    .stat-item {{
                        background-color: var(--primary-light);
                        border-radius: 10px;
                        padding: 15px;
                    }}

                    .stat-label {{
                        font-size: 0.9rem;
                        color: var(--light-text);
                        margin-bottom: 5px;
                    }}

                    .stat-value {{
                        font-size: 1.2rem;
                        font-weight: 600;
                    }}

                    .plot-grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                        gap: 30px;
                        margin-top: 20px;
                    }}

                    .plot-container {{
                        margin-bottom: 20px;
                        background-color: var(--lighter-bg);
                        border-radius: 12px;
                        padding: 20px;
                        box-shadow: 0 4px 12px var(--shadow);
                    }}

                    .plot-container h3 {{
                        font-size: 1.2rem;
                        margin-bottom: 15px;
                        color: var(--dark);
                    }}

                    .plot-container img {{
                        width: 100%;
                        border-radius: 8px;
                        transition: transform 0.3s ease;
                    }}

                    .plot-container img:hover {{
                        transform: scale(1.02);
                    }}

                    table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin: 20px 0;
                        font-size: 0.95rem;
                    }}

                    th, td {{
                        padding: 15px;
                        text-align: left;
                        border-bottom: 1px solid var(--border);
                    }}

                    th {{
                        background-color: var(--primary-light);
                        color: var(--primary);
                        font-weight: 600;
                        position: sticky;
                        top: 0;
                    }}

                    tr:hover {{
                        background-color: rgba(58, 134, 255, 0.05);
                    }}

                    .table-container {{
                        max-height: 500px;
                        overflow-y: auto;
                        border-radius: 8px;
                        border: 1px solid var(--border);
                    }}

                    .badge {{
                        display: inline-block;
                        padding: 4px 8px;
                        border-radius: 30px;
                        font-size: 0.8rem;
                        font-weight: 600;
                        text-transform: uppercase;
                    }}

                    .badge-hit {{
                        background-color: rgba(56, 176, 0, 0.15);
                        color: var(--success);
                    }}

                    .badge-miss {{
                        background-color: rgba(255, 90, 95, 0.15);
                        color: var(--danger);
                    }}

                    .badge-direct {{
                        background-color: rgba(255, 183, 3, 0.15);
                        color: var(--warning);
                    }}

                    footer {{
                        text-align: center;
                        padding: 20px;
                        color: var(--light-text);
                        font-size: 0.9rem;
                        border-top: 1px solid var(--border);
                        margin-top: 30px;
                    }}

                    .progress-bar {{
                        height: 10px;
                        background-color: var(--border);
                        border-radius: 5px;
                        margin-top: 5px;
                        overflow: hidden;
                    }}

                    .progress-fill {{
                        height: 100%;
                        border-radius: 5px;
                    }}

                    /* Responsive adjustments */
                    @media (max-width: 768px) {{
                        .dashboard {{
                            grid-template-columns: 1fr;
                        }}

                        .plot-grid {{
                            grid-template-columns: 1fr;
                        }}

                        .metric-value {{
                            font-size: 1.8rem;
                        }}
                    }}
                </style>
            </head>
            <body>
                <div class="header">
                    <div class="container">
                        <h1><i class="fas fa-chart-line"></i> GPTCache Performance Report</h1>
                        <p>Run #{run_id} | Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                        <p>Start time:      {start_time}</p>
                        <p>Total duration:  {total_time}</p>
                    </div>
                </div>

                <div class="container">
                    <!-- Key Metrics Dashboard -->
                    <div class="dashboard">
                        <div class="metric-card">
                            <i class="fas fa-server"></i>
                            <div class="metric-label">Total Requests</div>
                            <div class="metric-value primary">{total_requests}</div>
                            <div class="metric-context">Processed through cache</div>
                        </div>

                        <div class="metric-card">
                            <i class="fas fa-bolt"></i>
                            <div class="metric-label">Cache Hit Rate</div>
                            <div class="metric-value {self._get_rating_class(summary["cache_hit_rate"], [0.3, 0.7])}">
                                {summary["cache_hit_rate"]:.1%}
                            </div>
                            <div class="metric-context">
                                {cache_hits} hits out of {total_requests} requests
                            </div>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: {summary["cache_hit_rate"] * 100}%; background-color: {self._get_progress_color(summary["cache_hit_rate"], [0.3, 0.7])}"></div>
                            </div>
                        </div>

                        <div class="metric-card">
                            <i class="fas fa-tachometer-alt"></i>
                            <div class="metric-label">Speed Improvement</div>
                            <div class="metric-value success">{speed_improvement:.1f}x</div>
                            <div class="metric-context">
                                Cache: {avg_cache_time:.3f}s vs LLM: {avg_llm_time:.3f}s
                            </div>
                        </div>

                        <div class="metric-card">
                            <i class="fas fa-clock"></i>
                            <div class="metric-label">Time Saved</div>
                            <div class="metric-value success">{time_saved:.2f}s</div>
                            <div class="metric-context">
                                From {positive_hits} positive cache hits
                            </div>
                        </div>
                    </div>

                    <!-- Detailed Performance Section -->
                    <div class="section">
                        <h2 class="section-title">Detailed Performance Metrics</h2>

                        <div class="metrics-grid">
                            <div class="stat-item">
                                <div class="stat-label">Positive Hits</div>
                                <div class="stat-value success">{summary["positive_hits"]}</div>
                            </div>

                            <div class="stat-item">
                                <div class="stat-label">Negative Hits</div>
                                <div class="stat-value warning">{summary["negative_hits"]}</div>
                            </div>

                            <div class="stat-item">
                                <div class="stat-label">Cache Misses</div>
                                <div class="stat-value danger">{summary["cache_misses"]}</div>
                            </div>

                            <div class="stat-item">
                                <div class="stat-label">Direct LLM Calls</div>
                                <div class="stat-value primary">{summary["llm_direct_calls"]}</div>
                            </div>

                            <div class="stat-item">
                                <div class="stat-label">Positive Hit Rate</div>
                                <div class="stat-value {self._get_rating_class(summary["positive_hit_rate"], [0.7, 0.9])}">
                                    {summary["positive_hit_rate"]:.1%}
                                </div>
                            </div>

                            <div class="stat-item">
                                <div class="stat-label">Avg Cache Time</div>
                                <div class="stat-value">{summary["avg_cache_time"]:.3f}s</div>
                            </div>

                            <div class="stat-item">
                                <div class="stat-label">Avg Temperature Time</div>
                                <div class="stat-value">{summary["avg_temperature_time"]:.5f}s</div>
                            </div>
                        </div>
                    </div>
            """

            # Add plots if included
            if include_plots:
                html += """
                    <!-- Performance Visualizations -->
                    <div class="section">
                        <h2 class="section-title">Performance Visualizations</h2>

                        <div class="plot-container">
                            <h3><i class="fas fa-chart-pie"></i> Cache Hit Distribution</h3>
                            <img src="{}" alt="Cache Hit Distribution">
                        </div>

                        <div class="plot-container">
                            <h3><i class="fas fa-chart-line"></i> Similarity Score Distribution</h3>
                            <img src="{}" alt="Similarity Score Distribution">
                        </div>

                        <div class="plot-container">
                            <h3><i class="fas fa-cubes"></i> Temperature Analysis</h3>
                            <img src="{}" alt="Temperature Analysis">
                        </div>
                    </div>
                    <!-- LSH Algorithm Analysis -->
                    <div class="section">
                        <h2 class="section-title">LSH Algorithm Performance Analysis</h2>

                        <div class="plot-container">
                            <h3><i class="fas fa-temperature-high"></i> Temperature Dynamics</h3>
                            <p>Analysis of temperature evolution and its relationship with cache decisions.</p>
                            <img src="{}" alt="LSH Temperature Dynamics">
                        </div>

                        <div class="plot-container">
                            <h3><i class="fas fa-tachometer-alt"></i> Performance Metrics</h3>
                            <p>LSH computation efficiency and cache factor effectiveness.</p>
                            <img src="{}" alt="LSH Performance Metrics">
                        </div>

                        <div class="plot-container">
                            <h3><i class="fas fa-network-wired"></i> System Behavior</h3>
                            <p>Bucket lifecycle, density patterns, and system-wide correlations.</p>
                            <img src="{}" alt="LSH System Behavior">
                        </div>

                        <div class="plot-container">
                            <h3><i class="fas fa-brain"></i> Adaptive Behavior</h3>
                            <p>Simulated annealing patterns and exploration vs exploitation phases.</p>
                            <img src="{}" alt="LSH Adaptive Behavior">
                        </div>
                    </div>
                """.format(
                    plot_paths.get("distribution", "#"),
                    plot_paths.get("similarity", "#"),
                    plot_paths.get("temperature", "#"),
                    plot_paths.get("embedding_magnitudes", "#"),
                    lsh_plot_paths.get("temperature_dynamics", "#"),
                    lsh_plot_paths.get("performance_metrics", "#"),
                    lsh_plot_paths.get("system_behavior", "#"),
                    lsh_plot_paths.get("adaptive_behavior", "#")
                )

            # Get request details from DataFrame for the table
            df = self._prepare_run_dataframe(run_data)

            # If we have request data, add the table
            if len(df) > 0:
                df = df.head(20)  # Take first 20 for the report

                html += """
                    <!-- Request Details Section -->
                    <div class="section">
                        <h2 class="section-title">Request Details</h2>
                        <p>Showing the first 20 requests processed during this run</p>

                        <div class="table-container">
                            <table>
                                <thead>
                                    <tr>
                                        <th>#</th>
                                        <th>Query</th>
                                        <th>Event Type</th>
                                        <th>Response Time</th>
                                        <th>Similarity</th>
                                    </tr>
                                </thead>
                                <tbody>
                """

                for _, row in df.iterrows():
                    # Truncate query for display
                    query = row.get("query", "")
                    if isinstance(query, str):
                        query = query[:50] + "..." if len(query) > 50 else query
                    else:
                        query = str(query)

                    # Determine badge class based on event type
                    event_type = row.get("event_type", "UNKNOWN")
                    badge_class = "badge-direct"

                    if "HIT" in event_type:
                        badge_class = "badge-hit"
                    elif "MISS" in event_type:
                        badge_class = "badge-miss"

                    # Format similarity score
                    similarity = row.get("similarity_score", None)
                    if similarity is not None and pd.notna(similarity):
                        similarity_display = f"{similarity:.3f}"
                    else:
                        similarity_display = "N/A"

                    # Format row
                    html += f"""
                        <tr>
                            <td>{row.get("request_num", "-")}</td>
                            <td>{query}</td>
                            <td><span class="badge {badge_class}">{event_type}</span></td>
                            <td>{row.get("response_time", 0):.3f}s</td>
                            <td>{similarity_display}</td>
                        </tr>
                    """

                html += """
                            </tbody>
                        </table>
                    </div>
                </div>
                """
            else:
                html += """
                    <!-- No Request Data Message -->
                    <div class="section">
                        <h2 class="section-title">Request Details</h2>
                        <p>No request data available for this run.</p>
                    </div>
                """

            # Close HTML
            html += """
                    <footer>
                        <p>Generated by GPTCache Performance Analyzer on {}</p>
                        <p><i class="fas fa-code"></i> LLM Caching Project</p>
                    </footer>
                </div>

                <script>
                    // Simple JavaScript for interactivity
                    document.addEventListener('DOMContentLoaded', function() {{
                        // Add hover effect to table rows
                        const rows = document.querySelectorAll('tbody tr');
                        rows.forEach(row => {{
                            row.addEventListener('mouseenter', function() {{
                                this.style.backgroundColor = 'rgba(58, 134, 255, 0.1)';
                            }});
                            row.addEventListener('mouseleave', function() {{
                                this.style.backgroundColor = '';
                            }});
                        }});

                        // Make plot images enlargeable on click
                        const plots = document.querySelectorAll('.plot-container img');
                        plots.forEach(plot => {{
                            plot.style.cursor = 'pointer';
                            plot.addEventListener('click', function() {{
                                this.classList.toggle('enlarged');
                                if (this.classList.contains('enlarged')) {{
                                    this.style.position = 'fixed';
                                    this.style.top = '50%';
                                    this.style.left = '50%';
                                    this.style.transform = 'translate(-50%, -50%) scale(1.5)';
                                    this.style.maxWidth = '90vw';
                                    this.style.maxHeight = '90vh';
                                    this.style.zIndex = '1000';
                                    this.style.boxShadow = '0 0 0 1000px rgba(0,0,0,0.7)';
                                }} else {{
                                    this.style.position = '';
                                    this.style.top = '';
                                    this.style.left = '';
                                    this.style.transform = '';
                                    this.style.maxWidth = '';
                                    this.style.maxHeight = '';
                                    this.style.zIndex = '';
                                    this.style.boxShadow = '';
                                }}
                            }});
                        }});
                    }});
                </script>
            </body>
            </html>
            """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            return html
        except Exception as e:
            print(f"Error generating HTML report: {e}")
            traceback.print_exc()

            # Return a simple error report
            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Cache Performance Report - Error</title>
                <style>
                    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; }}
                    h1, h2 {{ color: #e74c3c; }}
                    .error-box {{ background-color: #f8d7da; border-radius: 8px; padding: 15px; margin-bottom: 20px; color: #721c24; }}
                </style>
            </head>
            <body>
                <h1>Cache Performance Report - Error</h1>
                <div class="error-box">
                    <h2>Error Generating Report</h2>
                    <p>{str(e)}</p>
                    <pre>{traceback.format_exc()}</pre>
                </div>
                <footer>
                    <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                </footer>
            </body>
            </html>
            """

    def _get_progress_color(self, value: float, thresholds: List[float]) -> str:
        """Helper to get a color for progress bars based on thresholds."""
        if value < thresholds[0]:
            return "#ff5a5f"  # Danger/poor (red)
        elif value < thresholds[1]:
            return "#ffb703"  # Warning (yellow)
        else:
            return "#38b000"  # Good (green)

    def _get_rating_class(self, value: float, thresholds: List[float]) -> str:
        """Helper to get a rating class based on thresholds."""
        if value < thresholds[0]:
            return "poor"
        elif value < thresholds[1]:
            return "warning"
        else:
            return "good"
    
    def save_html_report(self, run_data: Dict[str, Any], include_plots: bool = True) -> str:
        """
        Generate and save an HTML report for a run.
        
        Args:
            run_data: The run data dictionary
            include_plots: Whether to include plots in the report
            
        Returns:
            Path to the saved HTML report
        """
        try:
            # Generate HTML content
            html_content = self.generate_html_report(run_data, include_plots)
            
            # Get run ID for directory name
            summary = self.generate_performance_summary(run_data)
            run_id = summary.get("run_id", "Unknown")
            
            # Create output directory for this run
            run_dir = os.path.join(self.output_dir, f"run_{run_id}")
            os.makedirs(run_dir, exist_ok=True)
            
            # Save HTML file
            report_path = os.path.join(run_dir, "report.html")
            with open(report_path, "w") as f:
                f.write(html_content)
            
            return report_path
        except Exception as e:
            print(f"Error saving HTML report: {e}")
            traceback.print_exc()
            
            # Create error report
            error_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Error Generating Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                    h1 {{ color: #e74c3c; }}
                    .error-box {{ background-color: #f8d7da; border-radius: 8px; padding: 15px; color: #721c24; }}
                </style>
            </head>
            <body>
                <h1>Error Generating Report</h1>
                <div class="error-box">
                    <p>{str(e)}</p>
                    <pre>{traceback.format_exc()}</pre>
                </div>
            </body>
            </html>
            """
            
            # Create error directory
            error_dir = os.path.join(self.output_dir, "error_reports")
            os.makedirs(error_dir, exist_ok=True)
            
            # Save error report
            error_path = os.path.join(error_dir, f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
            with open(error_path, "w") as f:
                f.write(error_content)
            
            return error_path

    def plot_temperature_analysis(self, run_data: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
        """
        Create visualizations analyzing temperature performance and its relationship to cache behavior.
        
        Args:
            run_data: The run data dictionary
            save_path: If provided, the plot will be saved to this path
            
        Returns:
            The matplotlib figure
        """
        try:
            df = self._prepare_run_dataframe(run_data)
            
            # Check if we have temperature-related data
            has_temp_data = 'temperature' in df.columns and not df['temperature'].isna().all()
            has_temp_time = 'temperature_time' in df.columns and not df['temperature_time'].isna().all()
            
            if not has_temp_data and not has_temp_time:
                # Create a placeholder if no temperature data is available
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.text(0.5, 0.5, "No temperature data available for analysis.\nEnsure temperature and temperature_time are being logged with requests.", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=14)
                
                if save_path:
                    plt.savefig(save_path, bbox_inches='tight')
                
                return fig
            
            # Create a figure with multiple subplots
            fig = plt.figure(figsize=(16, 12))
            gs = plt.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
            
            # Calculate cumulative cache size (proxy for actual cache size)
            df['cache_size_proxy'] = range(1, len(df) + 1)
            
            # Plot 1: Temperature calculation time vs cache size
            ax1 = fig.add_subplot(gs[0, 0])
            
            # Extract temperature times from the aggregate metrics
            temp_times = run_data.get("temperature_times", [])
            
            if temp_times and len(temp_times) > 0:
                # Filter out zeros and create cache size proxy
                valid_temp_times = [t for t in temp_times if t > 0]
                
                if len(valid_temp_times) > 0:
                    cache_sizes = list(range(1, len(valid_temp_times) + 1))
                    
                    # Create scatter plot
                    # TODO: Figure out if this is the right approach. Cache size seems to be calculated by the number of temp calc measurements which would be incorrect to say the least.
                    scatter = ax1.scatter(cache_sizes, valid_temp_times, alpha=0.6, s=30, c='purple')
                    
                    # Add trend line
                    if len(valid_temp_times) > 1:
                        z = np.polyfit(cache_sizes, valid_temp_times, 1)
                        p = np.poly1d(z)
                        ax1.plot(cache_sizes, p(cache_sizes), 
                                "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.6f}x+{z[1]:.6f}')
                        ax1.legend()
                    
                    ax1.set_xlabel('Request Number (Cache Size Proxy)')
                    ax1.set_ylabel('Temperature Calculation Time (s)')
                    ax1.set_title('Temperature Calculation Time vs Cache Size')
                    ax1.grid(True, alpha=0.3)
                    
                    # Add statistics
                    mean_time = np.mean(valid_temp_times)
                    ax1.text(0.02, 0.98, f'Mean: {mean_time:.5f}s', 
                            transform=ax1.transAxes, fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                            verticalalignment='top')
                else:
                    ax1.text(0.5, 0.5, "No valid temperature time data", 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax1.transAxes, fontsize=12)
            else:
                ax1.text(0.5, 0.5, "No temperature time data available", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax1.transAxes, fontsize=12)
            
            # Plot 2: Temperature values over time
            ax2 = fig.add_subplot(gs[0, 1])
            if has_temp_data:
                temp_df = df[df['temperature'].notna()].copy()
                
                if len(temp_df) > 0:
                    # Color points by cache hit status
                    hit_mask = temp_df['event_type'].str.contains('HIT', na=False)
                    miss_mask = temp_df['event_type'].str.contains('MISS', na=False)
                    direct_mask = temp_df['event_type'].str.contains('DIRECT', na=False)
                    
                    if hit_mask.any():
                        ax2.scatter(temp_df[hit_mask]['request_num'], temp_df[hit_mask]['temperature'], 
                                  alpha=0.7, s=30, color='green', label='Cache Hit')
                    
                    if miss_mask.any():
                        ax2.scatter(temp_df[miss_mask]['request_num'], temp_df[miss_mask]['temperature'], 
                                  alpha=0.7, s=30, color='red', label='Cache Miss')
                    
                    if direct_mask.any():
                        ax2.scatter(temp_df[direct_mask]['request_num'], temp_df[direct_mask]['temperature'], 
                                  alpha=0.7, s=30, color='orange', label='Direct LLM')
                    
                    # Add trend line for temperature
                    if len(temp_df) > 1:
                        z = np.polyfit(temp_df['request_num'], temp_df['temperature'], 1)
                        p = np.poly1d(z)
                        ax2.plot(temp_df['request_num'], p(temp_df['request_num']), 
                                "b--", alpha=0.5, linewidth=1, label='Temperature Trend')
                    
                    ax2.set_xlabel('Request Number')
                    ax2.set_ylabel('Temperature Value')
                    ax2.set_title('Temperature Values Over Time')
                    ax2.grid(True, alpha=0.3)
                    ax2.set_ylim(-0.1, 2.1)  # Temperature range is [0, 2]
                    
                    # Add horizontal lines for reference
                    ax2.axhline(y=0, color='blue', linestyle=':', alpha=0.5, label='Cache Only (T=0)')
                    ax2.axhline(y=2, color='red', linestyle=':', alpha=0.5, label='LLM Only (T=2)')
                    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    
                    # Add statistics
                    mean_temp = temp_df['temperature'].mean()
                    ax2.text(0.02, 0.02, f'Mean: {mean_temp:.3f}', 
                            transform=ax2.transAxes, fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                else:
                    ax2.text(0.5, 0.5, "No valid temperature data", 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax2.transAxes, fontsize=12)
            else:
                ax2.text(0.5, 0.5, "No temperature data available", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax2.transAxes, fontsize=12)
            
            # Plot 3: Temperature distribution and effectiveness
            ax3 = fig.add_subplot(gs[1, 0])
            if has_temp_data:
                temp_data = df['temperature'].dropna()
                
                if len(temp_data) > 0:
                    # Create histogram with custom bins
                    n_bins = min(20, len(temp_data) // 3 + 1)
                    counts, bins, patches = ax3.hist(temp_data, bins=n_bins, alpha=0.7, 
                                                   color='skyblue', edgecolor='black')
                    
                    ax3.set_xlabel('Temperature Value')
                    ax3.set_ylabel('Frequency')
                    ax3.set_title('Temperature Value Distribution')
                    ax3.grid(True, alpha=0.3)
                    
                    # Add statistics
                    mean_temp = temp_data.mean()
                    median_temp = temp_data.median()
                    std_temp = temp_data.std()
                    
                    ax3.axvline(mean_temp, color='red', linestyle='--', 
                               label=f'Mean: {mean_temp:.3f}')
                    ax3.axvline(median_temp, color='orange', linestyle='--', 
                               label=f'Median: {median_temp:.3f}')
                    
                    # Add text box with statistics
                    stats_text = f'Mean: {mean_temp:.3f}\nMedian: {median_temp:.3f}\nStd: {std_temp:.3f}'
                    ax3.text(0.98, 0.98, stats_text, 
                            transform=ax3.transAxes, fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                            verticalalignment='top', horizontalalignment='right')
                    
                    ax3.legend()
                else:
                    ax3.text(0.5, 0.5, "No valid temperature data", 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax3.transAxes, fontsize=12)
            else:
                ax3.text(0.5, 0.5, "No temperature data available", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax3.transAxes, fontsize=12)
            
            # Plot 4: Temperature effectiveness analysis
            ax4 = fig.add_subplot(gs[1, 1])
            if has_temp_data and 'response_time' in df.columns:
                # Filter data with both temperature and response time
                correlation_df = df[df['temperature'].notna() & df['response_time'].notna()].copy()
                
                if len(correlation_df) > 1:
                    # Create scatter plot colored by cache hit status
                    hit_mask = correlation_df['event_type'].str.contains('HIT', na=False)
                    miss_mask = correlation_df['event_type'].str.contains('MISS', na=False)
                    direct_mask = correlation_df['event_type'].str.contains('DIRECT', na=False)
                    
                    if hit_mask.any():
                        ax4.scatter(correlation_df[hit_mask]['temperature'], 
                                  correlation_df[hit_mask]['response_time'],
                                  alpha=0.6, s=40, c='green', label='Cache Hit', marker='o')
                    
                    if miss_mask.any():
                        ax4.scatter(correlation_df[miss_mask]['temperature'], 
                                  correlation_df[miss_mask]['response_time'],
                                  alpha=0.6, s=40, c='red', label='Cache Miss', marker='s')
                    
                    if direct_mask.any():
                        ax4.scatter(correlation_df[direct_mask]['temperature'], 
                                  correlation_df[direct_mask]['response_time'],
                                  alpha=0.6, s=40, c='orange', label='Direct LLM', marker='^')
                    
                    ax4.set_xlabel('Temperature Value')
                    ax4.set_ylabel('Response Time (s)')
                    ax4.set_title('Temperature vs Response Time')
                    ax4.grid(True, alpha=0.3)
                    ax4.legend()
                    
                    # Calculate and display correlation coefficient
                    if len(correlation_df) > 2:
                        corr_coef = np.corrcoef(correlation_df['temperature'], correlation_df['response_time'])[0, 1]
                        ax4.text(0.05, 0.95, f'Correlation: {corr_coef:.3f}', 
                                transform=ax4.transAxes, fontsize=10,
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                    
                    # Add trend line
                    if len(correlation_df) > 2:
                        z = np.polyfit(correlation_df['temperature'], correlation_df['response_time'], 1)
                        p = np.poly1d(z)
                        temp_range = np.linspace(correlation_df['temperature'].min(), 
                                               correlation_df['temperature'].max(), 100)
                        ax4.plot(temp_range, p(temp_range), "purple", alpha=0.5, linewidth=2, linestyle='--')
                    
                    # Calculate cache hit rate by temperature ranges
                    temp_ranges = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]
                    hit_rates = []
                    for low, high in temp_ranges:
                        range_data = correlation_df[(correlation_df['temperature'] >= low) & 
                                                  (correlation_df['temperature'] < high)]
                        if len(range_data) > 0:
                            hit_rate = range_data['event_type'].str.contains('HIT', na=False).mean()
                            hit_rates.append(f'{low}-{high}: {hit_rate:.2%}')
                    
                    if hit_rates:
                        hit_rate_text = 'Hit Rates:\n' + '\n'.join(hit_rates)
                        ax4.text(0.95, 0.05, hit_rate_text, 
                                transform=ax4.transAxes, fontsize=8,
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                                verticalalignment='bottom', horizontalalignment='right')
                else:
                    ax4.text(0.5, 0.5, "Insufficient data for correlation analysis", 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax4.transAxes, fontsize=12)
            else:
                ax4.text(0.5, 0.5, "Temperature or response time data not available", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax4.transAxes, fontsize=12)
            
            # Add overall title
            run_id = run_data.get("summary", {}).get("run_id", "Unknown")
            fig.suptitle(f'Temperature Analysis - Run #{run_id}', fontsize=16, y=0.98)
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
            return fig
            
        except Exception as e:
            print(f"Error creating temperature analysis: {e}")
            traceback.print_exc()
            
            # Create error figure
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, f"Error creating temperature analysis: {e}", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14)
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
            
            return fig

    def plot_embedding_magnitude_distribution(self, run_data: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
        """
        Create histogram and density plots of embedding L2 norms (magnitudes) with statistical analysis.
        
        Args:
            run_data: The run data dictionary
            save_path: If provided, the plot will be saved to this path
            
        Returns:
            The matplotlib figure
        """
        try:
            from scipy import stats
            from scipy.stats import norm, gaussian_kde
            
            df = self._prepare_run_dataframe(run_data)
            
            # Check if we have magnitude data
            has_magnitude_data = 'magnitude' in df.columns and not df['magnitude'].isna().all()
            
            if not has_magnitude_data or len(df) == 0:
                # Create a placeholder if no magnitude data is available
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.text(0.5, 0.5, "No magnitude data available for analysis.\nEnsure magnitude values are being logged with requests.", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=14)
                
                if save_path:
                    plt.savefig(save_path, bbox_inches='tight')
                
                return fig
            
            # Extract magnitude data, filtering out NaN values
            magnitude_df = df[df['magnitude'].notna()].copy()
            
            if len(magnitude_df) == 0:
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.text(0.5, 0.5, "All magnitude values are NaN", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=14)
                
                if save_path:
                    plt.savefig(save_path, bbox_inches='tight')
                
                return fig
            
            magnitudes = magnitude_df['magnitude'].values
            
            # Create figure with multiple subplots
            fig = plt.figure(figsize=(16, 12))
            gs = plt.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
            
            # Plot 1: Basic histogram with fitted distributions
            ax1 = fig.add_subplot(gs[0, 0])
            
            # Create histogram
            n_bins = min(50, len(magnitudes) // 10 + 10)  # Adaptive bin count
            counts, bins, patches = ax1.hist(magnitudes, bins=n_bins, alpha=0.7, 
                                           color='skyblue', edgecolor='black', 
                                           density=True, label='Observed Data')
            
            # Fit and plot normal distribution
            mu, sigma = norm.fit(magnitudes)
            x_norm = np.linspace(magnitudes.min(), magnitudes.max(), 100)
            ax1.plot(x_norm, norm.pdf(x_norm, mu, sigma), 'r-', linewidth=2, 
                    label=f'Normal Fit (μ={mu:.3f}, σ={sigma:.3f})')
            
            # Add KDE (Kernel Density Estimation)
            kde = gaussian_kde(magnitudes)
            ax1.plot(x_norm, kde(x_norm), 'g--', linewidth=2, 
                    label='KDE (Smoothed Density)')
            
            ax1.set_xlabel('Embedding L2 Norm (Magnitude)')
            ax1.set_ylabel('Density')
            ax1.set_title('Distribution of Embedding Magnitudes')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Add statistics text box
            stats_text = f'''Statistics:
    Mean: {np.mean(magnitudes):.4f}
    Median: {np.median(magnitudes):.4f}
    Std: {np.std(magnitudes):.4f}
    Min: {np.min(magnitudes):.4f}
    Max: {np.max(magnitudes):.4f}
    Skewness: {stats.skew(magnitudes):.3f}
    Kurtosis: {stats.kurtosis(magnitudes):.3f}'''
            
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
                    verticalalignment='top')
            
            # Plot 2: Box plots by cache outcome
            ax2 = fig.add_subplot(gs[0, 1])
            
            # Prepare data for box plot based on cache outcomes
            cache_outcomes = []
            magnitude_values = []
            
            # Define cache outcome categories based on your existing logic
            for idx, row in magnitude_df.iterrows():
                magnitude = row['magnitude']
                
                # Use the same logic as your existing cache analysis
                if pd.isna(row.get('used_cache')) or row.get('used_cache') == False:
                    category = 'Direct LLM'
                elif row.get('cache_hit') == True:
                    if row.get('positive_hit', True):  # Default to positive if not specified
                        category = 'Cache Hit (Positive)'
                    else:
                        category = 'Cache Hit (Negative)'
                else:
                    category = 'Cache Miss'
                
                cache_outcomes.append(category)
                magnitude_values.append(magnitude)
            
            # Create DataFrame for box plot
            box_data = pd.DataFrame({
                'Magnitude': magnitude_values,
                'Cache_Outcome': cache_outcomes
            })
            
            if len(box_data) > 0 and len(box_data['Cache_Outcome'].unique()) > 1:
                # Create box plot
                sns.boxplot(data=box_data, x='Cache_Outcome', y='Magnitude', ax=ax2)
                ax2.set_title('Magnitude Distribution by Cache Outcome')
                ax2.set_xlabel('Cache Outcome')
                ax2.set_ylabel('Embedding L2 Norm')
                
                # Rotate x-axis labels for readability
                ax2.tick_params(axis='x', rotation=45)
                
                # Add sample sizes
                for i, category in enumerate(box_data['Cache_Outcome'].unique()):
                    count = len(box_data[box_data['Cache_Outcome'] == category])
                    ax2.text(i, ax2.get_ylim()[1], f'n={count}', 
                            ha='center', va='bottom', fontsize=8)
            else:
                ax2.text(0.5, 0.5, "Insufficient cache outcome variety for comparison", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax2.transAxes, fontsize=12)
            
            # Plot 3: Q-Q plot against normal distribution
            ax3 = fig.add_subplot(gs[1, 0])
            
            stats.probplot(magnitudes, dist="norm", plot=ax3)
            ax3.set_title('Q-Q Plot: Magnitudes vs Normal Distribution')
            ax3.grid(True, alpha=0.3)
            
            # Add correlation coefficient for Q-Q plot
            sorted_magnitudes = np.sort(magnitudes)
            theoretical_quantiles = norm.ppf(np.linspace(0.01, 0.99, len(sorted_magnitudes)))
            correlation = np.corrcoef(sorted_magnitudes, theoretical_quantiles)[0, 1]
            ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                    transform=ax3.transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Plot 4: Cumulative distribution with percentiles
            ax4 = fig.add_subplot(gs[1, 1])
            
            # Empirical CDF
            sorted_mags = np.sort(magnitudes)
            y_values = np.arange(1, len(sorted_mags) + 1) / len(sorted_mags)
            ax4.plot(sorted_mags, y_values, 'b-', linewidth=2, label='Empirical CDF')
            
            # Fitted normal CDF
            ax4.plot(x_norm, norm.cdf(x_norm, mu, sigma), 'r--', linewidth=2, 
                    label='Normal CDF Fit')
            
            # Add percentile markers
            percentiles = [25, 50, 75, 90, 95]
            percentile_values = np.percentile(magnitudes, percentiles)
            
            for p, val in zip(percentiles, percentile_values):
                ax4.axvline(val, color='gray', linestyle=':', alpha=0.7)
                ax4.text(val, p/100, f'{p}th: {val:.3f}', rotation=90, 
                        fontsize=8, ha='right', va='bottom')
            
            ax4.set_xlabel('Embedding L2 Norm (Magnitude)')
            ax4.set_ylabel('Cumulative Probability')
            ax4.set_title('Cumulative Distribution of Magnitudes')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Add overall title
            fig.suptitle(f'Embedding Magnitude Analysis (n={len(magnitudes)} queries)', 
                        fontsize=16, y=0.98)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
            # Print summary statistics
            print(f"\n=== Embedding Magnitude Analysis Summary ===")
            print(f"Total queries analyzed: {len(magnitudes)}")
            print(f"Mean magnitude: {np.mean(magnitudes):.4f}")
            print(f"Median magnitude: {np.median(magnitudes):.4f}")
            print(f"Standard deviation: {np.std(magnitudes):.4f}")
            print(f"Range: [{np.min(magnitudes):.4f}, {np.max(magnitudes):.4f}]")
            print(f"Skewness: {stats.skew(magnitudes):.3f} ({'right-skewed' if stats.skew(magnitudes) > 0 else 'left-skewed' if stats.skew(magnitudes) < 0 else 'symmetric'})")
            print(f"25th percentile: {np.percentile(magnitudes, 25):.4f}")
            print(f"75th percentile: {np.percentile(magnitudes, 75):.4f}")
            print(f"95th percentile: {np.percentile(magnitudes, 95):.4f}")
            
            # Analysis insights for temperature algorithm
            print(f"\n=== Insights for Temperature Algorithm ===")
            if len(box_data) > 0:
                # Calculate hit rates by magnitude quartiles
                q1, q3 = np.percentile(magnitudes, [25, 75])
                low_mag = magnitude_df[magnitude_df['magnitude'] <= q1]
                high_mag = magnitude_df[magnitude_df['magnitude'] >= q3]
                
                if len(low_mag) > 0 and len(high_mag) > 0:
                    low_hit_rate = len(low_mag[low_mag['cache_hit'] == True]) / len(low_mag) if len(low_mag) > 0 else 0
                    high_hit_rate = len(high_mag[high_mag['cache_hit'] == True]) / len(high_mag) if len(high_mag) > 0 else 0
                    
                    print(f"Cache hit rate for low magnitude queries (≤{q1:.3f}): {low_hit_rate:.1%}")
                    print(f"Cache hit rate for high magnitude queries (≥{q3:.3f}): {high_hit_rate:.1%}")
                    
                    if high_hit_rate < low_hit_rate:
                        print("→ High magnitude queries have lower hit rates - consider higher temperature for complex queries")
                    elif high_hit_rate > low_hit_rate:
                        print("→ High magnitude queries have higher hit rates - current algorithm working well")
                    else:
                        print("→ Similar hit rates across magnitude ranges")
            
            return fig
            
        except Exception as e:
            print(f"Error creating embedding magnitude distribution plot: {e}")
            traceback.print_exc()
            
            # Return error plot
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, f"Error creating magnitude distribution plot:\n{str(e)}", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
            
            return fig

    def extract_lsh_debug_data(self, run_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Extract LSH debug information into a DataFrame for analysis.
        """
        lsh_debug_info = run_data.get("lsh_debug_info", [])

        if not lsh_debug_info:
            print("No LSH debug info found in run data")
            return pd.DataFrame()

        # Convert list of debug info dicts to DataFrame
        df = pd.DataFrame(lsh_debug_info)

        # Add cache hit information if available
        if "requests" in run_data and len(run_data["requests"]) == len(lsh_debug_info):
            cache_hits = []
            for req in run_data["requests"]:
                cache_hit = req.get("cache_hit", None)
                # Convert to boolean, handling None and various falsy values
                if cache_hit is None:
                    cache_hits.append(False)
                else:
                    cache_hits.append(bool(cache_hit))

            df["cache_hit"] = cache_hits

        # Ensure numeric columns are properly typed
        numeric_columns = ['temperature', 'bucket_density', 'neighbor_contribution', 
                          'density', 'cache_factor', 'hamming_distance_from_last', 
                          'bucket_age', 'bucket_count', 'bucket_reuse_rate', 
                          'computation_time_ms']

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        return df

    def plot_temperature_dynamics(self, run_data: Dict[str, Any], save_path: str) -> None:
        """Create comprehensive temperature analysis plots"""
        lsh_df = self.extract_lsh_debug_data(run_data)
        
        if lsh_df.empty:
            print("No LSH debug data available for temperature dynamics plot")
            return
        
        fig = plt.figure(figsize=(15, 10))
        
        # Plot 1: Temperature evolution with cache decisions
        ax1 = plt.subplot(3, 1, 1)
        temps = lsh_df['temperature'].values
        times = range(len(temps))
        
        # Color-code by cache decision
        colors = ['green' if hit else 'red' for hit in lsh_df.get('cache_hit', [False] * len(temps))]
        scatter = ax1.scatter(times, temps, c=colors, alpha=0.6, s=30)
        ax1.plot(times, temps, 'k-', alpha=0.3, linewidth=0.5)
        
        ax1.set_ylabel('Temperature')
        ax1.set_title('Temperature Evolution with Cache Decisions (Green=Hit, Red=Miss)')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Temperature vs Cache Hit Rate (rolling window)
        ax2 = plt.subplot(3, 1, 2)
        if 'cache_hit' in lsh_df.columns:
            window_size = min(50, len(lsh_df) // 5)  # Adaptive window size
            rolling_hit_rate = lsh_df['cache_hit'].rolling(window_size).mean()
            ax2.plot(rolling_hit_rate.index, rolling_hit_rate.values, 'b-', 
                    label=f'Hit Rate ({window_size}-request window)')
            ax2_twin = ax2.twinx()
            ax2_twin.plot(times, temps, 'r-', alpha=0.5, label='Temperature')
            
            ax2.set_ylabel('Cache Hit Rate', color='b')
            ax2_twin.set_ylabel('Temperature', color='r')
            ax2.set_xlabel('Request Number')
            ax2.legend(loc='upper left')
            ax2_twin.legend(loc='upper right')
        
        # Plot 3: Temperature distribution
        ax3 = plt.subplot(3, 1, 3)
        ax3.hist(temps, bins=50, edgecolor='black', alpha=0.7)
        ax3.axvline(np.mean(temps), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(temps):.3f}')
        ax3.set_xlabel('Temperature')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Temperature Distribution')
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_lsh_performance_metrics(self, run_data: Dict[str, Any], save_path: str) -> None:
        """Visualize LSH-specific performance metrics"""
        lsh_df = self.extract_lsh_debug_data(run_data)

        if lsh_df.empty:
            print("No LSH debug data available for performance metrics plot")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Computation time analysis
        comp_times = lsh_df['computation_time_ms'].values
        cache_factors = lsh_df['cache_factor'].values

        if 'cache_hit' in lsh_df.columns:
            # Convert cache_hit to boolean array, handling None/NaN values
            cache_hits = lsh_df['cache_hit'].fillna(False)

            # Ensure boolean type
            if cache_hits.dtype != bool:
                # Handle various possible values
                cache_hits = cache_hits.apply(lambda x: bool(x) if x is not None else False)

            hit_mask = cache_hits.values.astype(bool)
            hit_times = comp_times[hit_mask]
            miss_times = comp_times[~hit_mask]

            if len(hit_times) > 0 and len(miss_times) > 0:
                ax1.boxplot([hit_times, miss_times], labels=['Cache Hit', 'Cache Miss'])
            elif len(hit_times) > 0:
                ax1.boxplot([hit_times], labels=['Cache Hit'])
            elif len(miss_times) > 0:
                ax1.boxplot([miss_times], labels=['Cache Miss'])
            else:
                ax1.hist(comp_times, bins=30, alpha=0.7)
        else:
            ax1.hist(comp_times, bins=30, alpha=0.7)

        ax1.set_ylabel('Computation Time (ms)')
        ax1.set_title('LSH Computation Time Distribution')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Bucket reuse efficiency
        ax2.plot(lsh_df['bucket_reuse_rate'].values, 'g-', linewidth=2)
        ax2.fill_between(range(len(lsh_df)), 0, lsh_df['bucket_reuse_rate'].values, alpha=0.3)
        ax2.set_xlabel('Request Number')
        ax2.set_ylabel('Bucket Reuse Rate')
        ax2.set_title('LSH Bucket Reuse Efficiency Over Time')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Cache factor effectiveness
        if 'cache_hit' in lsh_df.columns:
            # Use the cleaned cache_hits from above
            cache_hit_numeric = hit_mask.astype(float)
            ax3.scatter(cache_factors, cache_hit_numeric, alpha=0.5)

            # Add trend line only if we have valid data
            if len(cache_factors) > 1 and np.std(cache_factors) > 0:
                z = np.polyfit(cache_factors, cache_hit_numeric, 1)
                p = np.poly1d(z)
                ax3.plot(sorted(cache_factors), p(sorted(cache_factors)), "r--", alpha=0.8)
        else:
            ax3.hist(cache_factors, bins=30, alpha=0.7)

        ax3.set_xlabel('Cache Factor')
        ax3.set_ylabel('Cache Hit (1) / Miss (0)')
        ax3.set_title('Cache Factor vs Cache Decision')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Hamming distance analysis
        hamming_dists = lsh_df['hamming_distance_from_last'].values
        ax4.hist(hamming_dists, bins=50, edgecolor='black', alpha=0.7)
        if len(hamming_dists) > 0:
            ax4.axvline(np.mean(hamming_dists), color='red', linestyle='--', 
                        label=f'Mean: {np.mean(hamming_dists):.1f}')
        ax4.set_xlabel('Hamming Distance from Previous Query')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Query Similarity Distribution (LSH)')
        ax4.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_lsh_system_behavior(self, run_data: Dict[str, Any], save_path: str) -> None:
        """Analyze LSH system behavior patterns"""
        lsh_df = self.extract_lsh_debug_data(run_data)
        
        if lsh_df.empty:
            print("No LSH debug data available for system behavior plot")
            return
        
        fig = plt.figure(figsize=(15, 12))
        
        # Plot 1: Bucket lifecycle and growth
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(lsh_df['total_unique_buckets'].values, 'b-', label='Unique Buckets')
        ax1.set_xlabel('Request Number')
        ax1.set_ylabel('Total Unique Buckets')
        ax1.set_title('LSH Cache Growth Pattern')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Bucket density vs temperature
        ax2 = plt.subplot(3, 2, 2)
        scatter = ax2.scatter(lsh_df['bucket_density'].values, 
                            lsh_df['temperature'].values, 
                            c=range(len(lsh_df)), 
                            cmap='viridis', alpha=0.6)
        ax2.set_xlabel('Bucket Density')
        ax2.set_ylabel('Temperature')
        ax2.set_title('Temperature Adaptation to Bucket Density')
        plt.colorbar(scatter, ax=ax2, label='Request Number')
        
        # Plot 3: Density components
        ax3 = plt.subplot(3, 2, 3)
        ax3.plot(lsh_df['bucket_density'].values, label='Bucket Density', alpha=0.7)
        ax3.plot(lsh_df['neighbor_contribution'].values, label='Neighbor Contribution', alpha=0.7)
        ax3.plot(lsh_df['density'].values, label='Overall Density', linewidth=2)
        ax3.set_xlabel('Request Number')
        ax3.set_ylabel('Density Value')
        ax3.set_title('LSH Density Components Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Bucket age distribution
        ax4 = plt.subplot(3, 2, 4)
        bucket_ages = lsh_df['bucket_age'].values
        ax4.hist(bucket_ages, bins=30, edgecolor='black', alpha=0.7)
        ax4.set_xlabel('Bucket Age')
        ax4.set_ylabel('Frequency')
        ax4.set_title('LSH Bucket Age Distribution')
        
        # Plot 5: Bucket count distribution
        ax5 = plt.subplot(3, 2, 5)
        bucket_counts = lsh_df['bucket_count'].values
        ax5.hist(bucket_counts, bins=30, edgecolor='black', alpha=0.7, color='orange')
        ax5.set_xlabel('Bucket Count')
        ax5.set_ylabel('Frequency')
        ax5.set_title('LSH Bucket Visit Frequency')
        
        # Plot 6: Correlation heatmap
        ax6 = plt.subplot(3, 2, 6)
        
        # Select numerical columns for correlation
        corr_columns = ['temperature', 'bucket_density', 'neighbor_contribution', 
                       'density', 'cache_factor', 'hamming_distance_from_last', 
                       'bucket_age', 'bucket_count', 'bucket_reuse_rate']
        
        available_columns = [col for col in corr_columns if col in lsh_df.columns]
        corr_matrix = lsh_df[available_columns].corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, linewidths=0.5, ax=ax6, fmt='.2f')
        ax6.set_title('LSH Metrics Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_lsh_adaptive_behavior(self, run_data: Dict[str, Any], save_path: str) -> None:
        """Visualize the adaptive/annealing behavior of the LSH system"""
        lsh_df = self.extract_lsh_debug_data(run_data)
        
        if lsh_df.empty:
            print("No LSH debug data available for adaptive behavior plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        temps = lsh_df['temperature'].values
        
        # Plot 1: Temperature change rate
        temp_changes = np.diff(temps)
        ax1.plot(temp_changes, 'b-', alpha=0.5)
        
        # Add smoothed trend
        window = min(50, len(temp_changes) // 5)
        if window > 1:
            smoothed = pd.Series(temp_changes).rolling(window).mean()
            ax1.plot(smoothed, 'r-', linewidth=2, label=f'Smoothed ({window}-request window)')
        
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Request Number')
        ax1.set_ylabel('Temperature Change')
        ax1.set_title('Temperature Adjustment Rate (Cooling/Heating)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Phase identification
        ax2.plot(temps, 'k-', alpha=0.3)
        
        # Simple phase identification
        exploration_threshold = 1.0
        exploitation_mask = temps <= exploration_threshold
        exploration_mask = temps > exploration_threshold
        
        # Plot phases as background colors
        for i in range(len(temps)):
            if exploration_mask[i]:
                ax2.axvspan(i, i+1, color='red', alpha=0.2)
            else:
                ax2.axvspan(i, i+1, color='blue', alpha=0.1)
        
        ax2.set_xlabel('Request Number')
        ax2.set_ylabel('Temperature')
        ax2.set_title('Exploration (red) vs Exploitation (blue) Phases')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Temperature vs bucket age
        h = ax3.hist2d(lsh_df['bucket_age'].values, temps, bins=[20, 20], cmap='YlOrRd')
        plt.colorbar(h[3], ax=ax3, label='Count')
        ax3.set_xlabel('Bucket Age')
        ax3.set_ylabel('Temperature')
        ax3.set_title('Temperature Adaptation to Bucket Age')
        
        # Plot 4: Bucket reuse rate vs temperature
        ax4.scatter(lsh_df['bucket_reuse_rate'].values, temps, 
                   c=range(len(lsh_df)), cmap='viridis', alpha=0.6)
        ax4.set_xlabel('Bucket Reuse Rate')
        ax4.set_ylabel('Temperature')
        ax4.set_title('Temperature Response to Bucket Reuse')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

# Helper function to generate a report for the latest run
def generate_latest_run_report(log_dir: str = "cache_logs", output_dir: str = "cache_reports") -> str:
    """
    Generate a report for the latest run.
    
    Args:
        log_dir: Directory where log files are stored
        output_dir: Directory where reports will be saved
        
    Returns:
        Path to the saved report
    """
    try:
        analyzer = CachePerformanceAnalyzer(log_dir=log_dir, output_dir=output_dir)
        latest_run = analyzer.load_run_data()
        return analyzer.save_html_report(latest_run)
        #return analyzer.generate_comprehensive_lsh_report(latest_run, output_dir=output_dir)
    except Exception as e:
        print(f"Error generating latest run report: {e}")
        traceback.print_exc()
        
        # Create directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        error_dir = os.path.join(output_dir, "error_reports")
        os.makedirs(error_dir, exist_ok=True)
        
        # Create an error report
        error_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Error Generating Latest Run Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                h1 {{ color: #e74c3c; }}
                .error-box {{ background-color: #f8d7da; border-radius: 8px; padding: 15px; color: #721c24; }}
            </style>
        </head>
        <body>
            <h1>Error Generating Latest Run Report</h1>
            <div class="error-box">
                <p>{str(e)}</p>
                <pre>{traceback.format_exc()}</pre>
            </div>
        </body>
        </html>
        """
        
        # Save error report
        error_path = os.path.join(error_dir, f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        with open(error_path, "w") as f:
            f.write(error_content)
        
        return error_path
    
