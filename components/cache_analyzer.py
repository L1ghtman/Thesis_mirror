import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import traceback

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
                    "start_time": datetime.now().isoformat(),
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
            "positive_hits", "negative_hits", "llm_direct_calls"
        ]
        
        for field in required_fields:
            if field not in run_data:
                if field == "start_time":
                    run_data[field] = datetime.now().isoformat()
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
                "cache_hit", "positive_hit", "response_time", 
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
                "cache_hit", "positive_hit", "response_time", 
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
            
            # Calculate averages safely
            avg_cache_time = sum(cache_response_times) / len(cache_response_times) if cache_response_times else 0
            avg_llm_time = sum(llm_response_times) / len(llm_response_times) if llm_response_times else 0
            
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
            similarity_df = df[df["similarity_score"].notna()]
            
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
            
            # Create a histogram with positive/negative hit coloring
            ax = sns.histplot(
                data=similarity_df,
                x="similarity_score",
                hue="positive_hit",
                palette=["#e74c3c", "#2ecc71"],
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
            
            plt.legend(title="Hit Type")
            
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

            # Calculate performance stats for dashboard
            total_requests = summary["total_requests"]
            cache_hits = summary["cache_hits"]
            positive_hits = summary["positive_hits"]
            time_saved = summary["time_saved"]
            avg_cache_time = summary["avg_cache_time"]
            avg_llm_time = summary["avg_llm_time"]

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
                    </div>
                """.format(
                    plot_paths.get("distribution", "#"),
                    #plot_paths.get("response_times", "#"),
                    plot_paths.get("similarity", "#")
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