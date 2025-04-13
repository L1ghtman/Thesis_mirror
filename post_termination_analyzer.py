from components.cache_analyzer import CachePerformanceAnalyzer

analyzer = CachePerformanceAnalyzer(log_dir="cache_logs")
latest_run = analyzer.load_run_data()
analyzer.save_html_report(latest_run)
