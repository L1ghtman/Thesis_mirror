import json
import os
from dataclasses import dataclass

@dataclass
class RequestResponseItem:
    query: str
    response: str

def read_json_data(filepath):
    """
    Reads data from a JSON file.
    """

    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
        
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {filepath}: {e}")
        return None

    except Exception as e:
        print(f"An unexpected error occurred while reading {filepath}: {e}")
        return None

def ingest_data(filepath):
    """
    Reads run data and returns list of dataclass items with request and response as values
    """
    raw_data = read_json_data(filepath)
    if not raw_data:
        return []

    # get request
    requests = raw_data.get("requests", "")

    items = []
    for entry in requests:
        item = RequestResponseItem(
            query=entry.get("query", ""),
            response=entry.get("response", "")
        )
        items.append(item)
    
    return items
