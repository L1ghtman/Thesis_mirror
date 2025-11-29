#!/usr/bin/env python3
"""Wait for llama.cpp server to be ready and send warmup request."""

import urllib.request
import urllib.error
import json
import time
import sys

def main():
    max_wait = 300
    start = time.time()
    
    # Wait for health endpoint
    while time.time() - start < max_wait:
        try:
            urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=5)
            print(f"✓ Server ready after {int(time.time() - start)}s")
            break
        except urllib.error.URLError:
            time.sleep(1)
    else:
        print("ERROR: Server timeout")
        sys.exit(1)
    
    # Warmup request
    print("Sending warmup request...")
    try:
        data = json.dumps({"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 5}).encode()
        req = urllib.request.Request('http://127.0.0.1:8080/v1/chat/completions', data=data, headers={'Content-Type': 'application/json'})
        urllib.request.urlopen(req, timeout=120)
        print("✓ Warmup complete")
    except Exception as e:
        print(f"⚠ Warmup failed: {e}")

if __name__ == "__main__":
    main()