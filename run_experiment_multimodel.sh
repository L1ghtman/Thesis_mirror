#!/bin/bash
#SBATCH --partition=gpu_short,gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH --job-name=gptcache
#SBATCH --output=hpc_outputs/gptcache_%j.out
#SBATCH --error=hpc_outputs/gptcache_%j.err

echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"

# Parse command-line arguments
CONFIG_FILE="${1:-configs/base.yaml}"  # Use first argument or default to configs/base.yaml
#RM_CACHE="${2:-true}"

echo "Using config file: $CONFIG_FILE"

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export TOKENIZERS_PARALLELISM=false

NODE_LOCAL_TMP="/scratch/singularity_${SLURM_JOB_ID}_${USER}"
export SINGULARITY_TMPDIR=$NODE_LOCAL_TMP
export SINGULARITY_CACHEDIR=$NODE_LOCAL_TMP/cache

mkdir -p $SINGULARITY_TMPDIR
mkdir -p $SINGULARITY_CACHEDIR

# Debug output to verify paths
echo "=================================================="
echo "Singularity temp directories:"
echo "SINGULARITY_TMPDIR: $SINGULARITY_TMPDIR"
echo "SINGULARITY_CACHEDIR: $SINGULARITY_CACHEDIR"
echo "=================================================="

cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"

#if [ "$RM_CACHE" = "true" ]; then
#    rm -rf persistent_cache
#fi

mkdir -p cache_logs cache_reports persistent_cache

# Validate files exist
if [ ! -f "llm_cache_system_v7.sif" ]; then
    echo "ERROR: Singularity image not found!"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file '$CONFIG_FILE' not found!"
    echo "Usage: sbatch $0 [config_file.yaml]"
    echo "Example: sbatch $0 configs/experiment1.yaml"
    exit 1
fi

if [ ! -f "benchmark.py" ]; then
    echo "ERROR: benchmark.py not found!"
    exit 1
fi

echo "Loading Singularity module..."
module load singularity

# Extract model name from config file
#MODEL_NAME=$(python3 -c "
#import yaml
#import sys
#with open('$CONFIG_FILE', 'r') as f:
#    config = yaml.safe_load(f)
#    print(config.get('model', 'Llama-3.2-3B-Instruct-F16.gguf'))
#" 2>/dev/null || echo "Llama-3.2-3B-Instruct-F16.gguf")

echo "Attempting to extract model from config..."
#MODEL_NAME=$(python3 -c "
#import yaml
#import sys
#print('Reading config file: $CONFIG_FILE', file=sys.stderr)
#with open('$CONFIG_FILE', 'r') as f:
#    config = yaml.safe_load(f)
#    print('Config contents:', config, file=sys.stderr)
#    print(config.get('sys', {}).get('model', 'Llama-3.2-3B-Instruct-F16.gguf'))
#" 2>&1)
#echo "Extracted MODEL_NAME: $MODEL_NAME"
#
#echo "Using model: $MODEL_NAME"

# Extract model name from config file using shell tools (no Python dependencies)
MODEL_NAME=$(grep -A 20 '^sys:' "$CONFIG_FILE" | grep 'model:' | head -1 | awk -F'"' '{print $2}')

# Fallback to default if extraction failed
if [ -z "$MODEL_NAME" ]; then
    MODEL_NAME="Llama-3.2-3B-Instruct-F16.gguf"
    echo "Warning: Could not extract model from config, using default: $MODEL_NAME"
fi

echo "Using model: $MODEL_NAME"

echo "Starting LLM server directly with singularity exec..."

SANDBOX_DIR="/scratch/llm_sandbox_${SLURM_JOB_ID}_${USER}"
if [ ! -d "$SANDBOX_DIR" ]; then
    echo "Creating sandbox from SIF on local storage..."
    singularity build --sandbox "$SANDBOX_DIR" llm_cache_system_v7.sif
    echo "Sandbox created at: $SANDBOX_DIR"
else
    echo "Using existing sandbox at: $SANDBOX_DIR"
fi

# Directly execute llama-server - don't use the runscript at all
singularity exec --nv \
    --writable-tmpfs \
    "$SANDBOX_DIR" \
    llama-server \
    -m /opt/llama.cpp/models/$MODEL_NAME \
    --host 127.0.0.1 \
    --port 8080 \
    --ctx-size 4096 \
    --n-gpu-layers 99 \
    --parallel 1 &

SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"

# Cleanup function
cleanup() {
    echo "Cleaning up..."
    if [ ! -z "$SERVER_PID" ] && kill -0 $SERVER_PID 2>/dev/null; then
        echo "Stopping server (PID: $SERVER_PID)..."
        kill $SERVER_PID
        sleep 3
        kill -9 $SERVER_PID 2>/dev/null
    fi

    echo "Removing sandbox..."
    rm -rf "$SANDBOX_DIR" 2>/dev/null
}
trap cleanup EXIT INT TERM

# Wait for server to be ready
echo "Waiting for LLM server to load model (approximately 3-4 minutes)..."

# Wait 4 minutes for model to load
WAIT_TIME=240
for i in $(seq 1 $WAIT_TIME); do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server process died at $i seconds!"
        exit 1
    fi
    
    # Show progress every 30 seconds
    if [ $((i % 30)) -eq 0 ]; then
        echo "Still waiting... ($i/$WAIT_TIME seconds elapsed)"
    fi
    
    sleep 1
done

echo "Wait complete. Server should be ready now."

# Simple connectivity test using Python
echo "Testing server connectivity..."
python3 << 'PYEOF'
import urllib.request
import sys

try:
    response = urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=10)
    print(f"✓ Server is responding! Status: {response.getcode()}")
    sys.exit(0)
except Exception as e:
    print(f"⚠ Could not connect to server: {e}")
    print("Continuing anyway - server may still be starting...")
    sys.exit(0)  # Don't fail the job, just warn
PYEOF

echo "Starting GPTCache experiment..."

# Run the benchmark with the specified config file
singularity exec --nv \
    --writable-tmpfs \
    "$SANDBOX_DIR" \
    python3 benchmark.py --config "$CONFIG_FILE"

PYTHON_EXIT_CODE=$?

if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "GPTCache experiment completed successfully at: $(date)"
else
    echo "GPTCache experiment failed with exit code $PYTHON_EXIT_CODE at: $(date)"
fi

# Copy results
#TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
#RESULTS_DIR="results_${TIMESTAMP}"
#mkdir -p $RESULTS_DIR
#cp -r cache_logs cache_reports $RESULTS_DIR/ 2>/dev/null || echo "No results to copy"
#echo "Results saved to: $RESULTS_DIR"

echo "Job finished at: $(date)"
