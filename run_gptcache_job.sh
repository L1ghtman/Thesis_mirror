#!/bin/bash
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=gptcache
#SBATCH --output=gptcache_%j.out
#SBATCH --error=gptcache_%j.err

# Print job information
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"

# Set up environment
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export TOKENIZERS_PARALLELISM=false

# Change to the directory containing your code
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"

# Create necessary directories
mkdir -p cache_logs cache_reports persistent_cache

# Check if files exist
if [ ! -f "llm_cache_system_v5.sif" ]; then
    echo "ERROR: Singularity image llm_cache_system_v5.sif not found!"
    exit 1
fi

if [ ! -f "configs/base.yaml" ]; then
    echo "ERROR: Config file configs/base.yaml not found!"
    exit 1
fi

if [ ! -f "benchmark.py" ]; then
    echo "ERROR: Python script benchmark.py not found!"
    exit 1
fi

if [ ! -f "gptcache_advanced.py" ]; then
    echo "ERROR: Python script gptcache_advanced.py not found!"
    exit 1
fi

echo "Starting LLM server in background..."

# Start the singularity container with LLM server in background
# The container automatically starts the llama.cpp server
singularity run --nv llm_cache_system_v5.sif &
CONTAINER_PID=$!
echo "Container started with PID: $CONTAINER_PID"

# Function to cleanup on exit
cleanup() {
    echo "Cleaning up..."
    if [ ! -z "$CONTAINER_PID" ] && kill -0 $CONTAINER_PID 2>/dev/null; then
        echo "Stopping container (PID: $CONTAINER_PID)..."
        kill $CONTAINER_PID
        # Wait a bit, then force kill if necessary
        sleep 5
        if kill -0 $CONTAINER_PID 2>/dev/null; then
            echo "Force killing container..."
            kill -9 $CONTAINER_PID
        fi
    fi
}

# Set up cleanup trap
trap cleanup EXIT INT TERM

# Wait for the LLM server to be ready
echo "Waiting for LLM server to be ready..."
MAX_WAIT=120  # Wait up to 2 minutes
WAIT_COUNT=0

while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    # Check if server is responding
    if curl -s -f http://localhost:8080/health > /dev/null 2>&1; then
        echo "LLM server is ready!"
        break
    fi
    
    # Check if container is still running
    if ! kill -0 $CONTAINER_PID 2>/dev/null; then
        echo "ERROR: Container process died!"
        exit 1
    fi
    
    echo "Waiting for server... ($((WAIT_COUNT + 1))/$MAX_WAIT)"
    sleep 2
    WAIT_COUNT=$((WAIT_COUNT + 1))
done

if [ $WAIT_COUNT -eq $MAX_WAIT ]; then
    echo "ERROR: LLM server failed to start within $MAX_WAIT attempts"
    exit 1
fi

# Test server connection
echo "Testing server connection..."
if ! curl -s -f http://localhost:8080/health > /dev/null; then
    echo "ERROR: Cannot connect to LLM server"
    exit 1
fi

echo "LLM server is ready. Starting GPTCache experiment..."

# Run the Python script using singularity exec on the same container
# We use exec instead of run to avoid starting another server instance
singularity exec --nv llm_cache_system_v5.sif python3 gptcache_advanced.py

# Check exit status
PYTHON_EXIT_CODE=$?

if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "GPTCache experiment completed successfully at: $(date)"
else
    echo "GPTCache experiment failed with exit code $PYTHON_EXIT_CODE at: $(date)"
fi

# Optional: Copy results to a timestamped directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="results_${TIMESTAMP}"
mkdir -p $RESULTS_DIR
cp -r cache_logs cache_reports $RESULTS_DIR/ 2>/dev/null || echo "No results to copy"
echo "Results saved to: $RESULTS_DIR"

echo "Job finished at: $(date)"
