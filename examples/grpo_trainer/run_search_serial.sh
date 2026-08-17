#!/usr/bin/env bash
# Serial training: grpo search on 3B -> 7B -> 1.7B
# Only the first (3B) run does val_before_train
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $(conda info --base)/etc/profile.d/conda.sh

export no_proxy=localhost,127.0.0.1,0.0.0.0,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,33.0.0.0/8
export grpc_proxy=""

# Start retrieval server once
conda activate retriever
bash examples/search/retriever/retrieval_launch.sh > retrieval_server_grpo_serial.log 2>&1 &
RETRIEVER_PID=$!

echo "Waiting for retrieval server to start..."
for i in $(seq 1 400); do
    if curl -s -o /dev/null -w "%{http_code}" -X POST http://0.0.0.0:8000/retrieve -H "Content-Type: application/json" -d '{"query": "test", "topk": 1}' --max-time 5 | grep -q "200"; then
        echo "Retrieval server is ready!"
        break
    fi
    if [ $i -eq 400 ]; then
        echo "ERROR: Retrieval server failed to start after 400 attempts"
        kill $RETRIEVER_PID 2>/dev/null
        exit 1
    fi
    sleep 5
done

# Run 3B (with val_before_train)
echo "========== Starting GRPO 3B =========="
bash "$SCRIPT_DIR/run_search.sh"

# Run 7B
echo "========== Starting GRPO 7B =========="
bash "$SCRIPT_DIR/run_search_7b.sh"

# Run 1.7B
echo "========== Starting GRPO Qwen3-1.7B =========="
bash "$SCRIPT_DIR/run_search_qwen3_1b.sh"

# Cleanup
kill $RETRIEVER_PID 2>/dev/null
echo "All GRPO search training done."
