#!/bin/bash

# Distributed Training Runner for Wild-West Server
# Coordinates multiple processes across GPUs for distributed training

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOCK_DIR="/tmp/gpu_locks"

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] <config_name>"
    echo ""
    echo "Options:"
    echo "  -g, --gpus <gpu_ids>     Comma-separated GPU IDs (default: 1,2)"
    echo "  -n, --nodes <num>         Number of nodes (default: 1)"
    echo "  -p, --phase <phase>       Phase to run (default: run)"
    echo "  -b, --batch-size <size>   Batch size per GPU"
    echo "  -e, --epochs <num>        Number of epochs"
    echo "  -m, --master-port <port>  Master port for distributed training (default: 12355)"
    echo "  -a, --master-addr <addr>  Master address (default: localhost)"
    echo "  -l, --lock-gpus           Lock GPUs before running"
    echo "  -c, --check-gpus          Check GPU availability before running"
    echo "  -v, --verbose             Verbose output"
    echo "  -h, --help                Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 -g 1,2 experiment_1_remove_expletives"
    echo "  $0 -g 0,1,2,3 -n 2 experiment_2_baseline"
    echo "  $0 -g 1,2 -b 32 -e 10 experiment_3_remove_articles"
}

# Parse command line arguments
GPUS="1,2"
NODES=1
PHASE="run"
BATCH_SIZE=""
NUM_EPOCHS=""
MASTER_PORT="12355"
MASTER_ADDR="localhost"
LOCK_GPUS=false
CHECK_GPUS=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--gpus)
            GPUS="$2"
            shift 2
            ;;
        -n|--nodes)
            NODES="$2"
            shift 2
            ;;
        -p|--phase)
            PHASE="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -e|--epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        -m|--master-port)
            MASTER_PORT="$2"
            shift 2
            ;;
        -a|--master-addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        -l|--lock-gpus)
            LOCK_GPUS=true
            shift
            ;;
        -c|--check-gpus)
            CHECK_GPUS=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            CONFIG_NAME="$1"
            shift
            ;;
    esac
done

# Check if config name is provided
if [ -z "$CONFIG_NAME" ]; then
    echo -e "${RED}ERROR: Config name is required${NC}"
    show_usage
    exit 1
fi

# Function to log messages
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        "INFO")
            echo -e "${BLUE}[$timestamp] INFO: $message${NC}"
            ;;
        "WARN")
            echo -e "${YELLOW}[$timestamp] WARN: $message${NC}"
            ;;
        "ERROR")
            echo -e "${RED}[$timestamp] ERROR: $message${NC}"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[$timestamp] SUCCESS: $message${NC}"
            ;;
    esac
}

# Function to check GPU availability
check_gpu_availability() {
    log "INFO" "Checking GPU availability..."
    
    # Convert comma-separated GPUs to array
    IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
    
    for gpu_id in "${GPU_ARRAY[@]}"; do
        gpu_id=$(echo "$gpu_id" | tr -d ' ')
        
        # Get GPU memory info
        local memory_info=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | grep "^$gpu_id,")
        
        if [ -z "$memory_info" ]; then
            log "ERROR" "GPU $gpu_id not found"
            return 1
        fi
        
        local memory_used=$(echo "$memory_info" | cut -d',' -f2 | tr -d ' ')
        local memory_total=$(echo "$memory_info" | cut -d',' -f3 | tr -d ' ')
        local available_gb=$((memory_total - memory_used))
        
        log "INFO" "GPU $gpu_id: ${available_gb}GB available (${memory_used}GB used / ${memory_total}GB total)"
        
        if [ "$available_gb" -lt 10240 ]; then  # Less than 10GB available
            log "WARN" "GPU $gpu_id has limited memory (${available_gb}GB available)"
        fi
    done
    
    return 0
}

# Function to lock GPUs
lock_gpus() {
    log "INFO" "Locking GPUs: $GPUS"
    
    IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
    
    for gpu_id in "${GPU_ARRAY[@]}"; do
        gpu_id=$(echo "$gpu_id" | tr -d ' ')
        local lock_file="$LOCK_DIR/gpu_${gpu_id}.lock"
        
        if [ -f "$lock_file" ]; then
            log "ERROR" "GPU $gpu_id is already locked by:"
            cat "$lock_file"
            return 1
        fi
        
        echo "User: $(whoami)" > "$lock_file"
        echo "Time: $(date)" >> "$lock_file"
        echo "PID: $$" >> "$lock_file"
        echo "Config: $CONFIG_NAME" >> "$lock_file"
        echo "Phase: $PHASE (distributed)" >> "$lock_file"
        
        log "SUCCESS" "Locked GPU $gpu_id"
    done
}

# Function to unlock GPUs
unlock_gpus() {
    log "INFO" "Unlocking GPUs: $GPUS"
    
    IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
    
    for gpu_id in "${GPU_ARRAY[@]}"; do
        gpu_id=$(echo "$gpu_id" | tr -d ' ')
        local lock_file="$LOCK_DIR/gpu_${gpu_id}.lock"
        
        if [ -f "$lock_file" ]; then
            rm "$lock_file"
            log "SUCCESS" "Unlocked GPU $gpu_id"
        else
            log "WARN" "No lock found on GPU $gpu_id"
        fi
    done
}

# Function to run distributed training
run_distributed_training() {
    local config_file="$1"
    local world_size="$2"
    
    log "INFO" "Starting distributed training with $world_size processes"
    log "INFO" "GPUs: $GPUS"
    log "INFO" "Master: $MASTER_ADDR:$MASTER_PORT"
    
    # Convert GPUs to array
    IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
    
    # Start processes for each GPU
    local pids=()
    local rank=0
    
    for gpu_id in "${GPU_ARRAY[@]}"; do
        gpu_id=$(echo "$gpu_id" | tr -d ' ')
        
        # Set environment variables for this process
        export CUDA_VISIBLE_DEVICES="$gpu_id"
        export MASTER_ADDR="$MASTER_ADDR"
        export MASTER_PORT="$MASTER_PORT"
        export WORLD_SIZE="$world_size"
        export RANK="$rank"
        export NCCL_DEBUG=INFO
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        
        log "INFO" "Starting process $rank on GPU $gpu_id"
        
        # Build command
        local command="python -m model_foundry.cli $PHASE $config_file --distributed --world-size $world_size --rank $rank"
        
        if [ -n "$BATCH_SIZE" ]; then
            command="$command --batch-size $BATCH_SIZE"
        fi
        
        if [ -n "$NUM_EPOCHS" ]; then
            command="$command --epochs $NUM_EPOCHS"
        fi
        
        # Run in background
        if [ "$VERBOSE" = true ]; then
            singularity exec --nv \
                --bind "${PROJECT_DIR}":/workspace \
                "$PROJECT_DIR/singularity/training.sif" \
                bash -c "cd /workspace && $command" &
        else
            singularity exec --nv \
                --bind "${PROJECT_DIR}":/workspace \
                "$PROJECT_DIR/singularity/training.sif" \
                bash -c "cd /workspace && $command" > "$PROJECT_DIR/logs/distributed_${CONFIG_NAME}_rank${rank}.log" 2>&1 &
        fi
        
        pids+=($!)
        rank=$((rank + 1))
    done
    
    # Wait for all processes to complete
    log "INFO" "Waiting for all processes to complete..."
    for pid in "${pids[@]}"; do
        wait "$pid"
        local exit_code=$?
        if [ $exit_code -ne 0 ]; then
            log "ERROR" "Process $pid exited with code $exit_code"
        else
            log "SUCCESS" "Process $pid completed successfully"
        fi
    done
    
    log "SUCCESS" "All distributed training processes completed"
}

# Function to run single-node distributed training
run_single_node_distributed() {
    local config_file="$1"
    local world_size="$2"
    
    log "INFO" "Running single-node distributed training"
    log "INFO" "World size: $world_size"
    log "INFO" "GPUs: $GPUS"
    
    # Convert GPUs to array
    IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
    
    if [ ${#GPU_ARRAY[@]} -ne $world_size ]; then
        log "ERROR" "Number of GPUs (${#GPU_ARRAY[@]}) doesn't match world size ($world_size)"
        return 1
    fi
    
    # Start processes
    local pids=()
    local rank=0
    
    for gpu_id in "${GPU_ARRAY[@]}"; do
        gpu_id=$(echo "$gpu_id" | tr -d ' ')
        
        # Set environment variables
        export CUDA_VISIBLE_DEVICES="$gpu_id"
        export MASTER_ADDR="$MASTER_ADDR"
        export MASTER_PORT="$MASTER_PORT"
        export WORLD_SIZE="$world_size"
        export RANK="$rank"
        export NCCL_DEBUG=INFO
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        
        log "INFO" "Starting process $rank on GPU $gpu_id"
        
        # Build command
        local command="python -m model_foundry.cli $PHASE $config_file --distributed --world-size $world_size --rank $rank"
        
        if [ -n "$BATCH_SIZE" ]; then
            command="$command --batch-size $BATCH_SIZE"
        fi
        
        if [ -n "$NUM_EPOCHS" ]; then
            command="$command --epochs $NUM_EPOCHS"
        fi
        
        # Run in background
        if [ "$VERBOSE" = true ]; then
            singularity exec --nv \
                --bind "${PROJECT_DIR}":/workspace \
                "$PROJECT_DIR/singularity/training.sif" \
                bash -c "cd /workspace && $command" &
        else
            singularity exec --nv \
                --bind "${PROJECT_DIR}":/workspace \
                "$PROJECT_DIR/singularity/training.sif" \
                bash -c "cd /workspace && $command" > "$PROJECT_DIR/logs/distributed_${CONFIG_NAME}_rank${rank}.log" 2>&1 &
        fi
        
        pids+=($!)
        rank=$((rank + 1))
    done
    
    # Wait for all processes
    log "INFO" "Waiting for all processes to complete..."
    for pid in "${pids[@]}"; do
        wait "$pid"
        local exit_code=$?
        if [ $exit_code -ne 0 ]; then
            log "ERROR" "Process $pid exited with code $exit_code"
        else
            log "SUCCESS" "Process $pid completed successfully"
        fi
    done
}

# Main execution
main() {
    log "INFO" "Starting distributed experiment: $CONFIG_NAME"
    log "INFO" "Phase: $PHASE"
    log "INFO" "GPUs: $GPUS"
    log "INFO" "Nodes: $NODES"
    
    # Check GPU availability if requested
    if [ "$CHECK_GPUS" = true ]; then
        check_gpu_availability || exit 1
    fi
    
    # Lock GPUs if requested
    if [ "$LOCK_GPUS" = true ]; then
        lock_gpus || exit 1
        # Set up trap to unlock GPUs on exit
        trap 'unlock_gpus' EXIT
    fi
    
    # Define paths
    HOST_CONFIG_FILE="$PROJECT_DIR/configs/${CONFIG_NAME}.yaml"
    CONTAINER_CONFIG_FILE="configs/${CONFIG_NAME}.yaml"
    
    # Check for required files
    if [ ! -f "$HOST_CONFIG_FILE" ]; then
        log "ERROR" "Config file not found at $HOST_CONFIG_FILE"
        exit 1
    fi
    
    # Create logs directory
    mkdir -p "$PROJECT_DIR/logs"
    
    # Calculate world size
    IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
    local num_gpus=${#GPU_ARRAY[@]}
    local world_size=$((num_gpus * NODES))
    
    log "INFO" "Total GPUs: $num_gpus"
    log "INFO" "World size: $world_size"
    
    # Run distributed training
    if [ "$NODES" -eq 1 ]; then
        run_single_node_distributed "$CONTAINER_CONFIG_FILE" "$world_size"
    else
        log "ERROR" "Multi-node distributed training not yet implemented"
        log "ERROR" "Please use single node (NODES=1) for now"
        exit 1
    fi
    
    log "SUCCESS" "Distributed experiment completed: $CONFIG_NAME"
    log "SUCCESS" "Phase: $PHASE"
    log "SUCCESS" "GPUs used: $GPUS"
    log "SUCCESS" "World size: $world_size"
}

# Run main function
main "$@" 