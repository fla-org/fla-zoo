#!/bin/bash
# filepath: /home/bo/fla-zoo/run_benchmark.sh

# Set default values
OUTPUT_DIR="benchmark_results"
DEVICE="cuda"
DTYPE="float16"
BATCH_SIZE=2
HIDDEN_SIZE=1024
NUM_LAYERS=6
HEADS=64
WARMUP=10
RUNS=50

# Ensure output directory exists
mkdir -p $OUTPUT_DIR

# Colored output function
print_header() {
    echo -e "\033[1;34m==== $1 ====\033[0m"
}

# Basic usage display
usage() {
    echo "Usage: $0 [option]"
    echo "Options:"
    echo "  --quick        Run quick test (fewer models and sequence lengths)"
    echo "  --all-models   Test all available models"
    echo "  --seq-scaling  Test model scaling with different sequence lengths"
    echo "  --compare      Compare performance of several main models"
    echo "  --custom       Run test with custom parameters"
    echo "  --help         Display help information"
    exit 1
}

# Quick test - use fewer models and sequence lengths
run_quick_test() {
    print_header "Running Quick Test"
    
    python benchmark.py \
        --models nsa transformer deltanet gla hgrn retnet \
        --seq-lengths 64 256 1024 2048 4096 8192 16384 \
        --batch-size 1 \
        --hidden-size 1024 \
        --num-layers 6 \
        --heads 64 \
        --warmup 5 \
        --runs 10 \
        --output-dir "${OUTPUT_DIR}/quick_test"
}

# Test all models
run_all_models() {
    print_header "Testing All Available Models"
    
    python benchmark.py \
        --models transformer deltanet gla hgrn hgrn2 lightnet retnet nsa \
        --seq-lengths 128 512 1024 \
        --batch-size $BATCH_SIZE \
        --hidden-size $HIDDEN_SIZE \
        --num-layers $NUM_LAYERS \
        --heads $HEADS \
        --warmup $WARMUP \
        --runs $RUNS \
        --output-dir "${OUTPUT_DIR}/all_models"
}

# Test sequence length scaling performance
run_seq_scaling() {
    print_header "Testing Sequence Length Scaling Performance"
    
    python benchmark.py \
        --models transformer deltanet hgrn retnet \
        --seq-lengths 16 32 64 128 256 512 1024 2048 4096 8192 16384 \
        --batch-size $BATCH_SIZE \
        --hidden-size $HIDDEN_SIZE \
        --num-layers $NUM_LAYERS \
        --heads $HEADS \
        --warmup $WARMUP \
        --runs $RUNS \
        --output-dir "${OUTPUT_DIR}/seq_scaling"
}

# Compare several main models
run_compare() {
    print_header "Comparing Main Model Performance"
    
    python benchmark.py \
        --models transformer deltanet hgrn retnet \
        --seq-lengths 64 128 256 512 1024 2048 \
        --batch-size $BATCH_SIZE \
        --hidden-size $HIDDEN_SIZE \
        --num-layers $NUM_LAYERS \
        --heads $HEADS \
        --warmup $WARMUP \
        --runs $RUNS \
        --output-dir "${OUTPUT_DIR}/model_compare"
}

# Run with custom parameters
run_custom() {
    print_header "Running Custom Test"
    
    # Get parameters from command line
    echo "Enter models to test (space separated, e.g. 'transformer deltanet'):"
    read -a MODELS
    
    echo "Enter sequence lengths to test (space separated, e.g. '64 128 256'):"
    read -a SEQ_LENGTHS
    
    echo "Enter hidden size (default $HIDDEN_SIZE):"
    read INPUT
    CUSTOM_HIDDEN_SIZE=${INPUT:-$HIDDEN_SIZE}
    
    echo "Enter number of layers (default $NUM_LAYERS):"
    read INPUT
    CUSTOM_NUM_LAYERS=${INPUT:-$NUM_LAYERS}
    
    # Build model and sequence length parameters
    MODEL_ARGS=$(printf -- "--models %s " "${MODELS[@]}")
    SEQ_ARGS=$(printf -- "--seq-lengths %s " "${SEQ_LENGTHS[@]}")
    
    python benchmark.py \
        $MODEL_ARGS \
        $SEQ_ARGS \
        --batch-size $BATCH_SIZE \
        --hidden-size $CUSTOM_HIDDEN_SIZE \
        --num-layers $CUSTOM_NUM_LAYERS \
        --heads $HEADS \
        --warmup $WARMUP \
        --runs $RUNS \
        --output-dir "${OUTPUT_DIR}/custom_test"
}

# Process command line arguments
if [ $# -eq 0 ]; then
    usage
    exit 1
fi

case "$1" in
    --quick)
        run_quick_test
        ;;
    --all-models)
        run_all_models
        ;;
    --seq-scaling)
        run_seq_scaling
        ;;
    --compare)
        run_compare
        ;;
    --custom)
        run_custom
        ;;
    --help)
        usage
        ;;
    *)
        echo "Unknown option: $1"
        usage
        ;;
esac

print_header "Test Completed"
echo "Results saved in $OUTPUT_DIR directory"