#!/bin/bash

./tune_system.sh
caffeinate -i bash << 'EOF'

BASEDIR=$(pwd)
OMP_PATH=$(brew --prefix libomp)
RESULT_DIR="$BASEDIR/results/pr"
LAGRAPH_EXE="$BASEDIR/pr_runner"
SPLA_EXE="$BASEDIR/spla/build/pr"
FINAL_CSV="$RESULT_DIR/pr_results.csv"
LOG_FILE="$RESULT_DIR/pr_execution.log"

mkdir -p "$RESULT_DIR"
echo "=== PageRank Benchmark Log: $(date) ===" > "$LOG_FILE"

export DYLD_LIBRARY_PATH=/usr/local/lib:$OMP_PATH/lib:$DYLD_LIBRARY_PATH
export OMP_NUM_THREADS=$(sysctl -n hw.ncpu) # Все ядра

echo ">>> Compiling LAGraph PR runner..."
clang -O3 pr_runner.c -o "$LAGRAPH_EXE" \
  -Xpreprocessor -fopenmp \
  -I"$OMP_PATH/include" \
  -I"$BASEDIR/LAGraph/include" \
  -I/usr/local/include/suitesparse \
  "$BASEDIR/LAGraph/build/src/libLAGraph.a" \
  -L"$OMP_PATH/lib" \
  -L/usr/local/lib \
  -lgraphblas -lomp

echo "Graph,Library,Device,TotalTime_ms,Iters,MS_per_Iter" > "$FINAL_CSV"

GRAPHS=(
    "datasets/soc-LiveJournal1.mtx"
    "datasets/roadNet-CA.mtx"
    "datasets/rgg_n_2_22_s0.mtx"
)

RUNS=5

for G_REL in "${GRAPHS[@]}"; do
    G_PATH="$BASEDIR/$G_REL"
    if [ ! -f "$G_PATH" ]; then
        echo "Skip: $G_REL not found"
        continue
    fi
    G_NAME=$(basename "$G_REL")
    echo -e "\n>>> Testing Graph: $G_NAME"

    echo "  Running LAGraph (CPU)..."
    for ((i=1; i<=RUNS; i++)); do
        OUTPUT=$($LAGRAPH_EXE "$G_PATH" 100)
        echo "$OUTPUT" >> "$LOG_FILE"
        
        TOTAL_TIME=$(echo "$OUTPUT" | grep "TOTAL_TIME_MS:" | awk '{print $2}')
        ITERS=$(echo "$OUTPUT" | grep "ITERS:" | awk '{print $2}')
        MS_ITER=$(echo "$OUTPUT" | grep "MS_PER_ITER:" | awk '{print $2}')
        
        if [ -z "$TOTAL_TIME" ]; then
            echo "    Run $i: FAILED"
        else
            echo "$G_NAME,LAGraph,CPU,$TOTAL_TIME,$ITERS,$MS_ITER" >> "$FINAL_CSV"
            echo "    Run $i: $MS_ITER ms/iter ($ITERS iters)"
        fi
    done
done

echo -e "\n>>> Running SPLA (GPU)..."
for G_REL in "${GRAPHS[@]}"; do
    G_PATH="$BASEDIR/$G_REL"
    [ ! -f "$G_PATH" ] && continue
    G_NAME=$(basename "$G_REL")
    echo "Processing $G_NAME (GPU)..."
    
	TEMP_OUT=$( "$SPLA_EXE" --mtxpath="$G_PATH" --niters=$((RUNS+1)) --run-gpu=true --run-cpu=false 2>&1 )
    echo "$TEMP_OUT" >> "$LOG_FILE"
    
    RAW_LINE=$(echo "$TEMP_OUT" | grep "gpu(ms):")
    
    CLEAN_DATA=$(echo "$RAW_LINE" | sed 's/gpu(ms)://' | sed 's/ //g' | sed 's/,$//')
    
    IFS=',' read -ra TIMES <<< "$CLEAN_DATA"
    
    iter_idx=1
    for t in "${TIMES[@]}"; do
        if [ -n "$t" ]; then
            echo "$G_NAME,SPLA,GPU,$t,$iter_idx,$t" >> "$FINAL_CSV"
            
            if [ "$iter_idx" -eq 1 ]; then
                echo "    Iter $iter_idx: $t ms (Warm-up)"
            else
                echo "    Iter $iter_idx: $t ms"
            fi
            ((iter_idx++))
        fi
    done
done

echo -e "\nBenchmark finished! CSV: $FINAL_CSV"
