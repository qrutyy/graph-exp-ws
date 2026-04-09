#!/bin/bash

./tune_system.sh
caffeinate -i bash << 'EOF'

BASEDIR=$(pwd)
LAGRAPH_EXE="$BASEDIR/tc_runner"
SPLA_EXE="$BASEDIR/spla/build/tc"
OMP_PATH=$(brew --prefix libomp)

RESULT_DIR="$BASEDIR/results/tc"
FINAL_CSV="$RESULT_DIR/tc_results.csv"
LOG_FILE="$RESULT_DIR/tc_execution.log"

mkdir -p "$RESULT_DIR"
echo "=== TC Full Benchmark (Methods & Presort): $(date) ===" > "$LOG_FILE"

export OMP_NUM_THREADS=$(sysctl -n hw.ncpu)
export DYLD_LIBRARY_PATH=/usr/local/lib:"$OMP_PATH"/lib:$DYLD_LIBRARY_PATH

echo ">>> Compiling LAGraph TC runner..."
COMMON_FLAGS=(-O3 tc_runner.c -o "$LAGRAPH_EXE" -I"$BASEDIR/LAGraph/include" -I/usr/local/include/suitesparse "$BASEDIR/LAGraph/build/src/libLAGraph.a" -L/usr/local/lib -lgraphblas)
if [ -n "$OMP_PATH" ]; then
  clang "${COMMON_FLAGS[@]}" -Xpreprocessor -fopenmp -I"$OMP_PATH/include" -L"$OMP_PATH/lib" -lomp
else
  clang "${COMMON_FLAGS[@]}"
fi

echo "Graph,Library,Method,Sort,Triangles,Time_ms,IterType" > "$FINAL_CSV"

GRAPHS=(
    "datasets/roadNet-CA.mtx"
    "datasets/soc-LiveJournal1.mtx"
    "datasets/rgg_n_2_22_s0.mtx"
)

METHODS=(1 2 3 4 5 6)
PRESORTS=(0 1 2)
RUNS=10

for G_REL in "${GRAPHS[@]}"; do
    G_PATH="$BASEDIR/$G_REL"
    [ ! -f "$G_PATH" ] && continue
    G_NAME=$(basename "$G_REL")
    echo "--- Graph: $G_NAME ---" | tee -a "$LOG_FILE"

    for M in "${METHODS[@]}"; do
        for S in "${PRESORTS[@]}"; do
            # Burkhardt (1) и Cohen (2) игнорируют сортировку внутри LAGraph, 
            # но мы запустим их один раз (S=0), чтобы не делать лишнюю работу.
            if [[ ($M -eq 1 || $M -eq 2) && $S -eq 1 ]]; then continue; fi

            echo ">>> Running LAGraph Method $M Sort $S..."
            for ((i=1; i<=RUNS; i++)); do
                OUT=$($LAGRAPH_EXE "$G_PATH" "$M" "$S" 2>&1)
                
                MNAME=$(echo "$OUT" | grep "RESULT_SUCCESS" | awk -F'|' '{print $2}' | cut -d':' -f2)
                SNAME=$(echo "$OUT" | grep "RESULT_SUCCESS" | awk -F'|' '{print $3}' | cut -d':' -f2)
                TRI=$(echo "$OUT" | grep "TRIANGLES:" | awk '{print $2}')
                TIME_SEC=$(echo "$OUT" | grep "TIME:" | awk '{print $2}')
                
                if [ -n "$TIME_SEC" ]; then
                    TIME_MS=$(awk -v s="$TIME_SEC" 'BEGIN {print s * 1000}')
                    echo "$G_NAME,LAGraph,$MNAME,$SNAME,$TRI,$TIME_MS,Measured" >> "$FINAL_CSV"
                    echo "   [$MNAME-$SNAME] Run $i: $TIME_MS ms"
                fi
            done
        done
    done

    echo ">>> Running SPLA (GPU)..."
    TEMP_OUT=$($SPLA_EXE --mtxpath="$G_PATH" --niters=$((RUNS+1)) --run-gpu=true --run-cpu=false 2>&1)
    
    SPLA_TRI=$(echo "$TEMP_OUT" | grep "ntrins" | head -n 1 | awk '{print $4}')
    RAW_LINE=$(echo "$TEMP_OUT" | grep "gpu(ms):")
    CLEAN_DATA=$(echo "$RAW_LINE" | sed 's/gpu(ms)://' | sed 's/ //g' | sed 's/,$//')
    IFS=',' read -ra TIMES <<< "$CLEAN_DATA"
    
    idx=1
    for t in "${TIMES[@]}"; do
        if [ -n "$t" ]; then
            TYPE=$([ "$idx" -eq 1 ] && echo "Warmup" || echo "Measured")
            echo "$G_NAME,SPLA,Masked-mxm,None,$SPLA_TRI,$t,$TYPE" >> "$FINAL_CSV"
            ((idx++))
        fi
    done
done

echo "Done! Results: $FINAL_CSV"
