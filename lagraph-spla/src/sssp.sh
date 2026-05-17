#!/bin/bash

./tune_system.sh
caffeinate -i bash << 'EOF'

BASEDIR=$(pwd)
OMP_PATH=$(brew --prefix libomp)
RESULT_DIR="$BASEDIR/results/sssp"
LAGRAPH_EXE="$BASEDIR/sssp_runner"
SPLA_EXE="$BASEDIR/spla/build/sssp" 
FINAL_CSV="$RESULT_DIR/sssp_results.csv"
LOG_FILE="$RESULT_DIR/sssp_execution.log"

mkdir -p "$RESULT_DIR"
echo "=== SSSP Execution Log: $(date) ===" > "$LOG_FILE"

export DYLD_LIBRARY_PATH=/usr/local/lib:$OMP_PATH/lib:$DYLD_LIBRARY_PATH
export OMP_NUM_THREADS=12

echo ">>> Compiling LAGraph SSSP runner..."
clang -O3 sssp_runner.c -o "$LAGRAPH_EXE" \
  -Xpreprocessor -fopenmp \
  -I"$OMP_PATH/include" \
  -I"$BASEDIR/LAGraph/include" \
  -I/usr/local/include/suitesparse \
  "$BASEDIR/LAGraph/build/src/libLAGraph.a" \
  -L"$OMP_PATH/lib" \
  -L/usr/local/lib \
  -lgraphblas -lomp

echo "Graph,Library,Version,Delta,Iteration,Time_ms" > "$FINAL_CSV"

GRAPHS=("datasets/soc-LiveJournal1.mtx" "datasets/patents.mtx")
DELTAS=(0.5 1.0 2.0 5.0 7.5 10.0)
RUNS=20

for RTYPE in "Normal" "Sym"; do
    echo -e "\n>>> Running LAGraph ($RTYPE)..."
    for G_REL in "${GRAPHS[@]}"; do
        G_PATH="$BASEDIR/$G_REL"
        [ ! -f "$G_PATH" ] && continue
        G_NAME=$(basename "$G_REL")
        SYM_FLAG=0; KIND=1
        if [ "$RTYPE" == "Sym" ]; then SYM_FLAG=1; KIND=0; fi

        for DELTA in "${DELTAS[@]}"; do
            echo "Processing $G_NAME ($RTYPE) Delta=$DELTA..."
            for ((i=1; i<=RUNS; i++)); do
                echo -e "\n[LAGraph] Graph: $G_NAME, Version: $RTYPE, Delta: $DELTA, Run: $i" >> "$LOG_FILE"
                
                # Запуск
                OUTPUT=$($LAGRAPH_EXE "$G_PATH" "$KIND" "$SYM_FLAG" "$DELTA" 2>&1)
                echo "$OUTPUT" >> "$LOG_FILE"
                
                # Парсинг времени (ищем строку TIME:)
                TIME_SEC=$(echo "$OUTPUT" | grep "TIME:" | awk '{print $2}')
                
                if [ -n "$TIME_SEC" ]; then
                    # Безопасный расчет ms
                    TIME_MS=$(awk -v s="$TIME_SEC" 'BEGIN {print s * 1000}')
                    echo "$G_NAME,LAGraph,$RTYPE,$DELTA,$i,$TIME_MS" >> "$FINAL_CSV"
                    echo "  Run $i: $TIME_MS ms"
                else
                    echo "  Run $i: FAILED"
                fi
            done
        done
    done
done

echo -e "\n>>> Running SPLA (GPU)..."
for G in "${GRAPHS[@]}"; do
    [ ! -f "$BASEDIR/$G" ] && continue
    G_NAME=$(basename "$G")
    echo "Processing $G_NAME..."
    
    echo -e "\n[SPLA] Graph: $G_NAME, Mode: GPU" >> "$LOG_FILE"
    
    TEMP_OUT=$( "$SPLA_EXE" --mtxpath="$BASEDIR/$G" --niters=$RUNS --source=0 --run-cpu=false 2>&1 )
    echo "$TEMP_OUT" >> "$LOG_FILE"
    
    RAW_LINE=$(echo "$TEMP_OUT" | grep "gpu(ms):")
    CLEAN_DATA=$(echo "$RAW_LINE" | sed 's/gpu(ms)://' | sed 's/ //g' | sed 's/,$//')
    
    IFS=',' read -ra TIMES <<< "$CLEAN_DATA"
    it=1
    for t in "${TIMES[@]}"; do
        echo "$G_NAME,SPLA,GPU,N/A,$it,$t" >> "$FINAL_CSV"
        ((it++))
    done
done

echo -e "\nBenchmark finished!"
echo "CSV results: $FINAL_CSV"
echo "Detailed log: $LOG_FILE"
