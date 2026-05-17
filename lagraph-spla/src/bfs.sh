#!/bin/bash

export LC_ALL=C

./tune_system.sh
caffeinate -i bash << 'EOF'

BASEDIR=$(pwd)
RUNS=30
RESULT_DIR="$BASEDIR/results/bfs"
LAGRAPH_DIR="$BASEDIR/LAGraph"
mkdir -p "$RESULT_DIR"

SPLA_EXE="$BASEDIR/spla/build/bfs"
LAGRAPH_EXE="$BASEDIR/bfs_runner"
FINAL_CSV="$RESULT_DIR/results.csv"

GRAPHS=(
    "$BASEDIR/datasets/soc-LiveJournal1.mtx"
    "$BASEDIR/datasets/patents.mtx"
)

echo "Graph,Library,Device,Iteration,Time_ms" > "$FINAL_CSV"

echo ">>> Compiling LAGraph runner..."
OMP_PATH=$(brew --prefix libomp)

clang -O3 bfs_runner.c -o "$BASEDIR/bfs_runner" \
  -Xpreprocessor -fopenmp \
  -I"$OMP_PATH/include" \
  -I"$LAGRAPH_DIR/include" \
  -I/usr/local/include/suitesparse \
  "$LAGRAPH_DIR/build/src/libLAGraph.a" \
  -L"$OMP_PATH/lib" \
  -L/usr/local/lib \
  -lgraphblas -lomp \
  -Wl,-rpath,/usr/local/lib \
  -Wl,-rpath,"$OMP_PATH/lib"

export OMP_NUM_THREADS=10

for RTYPE in "Normal" "Sym"; do
    echo -e "\n>>> Running LAGraph ($RTYPE)..."
    LOG_FILE="$RESULT_DIR/lagraph_$RTYPE.log"

    echo "--- Start log for $RTYPE ---" > "$LOG_FILE"

    for G in "${GRAPHS[@]}"; do
        if [ ! -f "$G" ]; then continue; fi
        G_NAME=$(basename "$G")
        
        SYM_FLAG=0
        [[ "$RTYPE" == "Sym" ]] && SYM_FLAG=1
        
        KIND=0
        [[ "$G_NAME" == "patents.mtx" || "$G_NAME" == "soc-LiveJournal1.mtx" ]] && KIND=1
        [[ "$RTYPE" == "Sym" ]] && KIND=0

        echo "Processing $G_NAME (Symmetrize: $SYM_FLAG)..." | tee -a "$LOG_FILE"

        for ((i=1; i<=RUNS; i++)); do
            OUTPUT=$("$LAGRAPH_EXE" "$G" "$KIND" "$SYM_FLAG" 2>&1)
            echo "$OUTPUT" >> "$LOG_FILE"
            
            TIME_SEC=$(echo "$OUTPUT" | grep "cpu(s):" | awk '{print $2}')

            if [ ! -z "$TIME_SEC" ]; then
                TIME_MS=$(awk "BEGIN {print $TIME_SEC * 1000}")
                echo "$G_NAME,LAGraph-$RTYPE,CPU,$i,$TIME_MS" >> "$FINAL_CSV"
        done
    done
done

echo -e "\n>>> Running SPLA (GPU)..."
for G in "${GRAPHS[@]}"; do
    if [ ! -f "$G" ]; then continue; fi
    G_NAME=$(basename "$G")
    echo "Processing $G_NAME..."
    
    RAW_LINE=$( "$SPLA_EXE" --mtxpath="$G" --niters=$RUNS --source=0 --run-cpu=false | grep "gpu(ms):" )
    CLEAN_DATA=$(echo "$RAW_LINE" | sed 's/gpu(ms)://' | sed 's/ //g' | sed 's/,$//')
    
    IFS=',' read -ra TIMES <<< "$CLEAN_DATA"
    it=1
    for t in "${TIMES[@]}"; do
        echo "$G_NAME,SPLA,GPU,$it,$t" >> "$FINAL_CSV"
        ((it++))
    done
done

echo -e "\nBenchmark finished! Results: $FINAL_CSV"
