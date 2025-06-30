#!/bin/bash

# Run the training with the specified parameters
run_training() {
    local config_name="$1"
    local params="$2"

    echo "========================================================================" | tee -a $LOG_FILE
    echo "STARTING TRAINING: $config_name at $(date)" | tee -a $LOG_FILE
    echo "PARAMETERS: $params" | tee -a $LOG_FILE
    echo "========================================================================" | tee -a $LOG_FILE

    python3 "$SCRIPT_DIR/run_evolution.py" $params 2>&1 | tee -a $LOG_FILE

    local status=${PIPESTATUS[0]}
    echo "========================================================================" | tee -a $LOG_FILE
    if [ $status -eq 0 ]; then
        echo "COMPLETED: $config_name at $(date)" | tee -a $LOG_FILE
    else
        echo "FAILED: $config_name with status $status at $(date)" | tee -a $LOG_FILE
    fi
    echo "========================================================================" | tee -a $LOG_FILE
    echo "" | tee -a $LOG_FILE

    return $status
}

# Setup
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_DIR="$SCRIPT_DIR/../training_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/training_runs_${TIMESTAMP}.log"
mkdir -p $LOG_DIR
echo "Starting training sequence at $(date)" | tee $LOG_FILE
echo "Log file: $LOG_FILE" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE
COMMON_PARAMS="--num_games 50 --num_threads 5 --num_training 5"

# Configurations
run_training "Fixed Mode Training" "$COMMON_PARAMS --max_evaluations 7000 --pop_size 100 --evaluation_mode fixed"
FIXED_STATUS=$?

run_training "Hybrid Mode Training" "$COMMON_PARAMS --max_evaluations 7000 --pop_size 20 --evaluation_mode hybrid"
HYBRID_STATUS=$?

run_training "Coevolution Mode Training" "$COMMON_PARAMS --max_evaluations 4000 --pop_size 10 --evaluation_mode coevolution"
COEVO_STATUS=$?

# Summary
echo "Training Sequence Summary:" | tee -a $LOG_FILE
echo "- Fixed Mode: $([ $FIXED_STATUS -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')" | tee -a $LOG_FILE
echo "- Hybrid Mode: $([ $HYBRID_STATUS -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')" | tee -a $LOG_FILE
echo "- Coevolution Mode: $([ $COEVO_STATUS -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')" | tee -a $LOG_FILE
echo "All training runs completed at $(date)" | tee -a $LOG_FILE

# Usage instructions:
# chmod +x run_all_trainings_base.sh
# nohup ./run_all_trainings_base.sh &
# tail -f ../training_logs/training_runs_*.log
# grep -A4 "Training Sequence Summary" ../training_logs/training_runs_*.log