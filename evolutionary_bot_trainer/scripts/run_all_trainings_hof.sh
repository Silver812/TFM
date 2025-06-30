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
COMMON_PARAMS="--num_games 50 --num_threads 10 --num_training 5"

# Configurations
run_training "Fixed Mode Training HoF 4" "$COMMON_PARAMS --max_evaluations 1000 --pop_size 100  --intra_run_hof_size 3 --evaluation_mode fixed"
FIXED_STATUS=$?

run_training "Hybrid Mode Training HoF 4" "$COMMON_PARAMS --max_evaluations 500 --pop_size 10 --intra_run_hof_size 3 --evaluation_mode hybrid --hybrid_schedule_str fixed:0.4,coevolution:0.2,fixed:0.4"
HYBRID_STATUS=$?

run_training "Coevolution Mode Training HoF 4" "$COMMON_PARAMS --max_evaluations 200 --pop_size 10 --intra_run_hof_size 3 --evaluation_mode coevolution"
COEVO_STATUS=$?

run_training "Fixed Mode Training HoF 0" "$COMMON_PARAMS --max_evaluations 1000 --pop_size 100  --intra_run_hof_size 0 --evaluation_mode fixed"
FIXED_STATUS2=$?

run_training "Hybrid Mode Training HoF 0" "$COMMON_PARAMS --max_evaluations 500 --pop_size 10 --intra_run_hof_size 0 --evaluation_mode hybrid --hybrid_schedule_str fixed:0.4,coevolution:0.2,fixed:0.4"
HYBRID_STATUS2=$?

run_training "Coevolution Mode Training HoF 0" "$COMMON_PARAMS --max_evaluations 200 --pop_size 10 --intra_run_hof_size 0 --evaluation_mode coevolution"
COEVO_STATUS2=$?

# Summary 1
echo "Training Sequence Summary:" | tee -a $LOG_FILE
echo "- Fixed Mode HoF 3: $([ $FIXED_STATUS -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')" | tee -a $LOG_FILE
echo "- Hybrid Mode HoF 3: $([ $HYBRID_STATUS -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')" | tee -a $LOG_FILE
echo "- Coevolution Mode HoF 3: $([ $COEVO_STATUS -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')" | tee -a $LOG_FILE
echo "- Fixed Mode HoF 0: $([ $FIXED_STATUS2 -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')" | tee -a $LOG_FILE
echo "- Hybrid Mode HoF 0: $([ $HYBRID_STATUS2 -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')" | tee -a $LOG_FILE
echo "- Coevolution Mode HoF 0: $([ $COEVO_STATUS2 -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')" | tee -a $LOG_FILE
echo "All training runs completed at $(date)" | tee -a $LOG_FILE

# Usage instructions:
# chmod +x run_all_trainings_1h_hof.sh
# nohup ./run_all_trainings_1h_hof.sh &
# tail -f ../training_logs/training_runs_*.log
# grep -A4 "Training Sequence Summary" ../training_logs/training_runs_*.log