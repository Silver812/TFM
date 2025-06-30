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
LOG_FILE="${LOG_DIR}/hybrid_tests_${TIMESTAMP}.log"
mkdir -p $LOG_DIR
echo "Starting hybrid schedule testing at $(date)" | tee $LOG_FILE
echo "Log file: $LOG_FILE" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Common parameters for all tests
COMMON_PARAMS="--num_games 50 --num_threads 5 --num_training 5 --max_evaluations 500 --pop_size 10 --intra_run_hof_size 0 --evaluation_mode hybrid"

# Array of hybrid schedules to test
declare -a HYBRID_SCHEDULES=(
    "fixed:0.4,coevolution:0.3,fixed:0.3"
    "fixed:0.7,coevolution:0.3"
    "fixed:0.4,coevolution:0.2,fixed:0.4"
    "fixed:0.5,coevolution:0.5"
    "coevolution:0.6,fixed:0.4"
    "coevolution:0.4,fixed:0.2,coevolution:0.4"
    "fixed:0.1,coevolution:0.1,fixed:0.1,coevolution:0.1,fixed:0.1,coevolution:0.1,fixed:0.1,coevolution:0.1,fixed:0.1,coevolution:0.1"
    "fixed:0.2,coevolution:0.2,fixed:0.2,coevolution:0.2,fixed:0.2"
    "fixed:0.8,coevolution:0.2"
    "coevolution:0.8,fixed:0.2"
    "fixed:0.2,coevolution:0.5,fixed:0.3"
    "fixed:0.3,coevolution:0.4,fixed:0.3"
    "coevolution:0.3,fixed:0.4,coevolution:0.3"
    "fixed:0.1,coevolution:0.8,fixed:0.1"
    "coevolution:0.1,fixed:0.8,coevolution:0.1"
    "fixed:0.1,coevolution:0.2,fixed:0.1,coevolution:0.6"
    "coevolution:0.1,fixed:0.2,coevolution:0.1,fixed:0.6"
    "fixed:0.9,coevolution:0.1"
    "coevolution:0.9,fixed:0.1"
    "fixed:0.33,coevolution:0.34,fixed:0.33"
)

# Create array to store status results
declare -a STATUSES=()

# Run all hybrid configurations
for i in "${!HYBRID_SCHEDULES[@]}"; do
    schedule="${HYBRID_SCHEDULES[$i]}"
    
    # Format: segments-F{fixed_pct}C{coevo_pct}
    segments=$(echo "$schedule" | tr -cd ',' | wc -c)
    segments=$((segments + 1))
    
    fixed_pct=$(echo "$schedule" | grep -o "fixed:[0-9.]*" | cut -d: -f2 | awk '{sum+=$1} END {print int(sum*100)}')
    coevo_pct=$(echo "$schedule" | grep -o "coevolution:[0-9.]*" | cut -d: -f2 | awk '{sum+=$1} END {print int(sum*100)}')
    
    config_name="Hybrid-${segments}seg-F${fixed_pct}C${coevo_pct}"
    
    # Add sequence number for identification
    config_name="$((i+1))-$config_name"
    
    # Run the training
    run_training "$config_name" "$COMMON_PARAMS --hybrid_schedule_str $schedule"
    STATUSES[$i]=$?
done

# Summary
echo "Hybrid Schedule Testing Summary:" | tee -a $LOG_FILE
for i in "${!HYBRID_SCHEDULES[@]}"; do
    status_msg=$([ ${STATUSES[$i]} -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')
    
    segments=$(echo "${HYBRID_SCHEDULES[$i]}" | tr -cd ',' | wc -c)
    segments=$((segments + 1))
    
    fixed_pct=$(echo "${HYBRID_SCHEDULES[$i]}" | grep -o "fixed:[0-9.]*" | cut -d: -f2 | awk '{sum+=$1} END {print int(sum*100)}')
    coevo_pct=$(echo "${HYBRID_SCHEDULES[$i]}" | grep -o "coevolution:[0-9.]*" | cut -d: -f2 | awk '{sum+=$1} END {print int(sum*100)}')
    
    echo "- $((i+1)) [$status_msg]: ${segments}seg F${fixed_pct}%/C${coevo_pct}% (${HYBRID_SCHEDULES[$i]})" | tee -a $LOG_FILE
done
echo "All hybrid tests completed at $(date)" | tee -a $LOG_FILE

# Usage instructions:
# chmod +x run_all_trainings_hybrid.sh
# nohup ./run_all_trainings_hybrid.sh &
# tail -f ../training_logs/hybrid_tests_*.log
# grep -A20 "Hybrid Schedule Testing Summary:" ../training_logs/hybrid_tests_*.log