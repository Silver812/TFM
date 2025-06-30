# PowerShell script to run a sequence of training tasks on Windows

# Function to run a single training configuration
function Start-Training {
    param(
        [string]$ConfigName,
        [string]$Params,
        [string]$LogFile
    )

    "========================================================================" | Tee-Object -Append -FilePath $LogFile | Out-Null
    "STARTING TRAINING: $ConfigName at $(Get-Date)" | Tee-Object -Append -FilePath $LogFile | Out-Null
    "PARAMETERS: $Params" | Tee-Object -Append -FilePath $LogFile | Out-Null
    "========================================================================" | Tee-Object -Append -FilePath $LogFile | Out-Null

    # Split parameters into array
    $paramsArray = $Params.Split(' ')

    # Execute evolution script and capture all output to log file
    python "$PSScriptRoot\run_evolution.py" $paramsArray *>&1 | Tee-Object -Append -FilePath $LogFile | Out-Null
    $status = $LASTEXITCODE

    "========================================================================" | Tee-Object -Append -FilePath $LogFile | Out-Null
    if ($status -eq 0) {
        "COMPLETED: $ConfigName at $(Get-Date)" | Tee-Object -Append -FilePath $LogFile | Out-Null
    }
    else {
        "FAILED: $ConfigName with status $status at $(Get-Date)" | Tee-Object -Append -FilePath $LogFile | Out-Null
    }
    "========================================================================" | Tee-Object -Append -FilePath $LogFile | Out-Null
    "" | Tee-Object -Append -FilePath $LogFile | Out-Null

    return $status
}

# Setup
$ScriptDir = $PSScriptRoot
$LogDir = Join-Path -Path $ScriptDir -ChildPath "..\training_logs"
$Timestamp = (Get-Date -Format "yyyyMMdd_HHmmss")
$LogFile = Join-Path -Path $LogDir -ChildPath "training_runs_$Timestamp.log"

if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
}

"Starting training sequence at $(Get-Date)" | Tee-Object -FilePath $LogFile
"Log file: $LogFile" | Tee-Object -Append -FilePath $LogFile
"" | Tee-Object -Append -FilePath $LogFile

# Common parameters
$CommonParams = "--num_games 200 --num_threads 8 --num_training 5"

# Configurations
$FixedStatus = Start-Training "Fixed Mode Training HoF 3" "$CommonParams --max_evaluations 500 --pop_size 10 --intra_run_hof_size 3 --evaluation_mode fixed" -LogFile $LogFile
$HybridStatus = Start-Training "Hybrid Mode Training HoF 3" "$CommonParams --max_evaluations 500 --pop_size 10 --intra_run_hof_size 3 --evaluation_mode hybrid --hybrid_schedule_str fixed:0.4,coevolution:0.2,fixed:0.4" -LogFile $LogFile
$CoevoStatus = Start-Training "Coevolution Mode Training HoF 3" "$CommonParams --max_evaluations 500 --pop_size 10 --intra_run_hof_size 3 --evaluation_mode coevolution" -LogFile $LogFile
$FixedStatus2 = Start-Training "Fixed Mode Training HoF 0" "$CommonParams --max_evaluations 500 --pop_size 10 --intra_run_hof_size 0 --evaluation_mode fixed" -LogFile $LogFile
$HybridStatus2 = Start-Training "Hybrid Mode Training HoF 0" "$CommonParams --max_evaluations 500 --pop_size 10 --intra_run_hof_size 0 --evaluation_mode hybrid --hybrid_schedule_str fixed:0.4,coevolution:0.2,fixed:0.4" -LogFile $LogFile
$CoevoStatus2 = Start-Training "Coevolution Mode Training HoF 0" "$CommonParams --max_evaluations 500 --pop_size 10 --intra_run_hof_size 0 --evaluation_mode coevolution" -LogFile $LogFile

# Summary
"Training Sequence Summary:" | Tee-Object -Append -FilePath $LogFile
" - Fixed Mode Training HoF 3: $(if ($FixedStatus -eq 0) {'SUCCESS'} else {'FAILED'})" | Tee-Object -Append -FilePath $LogFile
" - Hybrid Mode Training HoF 3: $(if ($HybridStatus -eq 0) {'SUCCESS'} else {'FAILED'})" | Tee-Object -Append -FilePath $LogFile
" - Coevolution Mode Training HoF 3: $(if ($CoevoStatus -eq 0) {'SUCCESS'} else {'FAILED'})" | Tee-Object -Append -FilePath $LogFile
" - Fixed Mode Training HoF 0: $(if ($FixedStatus2 -eq 0) {'SUCCESS'} else {'FAILED'})" | Tee-Object -Append -FilePath $LogFile
" - Hybrid Mode Training HoF 0: $(if ($HybridStatus2 -eq 0) {'SUCCESS'} else {'FAILED'})" | Tee-Object -Append -FilePath $LogFile
" - Coevolution Mode Training HoF 0: $(if ($CoevoStatus2 -eq 0) {'SUCCESS'} else {'FAILED'})" | Tee-Object -Append -FilePath $LogFile
"All training runs completed at $(Get-Date)" | Tee-Object -Append -FilePath $LogFile