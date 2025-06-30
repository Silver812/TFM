# PowerShell script to run a sequence of hybrid schedule training tasks.

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

	$paramsArray = $Params.Split(' ')
	python "$PSScriptRoot\run_evolution.py" $paramsArray *>&1 | Tee-Object -Append -FilePath $LogFile | Out-Null
	$status = $LASTEXITCODE

	"Python script finished with exit code: $status" | Tee-Object -Append -FilePath $LogFile | Out-Null
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
$LogFile = Join-Path -Path $LogDir -ChildPath "hybrid_tests_$Timestamp.log"

if (-not (Test-Path $LogDir)) {
	New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
}

"Starting hybrid schedule testing at $(Get-Date)" | Tee-Object -FilePath $LogFile
"Log file: $LogFile" | Tee-Object -Append -FilePath $LogFile
"" | Tee-Object -Append -FilePath $LogFile

# Configurations & Parameters
$CommonParams = "--num_games 200 --num_threads 8 --num_training 5 --max_evaluations 500 --pop_size 10 --intra_run_hof_size 0 --evaluation_mode hybrid"

$HybridSchedules = @(
	"fixed:0.4,coevolution:0.3,fixed:0.3",
	"fixed:0.7,coevolution:0.3",
	"fixed:0.4,coevolution:0.2,fixed:0.4",
	"fixed:0.5,coevolution:0.5",
	"coevolution:0.6,fixed:0.4",
	"coevolution:0.4,fixed:0.2,coevolution:0.4",
	"fixed:0.1,coevolution:0.1,fixed:0.1,coevolution:0.1,fixed:0.1,coevolution:0.1,fixed:0.1,coevolution:0.1,fixed:0.1,coevolution:0.1",
	"fixed:0.2,coevolution:0.2,fixed:0.2,coevolution:0.2,fixed:0.2",
	"fixed:0.8,coevolution:0.2",
	"coevolution:0.8,fixed:0.2",
	"fixed:0.2,coevolution:0.5,fixed:0.3",
	"fixed:0.3,coevolution:0.4,fixed:0.3",
	"coevolution:0.3,fixed:0.4,coevolution:0.3",
	"fixed:0.1,coevolution:0.8,fixed:0.1",
	"coevolution:0.1,fixed:0.8,coevolution:0.1",
	"fixed:0.1,coevolution:0.2,fixed:0.1,coevolution:0.6",
	"coevolution:0.1,fixed:0.2,coevolution:0.1,fixed:0.6",
	"fixed:0.9,coevolution:0.1",
	"coevolution:0.9,fixed:0.1",
	"fixed:0.33,coevolution:0.34,fixed:0.33"
)

$Statuses = @()
$ConfigNames = @()

# Run all hybrid configurations
for ($i = 0; $i -lt $HybridSchedules.Count; $i++) {
	$schedule = $HybridSchedules[$i]
	$segments = ($schedule.Split(',')).Count
	$fixed_pct_sum = ($schedule | Select-String -Pattern "fixed:([\d.]+)" -AllMatches).Matches.Groups[1].Value | ForEach-Object { [double]$_ } | Measure-Object -Sum | Select-Object -ExpandProperty Sum
	$coevo_pct_sum = ($schedule | Select-String -Pattern "coevolution:([\d.]+)" -AllMatches).Matches.Groups[1].Value | ForEach-Object { [double]$_ } | Measure-Object -Sum | Select-Object -ExpandProperty Sum
	$fixed_pct = [int]($fixed_pct_sum * 100)
	$coevo_pct = [int]($coevo_pct_sum * 100)
	$config_name = "$($i+1)-Hybrid-${segments}seg-F${fixed_pct}C${coevo_pct}"
	$ConfigNames += $config_name
	$status = Start-Training "$config_name" "$CommonParams --hybrid_schedule_str $schedule" -LogFile $LogFile
	$Statuses += $status
}

# Summary
"Hybrid Schedule Testing Summary:" | Tee-Object -Append -FilePath $LogFile
for ($i = 0; $i -lt $HybridSchedules.Count; $i++) {
	# Check the status from our array
	$status_msg = if ($Statuses[$i] -eq 0) { 'SUCCESS' } else { 'FAILED' }
    
	# Retrieve the pre-generated name
	$config_name = $ConfigNames[$i]
	$schedule = $HybridSchedules[$i]

	# Extract percentage part from the name for the summary line
	$pct_part = $config_name.Split('-')[2]
	$segments = $config_name.Split('-')[1]

	" - $config_name [$status_msg]: $pct_part ($schedule)" | Tee-Object -Append -FilePath $LogFile
}
"All hybrid tests completed at $(Get-Date)" | Tee-Object -Append -FilePath $LogFile