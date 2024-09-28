#!/bin/bash

# PDM-Lite Dataset Download Script
# This script downloads and extracts the PDM-Lite dataset in parallel

set -e  # Exit on error

# Create and navigate to the dataset directory
mkdir -p pdm_lite
cd pdm_lite

# Function to download and process a file
download_and_process() {
    local url=$1
    local filename=$(basename "$url")
    
    echo "Downloading $filename..."
    wget -N "$url"  # -N to avoid re-downloading if the file exists
    
    if [[ "$filename" == *.zip ]]; then
        echo "Extracting $filename..."
        unzip -q "$filename"
        rm "$filename"
        echo "$filename extracted and removed."
    fi
}

# Export function for parallel processing
export -f download_and_process

# Get the number of available processors
num_procs=$(nproc)

# Download main files in parallel (README.md and results.zip)
parallel -j "$num_procs" download_and_process ::: \
    "https://huggingface.co/datasets/Jens21/pdm_lite/resolve/main/results.zip" \
    "https://huggingface.co/datasets/Jens21/pdm_lite/resolve/main/README.md"

# Move into data directory for further downloads
mkdir -p data
cd data

# Array of data files to download
files=(
    "Accident.zip"
    "AccidentTwoWays.zip"
    "BlockedIntersection.zip"
    "ConstructionObstacle.zip"
    "ConstructionObstacleTwoWays.zip"
    "ControlLoss.zip"
    "CrossingBicycleFlow.zip"
    "DynamicObjectCrossing.zip"
    "EnterActorFlow.zip"
    "EnterActorFlowV2.zip"
    "HardBreakRoute.zip"
    "HazardAtSideLane.zip"
    "HazardAtSideLaneTwoWays.zip"
    "HighwayCutIn.zip"
    "HighwayExit.zip"
    "InterurbanActorFlow.zip"
    "InterurbanAdvancedActorFlow.zip"
    "InvadingTurn.zip"
    "MergerIntoSlowTraffic.zip"
    "MergerIntoSlowTrafficV2.zip"
    "NonSignalizedJunctionLeftTurn.zip"
    "NonSignalizedJunctionRightTurn.zip"
    "OppositeVehicleRunningRedLight.zip"
    "OppositeVehicleTakingPriority.zip"
    "ParkedObstacle.zip"
    "ParkedObstacleTwoWays.zip"
    "ParkingCrossingPedestrian.zip"
    "ParkingCutIn.zip"
    "ParkingExit.zip"
    "PedestrianCrossing.zip"
    "PriorityAtJunction.zip"
    "SignalizedJunctionLeftTurn.zip"
    "SignalizedJunctionRightTurn.zip"
    "StaticCutIn.zip"
    "VehicleOpensDoorTwoWays.zip"
    "VehicleTurningRoute.zip"
    "VehicleTurningRoutePedestrian.zip"
    "YieldToEmergencyVehicle.zip"
)

# Base URL for data files
base_url="https://huggingface.co/datasets/Jens21/pdm_lite/resolve/main/data"

# Download and process files in parallel using available CPU cores
echo "Downloading and extracting data files using $num_procs processors..."
parallel -j "$num_procs" download_and_process "${base_url}/{}" ::: "${files[@]}"

echo "All downloads and extractions complete."
