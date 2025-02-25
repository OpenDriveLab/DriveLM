"""
PDM Lite Carla Leaderboard 2 Downloader
"""

import pathlib
import os
import requests
from tqdm import tqdm
import concurrent.futures
from typing import List, Tuple
import threading
import zipfile
import shutil
from pathlib import Path

dataset_name = "PDM_Lite_Carla_LB2"

# Create main dataset directory
pathlib.Path(dataset_name).mkdir(exist_ok=True)

base_url = (
    "https://huggingface.co/datasets/autonomousvision/PDM_Lite_Carla_LB2/resolve/main/"
)

# Thread-local storage for progress bars
thread_local = threading.local()


class DownloadTracker:
    def __init__(self):
        self.total_files = 0
        self.completed_files = 0
        self.lock = threading.Lock()
        self.progress_bar = None

    def increment(self):
        with self.lock:
            self.completed_files += 1
            if self.progress_bar:
                self.progress_bar.update(1)


tracker = DownloadTracker()


def get_session() -> requests.Session:
    """Creates or returns a thread-local session object."""
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    return thread_local.session


class SecurityError(Exception):
    """Custom exception for security-related issues."""

    pass


def unzip_file(args: Tuple[str, str, str]) -> None:
    """
    Unzips a single file with validation and error handling.

    Args:
        args: Tuple containing (url, destination, filename)
    """
    url, destination, filename = args

    if not filename.endswith(".zip"):
        return

    try:
        zip_path = Path(destination)
        parent_dir = zip_path.parent  # We'll extract directly to parent directory

        # Validate zip file
        if not zipfile.is_zipfile(destination):
            raise zipfile.BadZipFile(f"{filename} is not a valid ZIP file")

        with zipfile.ZipFile(destination, "r") as zip_ref:
            # Test zip file integrity
            if zip_ref.testzip() is not None:
                raise zipfile.BadZipFile(f"{filename} failed integrity check")

            # Get total uncompressed size
            zip_size = zip_path.stat().st_size
            total_size = sum(info.file_size for info in zip_ref.filelist)
            if total_size > zip_size * 10:  # Basic zip bomb protection
                raise SecurityError(f"Suspicious compression ratio in {filename}")

            # Check for suspicious paths and get root level directories
            root_level_items = set()
            for zip_info in zip_ref.filelist:
                # Get the first part of the path
                parts = Path(zip_info.filename).parts
                if parts:
                    root_level_items.add(parts[0])

                # Check for path traversal
                target_path = Path(parent_dir) / zip_info.filename
                if not str(target_path).startswith(str(parent_dir)):
                    raise SecurityError(f"Attempted path traversal in {filename}")

            # If there's exactly one root directory and it matches our zip name stem,
            # we'll extract directly to parent_dir to avoid redundancy
            if len(root_level_items) == 1 and zip_path.stem in root_level_items:
                extract_dir = parent_dir
            else:
                extract_dir = parent_dir / zip_path.stem
                extract_dir.mkdir(exist_ok=True)

            # Extract files with progress tracking
            for member in zip_ref.filelist:
                zip_ref.extract(member, str(extract_dir))

        print(f"Successfully extracted {filename} to {extract_dir}")

        # Remove the zip file after successful extraction
        zip_path.unlink()

    except zipfile.BadZipFile as e:
        print(f"Error: {filename} is corrupted: {e}")
    except SecurityError as e:
        print(f"Security Error with {filename}: {e}")
    except OSError as e:
        print(f"System Error processing {filename}: {e}")
    except Exception as e:
        print(f"Unexpected error extracting {filename}: {e}")
    finally:
        tracker.increment()


def download_file(args: Tuple[str, str, str]) -> None:
    """
    Download a single file using thread-local session.

    Args:
        args: Tuple containing (url, destination, filename)
    """
    url, destination, filename = args
    session = get_session()

    try:
        response = session.get(url, stream=True)
        response.raise_for_status()

        with open(destination, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)

        tracker.increment()

    except requests.exceptions.RequestException as e:
        print(f"\nError downloading {filename}: {e}")


def download_files(directory: str, files: List[str]) -> List[Tuple[str, str, str]]:
    """
    Prepare directory and return download information for multiple files.

    Args:
        directory: The relative directory path where files should be downloaded
        files: List of filenames to download

    Returns:
        List of tuples containing (url, destination_path, filename) for each file
    """
    full_dir_path = os.path.join(dataset_name, directory)
    pathlib.Path(full_dir_path).mkdir(exist_ok=True)

    download_info = []
    for file in files:
        download_url = base_url + directory + file
        destination_path = os.path.join(full_dir_path, file)
        download_info.append((download_url, destination_path, file))

    return download_info


# Dictionary of towns and their scenarios
town_scenarios = {
    "Town01": [
        "ControlLoss",
        "DynamicObjectCrossing",
        "OppositeVehicleRunningRedLight",
        "SignalizedJunctionLeftTurn",
        "SignalizedJunctionRightTurn",
        "VehicleTurningRoute",
    ],
    "Town02": [
        "ControlLoss",
        "DynamicObjectCrossing",
        "OppositeVehicleRunningRedLight",
        "SignalizedJunctionLeftTurn",
        "SignalizedJunctionRightTurn",
        "VehicleTurningRoute",
    ],
    "Town03": [
        "ControlLoss",
        "NoScenario",
        "DynamicObjectCrossing",
        "OppositeVehicleRunningRedLight",
        "SignalizedJunctionLeftTurn",
        "SignalizedJunctionRightTurn",
        "VehicleTurningRoute",
    ],
    "Town04": [
        "ControlLoss",
        "NoScenario",
        "DynamicObjectCrossing",
        "OppositeVehicleRunningRedLight",
        "SignalizedJunctionLeftTurn",
        "SignalizedJunctionRightTurn",
        "VehicleTurningRoute",
    ],
    "Town05": [
        "ControlLoss",
        "NoScenario",
        "DynamicObjectCrossing",
        "OppositeVehicleRunningRedLight",
        "SignalizedJunctionLeftTurn",
        "SignalizedJunctionRightTurn",
        "VehicleTurningRoute",
    ],
    "Town10": [
        "ControlLoss",
        "NoScenario",
        "DynamicObjectCrossing",
        "OppositeVehicleRunningRedLight",
        "SignalizedJunctionLeftTurn",
        "SignalizedJunctionRightTurn",
        "VehicleTurningRoute",
    ],
    "Town12": [
        "Accident",
        "AccidentTwoWays",
        "BlockedIntersection",
        "ConstructionObstacle",
        "ConstructionObstacleTwoWays",
        "ControlLoss",
        "CrossingBicycleFlow",
        "DynamicObjectCrossing",
        "EnterActorFlow",
        "EnterActorFlowV2",
        "HardBreakRoute",
        "HazardAtSideLane",
        "HazardAtSideLaneTwoWays",
        "HighwayCutIn",
        "HighwayExit",
        "InterurbanActorFlow",
        "InterurbanAdvancedActorFlow",
        "InvadingTurn",
        "MergerIntoSlowTraffic",
        "MergerIntoSlowTrafficV2",
        "NonSignalizedJunctionLeftTurn",
        "NonSignalizedJunctionRightTurn",
        "OppositeVehicleRunningRedLight",
        "OppositeVehicleTakingPriority",
        "ParkedObstacle",
        "ParkedObstacleTwoWays",
        "ParkingCrossingPedestrian",
        "ParkingCutIn",
        "ParkingExit",
        "PedestrianCrossing",
        "PriorityAtJunction",
        "SignalizedJunctionLeftTurn",
        "SignalizedJunctionRightTurn",
        "StaticCutIn",
        "VehicleOpensDoorTwoWays",
        "VehicleTurningRoute",
        "VehicleTurningRoutePedestrian",
        "YieldToEmergencyVehicle",
    ],
    "Town13": [
        "Accident",
        "AccidentTwoWays",
        "BlockedIntersection",
        "ConstructionObstacle",
        "ConstructionObstacleTwoWays",
        "ControlLoss",
        "DynamicObjectCrossing",
        "EnterActorFlow",
        "HardBreakRoute",
        "HazardAtSideLane",
        "HazardAtSideLaneTwoWays",
        "HighwayCutIn",
        "HighwayExit",
        "InterurbanActorFlow",
        "InterurbanAdvancedActorFlow",
        "InvadingTurn",
        "MergerIntoSlowTraffic",
        "MergerIntoSlowTrafficV2",
        "NonSignalizedJunctionLeftTurn",
        "NonSignalizedJunctionRightTurn",
        "OppositeVehicleRunningRedLight",
        "OppositeVehicleTakingPriority",
        "ParkedObstacle",
        "ParkedObstacleTwoWays",
        "ParkingCrossingPedestrian",
        "ParkingCutIn",
        "ParkingExit",
        "PedestrianCrossing",
        "PriorityAtJunction",
        "SignalizedJunctionLeftTurn",
        "SignalizedJunctionRightTurn",
        "StaticCutIn",
        "VehicleOpensDoorTwoWays",
        "VehicleTurningRoute",
        "VehicleTurningRoutePedestrian",
        "YieldToEmergencyVehicle",
    ],
}


def main():
    # Collect all download tasks
    download_tasks = []

    # Add README
    download_tasks.extend(download_files("", ["README.md"]))

    # Add all town files
    for town, scenarios in town_scenarios.items():
        # Add results.zip
        download_tasks.extend(download_files(f"{town}/", ["results.zip"]))

        # Add scenario data
        download_tasks.extend(
            download_files(
                f"{town}/data/", [f"{scenario}.zip" for scenario in scenarios]
            )
        )

    # Set up the progress tracking
    tracker.total_files = len(download_tasks)
    tracker.progress_bar = tqdm(
        total=tracker.total_files, desc="Downloading files", unit="file"
    )

    # Download files in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(download_file, download_tasks)

    tracker.progress_bar.close()
    print("\nDownload completed!")

    # Set up the progress tracking
    tracker.total_files = len(download_tasks)
    tracker.progress_bar = tqdm(
        total=tracker.total_files, desc="Unzip files", unit="file"
    )

    # Download files in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(unzip_file, download_tasks)

    tracker.progress_bar.close()
    print("\nUnzipping completed!")


if __name__ == "__main__":
    main()
