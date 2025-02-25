import glob
import gzip
import json
import os
import tqdm
import math
import numpy as np
import argparse

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import glob
import os
from functools import partial

# Constants
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def load_compressed_json(file_path):
    """
    Load a gzip-compressed JSON file.

    :param file_path: Path to the compressed JSON file
    :return: Loaded JSON data or None if there's an error
    """
    try:
        with gzip.open(file_path, "rb") as f:
            return json.loads(f.read().decode("utf-8"))
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {file_path}")
        return None


def save_compressed_json(file_path, data):
    """
    Save data as a gzip-compressed JSON file.

    :param file_path: Path to save the compressed JSON file
    :param data: Data to be saved
    """
    with gzip.open(file_path, "wt") as f:
        json.dump(data, f)


def is_valid_route(results_file):
    """
    Check if a route is valid based on the attributes status and DS (excluding MinSpeedInfractions) in the
    results file.

    :param results_file: Path to the results file
    :return: Boolean indicating if the route is valid
    """
    if not os.path.exists(results_file):
        return False

    with open(results_file) as f:
        results = json.load(f)
        if not results["_checkpoint"]["records"]:
            return False

        data = results["_checkpoint"]["records"][0]
        if data["status"] in [
            "Failed - Agent couldn't be set up",
            "Failed",
            "Failed - Simulation crashed",
            "Failed - Agent crashed",
        ]:
            return False

        # Count imperfect runs as failed (except minspeedinfractions)
        if data["scores"]["score_composed"] < 100.0:
            perfect_route = math.isclose(data["scores"]["score_route"], 100)
            only_min_speed_infractions = data["num_infractions"] == len(data["infractions"]["min_speed_infractions"])
            if not perfect_route or not only_min_speed_infractions:
                return False

    return True


def extract_frame_data(frame_path, measurements_path, keyframe_keys):
    """
    Extract steer, brake, and speed from a frame and its measurements.

    :param frame_path: Path to the frame file
    :param measurements_path: Path to the measurements file
    :param keyframe_keys: List of keys to extract from measurements
    :return: Tuple of extracted data
    """
    frame_data = load_compressed_json(frame_path)
    measurements = load_compressed_json(measurements_path)

    key_states = []
    for key in keyframe_keys:
        if key in measurements:
            key_states.append(measurements[key])

        if key == "trafficlight_state":
            key_states.append(frame_data[-1]["traffic_light_state"])

    assert len(key_states) == len(keyframe_keys), f"Mismatch in key states: {len(key_states)} != {len(keyframe_keys)}"

    return (
        key_states,
        measurements["steer"],
        measurements["control_brake"],
        measurements["speed"],
    )


def process_route(args, frames, route):
    """
    Extract keyframes of a single route.

    :param args: Parsed command-line arguments
    :param frames: List of frame paths
    :return: List of keyframe paths
    """
    keyframe_paths = []

    results_file = f"{route.replace('/data/', '/results/')}.json"
    if args.filter_routes_for_DS and not is_valid_route(results_file):
        return keyframe_paths

    prev_key_states = [False] * len(args.keyframe_keys)
    prev_steer, prev_brake, prev_speed = 0, 0, 0

    for frame_path in frames:
        frame_number = int(frame_path.split("/")[-1].split(".")[0])
        if frame_number < args.skip_first_n_frames:
            continue

        measurements_path = frame_path.replace("boxes", "measurements")
        if not all(os.path.exists(p) for p in [frame_path, measurements_path]):
            continue

        key_states, steer, brake, speed = extract_frame_data(frame_path, measurements_path, args.keyframe_keys)

        path_rgb = frame_path.replace("boxes", "rgb").replace("json.gz", "jpg")

        is_keyframe = (
            key_states != prev_key_states
            or (args.use_change_in_steer and abs(steer - prev_steer) > 0.1)
            or abs(speed - prev_speed) > 1.0
            or np.random.choice([True, False], p=[0.05, 0.95])
            or brake != prev_brake
            or (speed > 1 and brake)
        )

        if is_keyframe:
            keyframe_paths.append(path_rgb)

        prev_key_states = key_states
        prev_steer, prev_brake, prev_speed = steer, brake, speed

    return keyframe_paths


def process_single_route(args, route):
    """
    Process a single route and return its keyframe paths
    """
    try:
        frames = sorted(glob.glob(f"{route}/boxes/*.json.gz"))
        keyframe_paths = process_route(args, frames, route)
        return len(frames), keyframe_paths
    except Exception as e:
        print(f"Error processing route {route}: {str(e)}")
        return 0, []


def process_routes_parallel(args, routes, num_threads=None):
    """
    Process routes in parallel using ThreadPoolExecutor

    Args:
        args: Arguments needed for processing
        routes: List of routes to process
        num_threads: Number of threads to use (defaults to CPU count * 2)

    Returns:
        tuple: (total number of frames, list of all keyframe paths)
    """
    if num_threads is None:
        num_threads = 32

    # Create a partial function with the args parameter fixed
    process_func = partial(process_single_route, args)

    # Process routes in parallel with progress bar
    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_func, route) for route in routes]

        # Track progress with tqdm
        for future in tqdm(futures, desc="Processing routes", total=len(routes)):
            result = future.result()
            results.append(result)

    # Aggregate results
    num_frames_all = sum(frames for frames, _ in results)
    all_keyframe_paths = []
    for _, keyframe_paths in results:
        all_keyframe_paths.extend(keyframe_paths)

    return num_frames_all, all_keyframe_paths


def main(args):
    # Town12: Found 71215 keyframes out of 214631 frames
    # All towns: Found 196314 keyframes out of 581662 frames
    routes = glob.glob(os.path.join(args.path_dataset, "*/*/*/*"), recursive=True)

    num_frames_all, all_keyframe_paths = process_routes_parallel(args, routes)

    with open(args.path_keyframes, "w") as f:
        f.write("\n".join(all_keyframe_paths))

    print(f"Found {len(all_keyframe_paths)} keyframes out of {num_frames_all} frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Keyframe Extraction Tool",
        description="Extract keyframes from the PDM-Lite dataset",
        epilog="This tool extracts keyframes from the PDM-Lite dataset for DriveLM-CARLA.",
    )

    parser.add_argument(
        "--path-dataset",
        default="path/to/PDM_Lite_Carla_LB2",
        required=True,
        type=str,
        help="PDM_Lite_Carla_LB2 directory containing the dataset",
    )
    parser.add_argument(
        "--path-keyframes",
        default="path/to/keyframes.txt",
        required=True,
        type=str,
        help="Path of the output file containing the keyframes",
    )

    # At the very beginning, the illumination changes when choosing a non-default weather,
    # so we skip the first n frames.
    parser.add_argument(
        "--skip-first-n-frames",
        default=10,
        type=int,
        help="Number of frames to skip at the beginning of each route. At the very beginning the "
        "illumination changes, when choosing a non-default wheather, so we skip the first n frames ",
    )
    parser.add_argument(
        "--use-change-in-steer",
        action="store_true",
        default=True,
        help="Consider changes in steering as a criterion for keyframe selection",
    )
    parser.add_argument(
        "--filter-routes-for-DS",
        action="store_true",
        default=True,
        help="Filter routes based on Driving Score (DS)",
    )
    parser.add_argument(
        "--keyframe-keys",
        default=["light_hazard", "walker_hazard", "stop_sign_hazard"],
        nargs="+",
        help="Keys to consider for keyframe selection",
    )

    args = parser.parse_args()
    main(args)
