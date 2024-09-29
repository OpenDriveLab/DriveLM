"""
This script is used to start jobs on a SLURM cluster.
Also checkout start_expert_slurm_dynports.py
"""

import os
import subprocess
from pathlib import Path

def make_bash(data_save_root, code_root, route_number, route_full, ckpt_endpoint, save_path, seed, ckpt_endpoint_root):
    Path(f"{code_root}/{save_path}").mkdir(exist_ok=True)
    Path(f"{ckpt_endpoint_root}").mkdir(exist_ok=True)

    job_file = f"{data_save_root}/start_files/{route_number}.sh"
    # create folder
    carla_root = "/home/carla"
    run_command = "python leaderboard/leaderboard/leaderboard_evaluator_local.py --port=${FREE_WORLD_PORT} \
    --traffic-manager-port=${TM_PORT} --traffic-manager-seed=${TM_SEED} --routes=${ROUTES} --repetitions=${REPETITIONS} \
        --track=${CHALLENGE_TRACK_CODENAME} --checkpoint=${CHECKPOINT_ENDPOINT} --agent=${TEAM_AGENT} \
            --agent-config=${TEAM_CONFIG} --debug=0 --resume=${RESUME} --timeout=2400"

    qsub_template = f"""#!/bin/bash
. /opt/miniconda3/etc/profile.d/conda.sh
conda activate carla0914

export SCENARIO_RUNNER_ROOT={code_root}/scenario_runner
export LEADERBOARD_ROOT={code_root}/leaderboard

# carla
export CARLA_ROOT=/home/carla
export CARLA_SERVER={carla_root}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:{carla_root}/PythonAPI
export PYTHONPATH=$PYTHONPATH:{carla_root}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:{carla_root}/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:scenario_runner
export REPETITIONS=1
export DEBUG_CHALLENGE=0
export TEAM_AGENT={code_root}/team_code/autopilot.py
export CHALLENGE_TRACK_CODENAME=MAP
export ROUTES={route_full}
export TM_SEED={seed}

export CHECKPOINT_ENDPOINT={code_root}/{ckpt_endpoint}
export TEAM_CONFIG={route_full}
export RESUME=1
export DATAGEN=0
export SAVE_PATH={code_root}/{save_path}

echo "Start python"

export FREE_STREAMING_PORT=$1
export FREE_WORLD_PORT=$2
export TM_PORT=$3

echo "FREE_STREAMING_PORT: $FREE_STREAMING_PORT"
echo "FREE_WORLD_PORT: $FREE_WORLD_PORT"
echo "TM_PORT: $TM_PORT"

bash /home/carla/CarlaUE4.sh --world-port=$FREE_WORLD_PORT -RenderOffScreen -nosound -graphicsadapter=0 -carla-streaming-port=$FREE_STREAMING_PORT &

sleep 30

{run_command}    
"""

    with open(job_file, "w", encoding="utf-8") as f:
        f.write(qsub_template)
    return job_file


def make_jobsub_file(route_number, data_save_root, code_root, port, partition, dir_carla_img, carla_img_name):
    qsub_template = f"""#!/bin/bash
#SBATCH --job-name={route_number}
#SBATCH --partition={partition}
#SBATCH -o {code_root}/{data_save_root}/logs/qsub_out_{route_number}.log
#SBATCH -e {code_root}/{data_save_root}/logs/qsub_err_{route_number}.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60gb
#SBATCH --time=0-20:00
#SBATCH --gres=gpu:1
# -------------------------------

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "Job started: $dt"

echo "Current branch:"
git branch
echo "Current commit:"
git log -1
echo "Current hash:"
git rev-parse HEAD

cp -R {dir_carla_img} /scratch/$SLURM_JOB_ID/

echo "Copied singularity image"

export FREE_STREAMING_PORT=0 # this enables CARLA to select a free unused streaming port
export FREE_WORLD_PORT={port}
export TM_PORT={port+1}

pkill singularity
sleep 2

echo "start python"
pwd
singularity exec --nv -H {code_root} /scratch/$SLURM_JOB_ID/{carla_img_name} bash {data_save_root}/start_files/{route_number}.sh $FREE_STREAMING_PORT $FREE_WORLD_PORT $TM_PORT
pkill singularity
"""

    with open(f"{data_save_root}/jobsub_file/{route_number}.sh", "w", encoding="utf-8") as f:
        f.write(qsub_template)


if __name__ == "__main__":
    code_dir = "/path/to/PDM-Lite"
    carla_image_name = "carla_img.sif"
    dir_carla_image = f"/path/to/{carla_image_name}"
    cluster_partition = "cluster-partition"
    dataset_name = "dataset_name"
    routes_directory = "path/to/routes/directory"

    import random
    random.seed(42)
    seed_counter = -1

    n_rounds = 1 # how often each route gets evaluated
    port_counter = 1998 # the first port used is 2000
    for round_idx in range(n_rounds):
        data_folder = routes_directory
        data_save_dir = f"database/{dataset_name}_round_{round_idx}"

        Path(data_save_dir).mkdir(exist_ok=True, parents=True)
        Path(os.path.join(data_save_dir, "data")).mkdir(exist_ok=True)
        Path(os.path.join(data_save_dir, "results")).mkdir(exist_ok=True)
        Path(os.path.join(data_save_dir, "start_files")).mkdir(exist_ok=True)
        Path(os.path.join(data_save_dir, "logs")).mkdir(exist_ok=True)
        Path(os.path.join(data_save_dir, "jobsub_file")).mkdir(exist_ok=True)

        # find all .xml files in route_folder
        routes = []
        for root, dirs, files in os.walk(data_folder):
            for file in files:
                if file.endswith(".xml"):
                    routes.append(os.path.join(root, file))

        random.shuffle(routes)

        num_routes = len(routes)

        for i, route_full_pth in enumerate(routes):
            port_counter += 2
            seed_counter += 1
            route_nr = route_full_pth.split("/")[-1].split("_")[0]

            scenario, route_name = route_full_pth.split("/")[-2:]
            route_name = route_name.replace(".xml", "")

            ckeckpoint_endpoint_root = f"{data_save_dir}/results/{scenario}"
            ckeckpoint_endpoint = f"{data_save_dir}/results/{scenario}/{route_name}.json"
            save_pth = f"{data_save_dir}/data/{scenario}"

            make_jobsub_file(route_nr, data_save_dir, code_dir, port_counter, cluster_partition,
                            dir_carla_image, carla_image_name)
            make_bash(data_save_dir, code_dir, route_nr, route_full_pth, ckeckpoint_endpoint, save_pth,
                            seed_counter, ckeckpoint_endpoint_root)

            subprocess.run(f"sbatch {data_save_dir}/jobsub_file/{route_nr}.sh > /dev/null 2>&1",
                           shell=True, check=False)
