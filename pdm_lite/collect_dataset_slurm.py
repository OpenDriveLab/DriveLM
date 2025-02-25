"""
Generates a dataset for training on a SLURM cluster.
Each route file is parallelized on its own machine.
Monitors the data collection and continues crashed processes.
Best run inside a tmux terminal.
"""

from datetime import datetime
import os
import subprocess
import time
import glob
import json
from pathlib import Path
import random
import re
import xml.etree.ElementTree as ET


def make_bash(
    data_save_root,
    code_dir,
    route_file_number,
    agent_name,
    route_file,
    ckeckpoint_endpoint,
    save_pth,
    seed,
    carla_root,
    town,
    repetition,
):
    jobfile = f"{data_save_root}/start_files/{route_file_number}_Rep{repetition}.sh"
    # create folder
    Path(jobfile).parent.mkdir(parents=True, exist_ok=True)
    run_command = "python leaderboard/leaderboard/leaderboard_evaluator_local.py --port=${FREE_WORLD_PORT} \
        --traffic-manager-port=${TM_PORT} --traffic-manager-seed=${TM_SEED} --routes=${ROUTES} --repetitions=${REPETITIONS} \
            --track=${CHALLENGE_TRACK_CODENAME} --checkpoint=${CHECKPOINT_ENDPOINT} --agent=${TEAM_AGENT} \
                --agent-config=${TEAM_CONFIG} --debug=0 --resume=${RESUME} --timeout=600"

    qsub_template = f"""#!/bin/bash
export SCENARIO_RUNNER_ROOT={code_dir}/scenario_runner_autopilot
export LEADERBOARD_ROOT={code_dir}/leaderboard_autopilot

# carla
export CARLA_ROOT={carla_root}
export CARLA_SERVER={carla_root}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:{carla_root}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:leaderboard_autopilot
export PYTHONPATH=$PYTHONPATH:scenario_runner_autopilot
export REPETITIONS=1
export DEBUG_CHALLENGE=0
export TEAM_AGENT={agent_name}
export CHALLENGE_TRACK_CODENAME=MAP
export ROUTES={route_file}
export TOWN={town}
export REPETITION={repetition}
export TM_SEED={seed}

export CHECKPOINT_ENDPOINT={ckeckpoint_endpoint}
export TEAM_CONFIG={route_file}
export RESUME=1
export DATAGEN=1
export SAVE_PATH={save_pth}

echo "Start python"

export FREE_STREAMING_PORT=$1
export FREE_WORLD_PORT=$2
export TM_PORT=$3

echo "FREE_STREAMING_PORT: $FREE_STREAMING_PORT"
echo "FREE_WORLD_PORT: $FREE_WORLD_PORT"
echo "TM_PORT: $TM_PORT"

bash {carla_root}/CarlaUE4.sh --world-port=$FREE_WORLD_PORT -RenderOffScreen -nosound -graphicsadapter=0 -carla-streaming-port=$FREE_STREAMING_PORT &

sleep 60

{run_command}    
"""

    with open(jobfile, "w", encoding="utf-8") as f:
        f.write(qsub_template)
    return jobfile


def get_running_jobs(jobname, user_name):
    job_list = (
        subprocess.check_output(
            (
                f"SQUEUE_FORMAT2='jobid:10,username:{len(username)},name:130' squeue --sort V | grep {user_name} | \
                    grep {jobname} || true"
            ),
            shell=True,
        )
        .decode("utf-8")
        .splitlines()
    )
    currently_num_running_jobs = len(job_list)
    #  line is sth like "4767364   gwb791 eval_julian_4170_0   "
    routefile_number_list = [
        line.split("_")[-2] + "_" + line.split("_")[-1].strip() for line in job_list
    ]
    pid_list = [line.split(" ")[0] for line in job_list]
    return currently_num_running_jobs, routefile_number_list, pid_list


def get_last_line_from_file(filepath):  # this is used to check log files for errors
    try:
        with open(filepath, "rb", encoding="utf-8") as f:
            try:
                f.seek(-2, os.SEEK_END)
                while f.read(1) != b"\n":
                    f.seek(-2, os.SEEK_CUR)
            except OSError:
                f.seek(0)
            last_line = f.readline().decode()
    except:
        last_line = ""
    return last_line


def cancel_jobs_with_err_in_log(logroot, jobname, user_name):
    # check if the log file contains certain error messages, then terminate the job
    print("Checking logs for errors...")
    _, routefile_number_list, pid_list = get_running_jobs(jobname, user_name)
    for i, rf_num in enumerate(routefile_number_list):
        logfile_path = os.path.join(logroot, f"run_files/logs/qsub_out{rf_num}.log")
        last_line = get_last_line_from_file(logfile_path)
        terminate = False
        if "Actor" in last_line and "not found!" in last_line:
            terminate = True
        if "Watchdog exception - Timeout" in last_line:
            terminate = True
        if "Engine crash handling finished; re-raising signal 11" in last_line:
            terminate = True
        if terminate:
            print(
                f"Terminating route {rf_num} with pid {pid_list[i]} due to error in logfile."
            )
            subprocess.check_output(f"scancel {pid_list[i]}", shell=True)


def wait_for_jobs_to_finish(logroot, jobname, user_name, max_n_parallel_jobs):
    currently_running_jobs, _, _ = get_running_jobs(jobname, user_name)
    print(f"{currently_running_jobs}/{max_n_parallel_jobs} jobs are running...")
    counter = 0
    while currently_running_jobs >= max_n_parallel_jobs:
        if counter == 0:
            cancel_jobs_with_err_in_log(logroot, jobname, user_name)
        time.sleep(5)
        currently_running_jobs, _, _ = get_running_jobs(jobname, user_name)
        counter = (counter + 1) % 4


def get_num_jobs(job_name, username):
    len_usrn = len(username)
    num_running_jobs = int(
        subprocess.check_output(
            f"SQUEUE_FORMAT2='username:{len_usrn},name:130' squeue --sort V | grep {username} | grep {job_name} | wc -l",
            shell=True,
        )
        .decode("utf-8")
        .replace("\n", "")
    )
    with open("max_num_jobs.txt", "r", encoding="utf-8") as f:
        max_num_parallel_jobs = int(f.read())

    return num_running_jobs, max_num_parallel_jobs


def make_jobsub_file(
    data_save_root,
    jobname,
    route_file_number,
    partition_name,
    repetition,
    timeout="0-02:00",
):
    os.makedirs(f"{data_save_root}/slurm/run_files/logs", exist_ok=True)
    os.makedirs(f"{data_save_root}/slurm/run_files/job_files", exist_ok=True)
    jobfile = f"{data_save_root}/slurm/run_files/job_files/{route_file_number}_Rep{repetition}.sh"
    qsub_template = f"""#!/bin/bash
#SBATCH --job-name={jobname}_{route_file_number}
#SBATCH --partition={partition_name}
#SBATCH -o {data_save_root}/slurm/run_files/logs/qsub_out{route_file_number}.log
#SBATCH -e {data_save_root}/slurm/run_files/logs/qsub_err{route_file_number}.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40gb
#SBATCH --time={timeout}
#SBATCH --gres=gpu:1
# -------------------------------

eval "$(conda shell.bash hook)"
conda activate pdm_lite

echo "SLURMD_NODENAME: $SLURMD_NODENAME"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
scontrol show job $SLURM_JOB_ID

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "Job started: $dt"

echo "Current branch:"
git branch
echo "Current commit:"
git log -1
echo "Current hash:"
git rev-parse HEAD


export FREE_STREAMING_PORT=`comm -23 <(seq 10000 10400 | sort) <(ss -Htan | awk \'{{print $4}}\' | cut -d\':\' -f2 | sort -u) | shuf | head -n 1`
export FREE_WORLD_PORT=`comm -23 <(seq 20000 20400 | sort) <(ss -Htan | awk \'{{print $4}}\' | cut -d\':\' -f2 | sort -u) | shuf | head -n 1`
export TM_PORT=`comm -23 <(seq 30000 30400 | sort) <(ss -Htan | awk '{{print $4}}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`

sleep 2

echo "start python"
pwd
bash {data_save_root}/start_files/{route_file_number}_Rep{repetition}.sh $FREE_STREAMING_PORT $FREE_WORLD_PORT $TM_PORT
"""
    with open(jobfile, "w", encoding="utf-8") as f:
        f.write(qsub_template)
    return jobfile


if __name__ == "__main__":
    repetitions = 1
    repetition_start = 0
    partition = "2080-galvani"
    job_name = "collect"
    username = "gwb438"
    code_root = r"/mnt/lustre/work/geiger/gwb438/hiwi/DriveLM/pdm_lite"
    carla_root = "/mnt/lustre/work/geiger/gwb438/hiwi/test_DriveLM/carla"
    date = datetime.today().strftime("%Y_%m_%d")
    dataset_name = "pdm_lite_carla_lb2_" + date
    root_folder = r"results/data/"  # With ending slash
    data_save_directory = root_folder + dataset_name
    log_root = f"{data_save_directory}/slurm"

    route_folder = (
        "/mnt/lustre/work/geiger/gwb438/hiwi/DriveLM/pdm_lite/data/50x36_town_13"
    )

    # find all .xml files in route_folder
    routes = glob.glob(f"{route_folder}/**/*.xml", recursive=True)

    routes = routes[:3]

    port_offset = 0
    job_number = 1
    meta_jobs = {}

    # shuffle routes
    random.seed(42)
    random.shuffle(routes)
    seed_counter = (
        1000000 * repetition_start - 1
    )  # for the traffic manager, which is incremented so that we get different traffic each time

    num_routes = len(routes)
    for repetition in range(repetition_start, repetitions):
        for route in routes:
            with open(route, "r") as f:
                town = ET.fromstring(f.read()).find("route").attrib["town"]

            seed_counter += 1

            scenario_type = route.split("/")[-2]
            routefile_number = route.split("/")[-1].split(".")[
                0
            ]  # this is the number in the xml file name, e.g. 22_0.xml
            ckpt_endpoint = f"{code_root}/{data_save_directory}/results/{scenario_type}/{routefile_number}_result.json"

            save_path = f"{code_root}/{data_save_directory}/data/{scenario_type}"
            Path(save_path).mkdir(parents=True, exist_ok=True)
            agent = f"{code_root}/team_code/data_agent.py"

            bash_file = make_bash(
                data_save_directory,
                code_root,
                routefile_number,
                agent,
                route,
                ckpt_endpoint,
                save_path,
                seed_counter,
                carla_root,
                town,
                repetition,
            )
            job_file = make_jobsub_file(
                data_save_directory,
                job_name,
                routefile_number,
                partition,
                repetition,
                "0-04:00",
            )

            # Wait until submitting new jobs that the #jobs are at below max
            num_running_jobs, max_num_parallel_jobs = get_num_jobs(
                job_name=job_name, username=username
            )
            print(f"{num_running_jobs}/{max_num_parallel_jobs} jobs are running...")
            while num_running_jobs >= max_num_parallel_jobs:
                num_running_jobs, max_num_parallel_jobs = get_num_jobs(
                    job_name=job_name, username=username
                )
                time.sleep(0.05)

            print(
                f"Submitting job {job_number}/{num_routes}: {job_name}_{routefile_number}. ",
                end="",
            )
            time.sleep(1)
            jobid = (
                subprocess.check_output(f"sbatch {job_file}", shell=True)
                .decode("utf-8")
                .strip()
                .rsplit(" ", maxsplit=1)[-1]
            )
            print(f"Jobid: {jobid}")
            meta_jobs[jobid] = (
                False,
                job_file,
                ckpt_endpoint,
                0,
            )  # job_finished, job_file, result_file, resubmitted
            job_number += 1

    time.sleep(1)
    training_finished = False
    while not training_finished:
        num_running_jobs, _, _ = get_running_jobs(job_name, username)
        print(f"{num_running_jobs} jobs are running... Job: {job_name}")
        cancel_jobs_with_err_in_log(log_root, job_name, username)
        time.sleep(20)

        # resubmit unfinished jobs
        for k in list(meta_jobs.keys()):
            job_finished, job_file, result_file, resubmitted = meta_jobs[k]
            need_to_resubmit = False
            if not job_finished and resubmitted < 3:
                # check whether job is running
                if (
                    int(
                        subprocess.check_output(
                            f"squeue | grep {k} | wc -l", shell=True
                        )
                        .decode("utf-8")
                        .strip()
                    )
                    == 0
                ):
                    # check whether result file is finished?
                    if os.path.exists(result_file):
                        with open(result_file, "r", encoding="utf-8") as f_result:
                            evaluation_data = json.load(f_result)
                        progress = evaluation_data["_checkpoint"]["progress"]
                        if len(progress) < 2 or progress[0] < progress[1]:
                            need_to_resubmit = True
                        else:
                            for record in evaluation_data["_checkpoint"]["records"]:
                                if record["scores"]["score_route"] <= 0.00000000001:
                                    need_to_resubmit = True
                                if (
                                    record["status"]
                                    == "Failed - Agent couldn't be set up"
                                ):
                                    need_to_resubmit = True
                                if record["status"] == "Failed":
                                    need_to_resubmit = True
                                if record["status"] == "Failed - Simulation crashed":
                                    need_to_resubmit = True
                                if record["status"] == "Failed - Agent crashed":
                                    need_to_resubmit = True

                        if not need_to_resubmit:
                            # delete old job
                            print(f"Finished job {job_file}")
                            meta_jobs[k] = (True, None, None, 0)

                    else:
                        need_to_resubmit = True

            if need_to_resubmit:
                # rename old error files to still access it
                routefile_number = Path(job_file).stem
                print(
                    f"Resubmit job {routefile_number} (previous id: {k}). Waiting for jobs to finish..."
                )

                with open("max_num_jobs.txt", "r", encoding="utf-8") as f:
                    max_num_parallel_jobs = int(f.read())
                wait_for_jobs_to_finish(
                    log_root, job_name, username, max_num_parallel_jobs
                )

                time_now_log = time.time()
                os.system(
                    f'mkdir -p "{log_root}/run_files/logs_{routefile_number}_{time_now_log}"'
                )
                os.system(
                    f"cp {log_root}/run_files/logs/qsub_err{routefile_number}.log {log_root}/ \
                          run_files/logs_{routefile_number}_{time_now_log}"
                )
                os.system(
                    f"cp {log_root}/run_files/logs/qsub_out{routefile_number}.log {log_root}/ \
                          run_files/logs_{routefile_number}_{time_now_log}"
                )

                jobid = (
                    subprocess.check_output(f"sbatch {job_file}", shell=True)
                    .decode("utf-8")
                    .strip()
                    .rsplit(" ", maxsplit=1)[-1]
                )
                meta_jobs[jobid] = (False, job_file, result_file, resubmitted + 1)
                meta_jobs[k] = (True, None, None, 0)
                print(f"resubmitted job {routefile_number}. (new id: {jobid})")

        time.sleep(10)

        if num_running_jobs == 0:
            training_finished = True
