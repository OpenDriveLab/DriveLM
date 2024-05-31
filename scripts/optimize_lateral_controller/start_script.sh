#!/bin/bash

first_run=true

while true; do
    if [ "$first_run" = true ]; then
        echo "Starting script normally."
        python3 main.py
    else
        echo "Restarting script with --load option."
        python3 main.py --load
    fi

    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "Script exited successfully."
        break
    else
        echo "Script exited with error code $exit_code. Marking as a restart."
        first_run=false
        sleep 5  # Optional: Add a delay before restarting (adjust as needed)
    fi
done
