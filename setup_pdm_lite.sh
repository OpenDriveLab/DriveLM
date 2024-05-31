#!/usr/bin/env bash

# Download and install CARLA
mkdir carla
cd carla
wget https://leaderboard-public-contents.s3.us-west-2.amazonaws.com/CARLA_Leaderboard_2.0.tar.xz
tar -xf CARLA_Leaderboard_2.0.tar.xz
rm CARLA_Leaderboard_2.0.tar.xz
cd ..