sh /home/jens/Desktop/carla/CARLA_Leaderboard_20/CarlaUE4.sh -RenderOffScreen -carla-streaming-port=0 -carla-rpc-port=2000 &
sleep 6

python3 main_split_scenarios.py --path-in '../leaderboard/data/routes_training.xml' --save-path '../data/1_scenario_per_route_v5/training_1_scenario'
sleep 1
python3 main_split_scenarios.py --path-in '../leaderboard/data/routes_devtest.xml' --save-path '../data/1_scenario_per_route_v5/devtest_1_scenario'
sleep 1

python3 main_split_scenarios.py --path-in '../leaderboard/data/routes_training.xml' --save-path '../data/1_scenario_per_route_rm_unmanageable_v5/training_1_scenario' --exclude-scenarios Accident AccidentTwoWays ConstructionObstacle ConstructionObstacleTwoWays HazardAtSideLane HazardAtSideLaneTwoWays InvadingTurn ParkedObstacle ParkedObstacleTwoWays VehicleOpensDoorTwoWays YieldToEmergencyVehicle
sleep 1
python3 main_split_scenarios.py --path-in '../leaderboard/data/routes_devtest.xml' --save-path '../data/1_scenario_per_route_rm_unmanageable_v5/devtest_1_scenario' --exclude-scenarios Accident AccidentTwoWays ConstructionObstacle ConstructionObstacleTwoWays HazardAtSideLane HazardAtSideLaneTwoWays InvadingTurn ParkedObstacle ParkedObstacleTwoWays VehicleOpensDoorTwoWays YieldToEmergencyVehicle
sleep 1

python3 main_split_scenarios.py --path-in '../leaderboard/data/routes_training.xml' --save-path '../data/training_split_v5' --max-scenarios 1000
sleep 1
python3 main_split_scenarios.py --path-in '../leaderboard/data/routes_devtest.xml' --save-path '../data/devtest_split_v5' --max-scenarios 1000
sleep 1

python3 main_split_scenarios.py --path-in '../leaderboard/data/routes_training.xml' --save-path '../data/no_scenario_v5/training_no_scenario' --only-waypoints
sleep 1
python3 main_split_scenarios.py --path-in '../leaderboard/data/routes_devtest.xml' --save-path '../data/no_scenario_v5/devtest_no_scenario' --only-waypoints
sleep 1

python3 main_split_scenarios.py --path-in '../leaderboard/data/routes_validation.xml' --save-path '../data/no_scenario_v5/validation_no_scenario' --only-waypoints
sleep 1
python3 main_split_scenarios.py --path-in '../leaderboard/data/routes_validation.xml' --save-path '../data/1_scenario_per_route_rm_unmanageable_v5/validation_1_scenario' --exclude-scenarios Accident AccidentTwoWays ConstructionObstacle ConstructionObstacleTwoWays HazardAtSideLane HazardAtSideLaneTwoWays InvadingTurn ParkedObstacle ParkedObstacleTwoWays VehicleOpensDoorTwoWays YieldToEmergencyVehicle
sleep 1
python3 main_split_scenarios.py --path-in '../leaderboard/data/routes_validation.xml' --save-path '../data/1_scenario_per_route_v5/validation_1_scenario'
sleep 1
python3 main_split_scenarios.py --path-in '../leaderboard/data/routes_validation.xml' --save-path '../data/validation_split_v5' --max-scenarios 1000
sleep 1
python3 main_split_scenarios.py --path-in '../leaderboard/data/routes_validation.xml' --save-path '../data/validation_split_v5_filtered' --max-scenarios 1000 --exclude-scenarios Accident AccidentTwoWays ConstructionObstacle ConstructionObstacleTwoWays HazardAtSideLane HazardAtSideLaneTwoWays InvadingTurn ParkedObstacle ParkedObstacleTwoWays VehicleOpensDoorTwoWays YieldToEmergencyVehicle
