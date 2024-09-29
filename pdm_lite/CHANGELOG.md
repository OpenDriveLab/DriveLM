## Leaderboard changes

agent_wrapper_local.py
* Introduced variable to store env. var. DATAGEN
* Added camera.depth and camera.semantic_segmentation for datagen
* Adds a list of used sensor names and ids and passes it to the agent instance during initialization and deletes it afterwards
* Applies sensor attributes also to camera.depth and camera.semantic_segmentation not only for camera.rgb
* Allows different rotation_frequency and points_per_second attributes during DATAGEN

autonomous_agent_local.py
* Added the SENSOR_QUALIFIER and the MAP_QUALIFIER for the CARLA Autonomous Driving Challenge 2024
* Accepts the carla_host and carla_port of the carla client and whether to debug in the instructor
* Passes the list of sensors to the run_step method
* Sets the orignal route during set_global_plan to avoid bugs when executing downsample_route afterwards
  * self.org_dense_route_gps = global_plan_gps
  * self.org_dense_route_world_coord = global_plan_world_coord

leaderboard_evaluator_local.py
* prints "start leaderboard_evaluator_local.py", when executing the file
* Added semantic_segmentation and depth camera for datagen to available sensors
* Changed cleaup, to fix bug
  * self.agent_instance.destroy(results) instead of self.agent_instance.destroy()
  * del self.agent_instance instead of self.agent_instance = None
* Added a method to find a free port
* Sets the traffic_manager_port to a free port instead of args.traffic_manager_port and saves it as object attribute
* Only loads the Town if it's not already loaded to speed up start
* Sets CarlaDataProvider.seed, numpy, random and torch seed to args.traffic_manager_seed
* Passes a string with the date and time and the traffic manager object to the agent instance during setup
* Creates the directories of args.checkpoint (result file)
* Added method main_eval, to use the original settings for evaluation

scenario_manager_local.py
* Saves the thread as object attribute, which builds the scenario_loop to wait later that it correctly finishes (# Make sure the scenario thread finishes to avoid blocks)
* Sets the transform of the spectator to BEV

statistics_manager_local.py
* Gets a route_date_string when executing comppute_route_statistics
* Introduce a variable time_stamp and set it equals route_date_string
* Returns the route_record when executing compute_route_statistics


## Scenario runner changes

weather_sim.py:
* OSCWeatherBehavior.update, RouteWeatherBehavior.update: Turn off setting the weather in case of datagen (set env. var DATAGEN)

carla_data_provider.py:
* Added list active_scenarios. Each of the RouteObstacle scenarios save their relevant data for the Expert in this list
* Added method set_random_seed
  * sets seed of random (but not numpy etc.)
  * saves seed that was set.
  
atomic_criteria.py:
* 85 percentage completion during data collection (because small routes need a smaller percentage)
* Clipped stop_extent to 0.5 # because some stop signs are very small (< 2cm) and hence are not detected reliable
  * stop_extent.x = max(0.5, stop_extent.x)
  * stop_extent.y = max(0.5, stop_extent.y)
* Added xyz-location to .json result file where YieldToEmergencyVehicleTest or ScenarioTimeoutTest failed

construction_crash_vehicle.py
* add actors that are relevant for the Expert to CarlaDataProvider.active_scenarios

invading_turn.py
* add actors that are relevant for the Expert to CarlaDataProvider.active_scenarios

route_obstacles.py
* add actors that are relevant for the Expert to CarlaDataProvider.active_scenarios for scenarios Accident, ParkedObstacle, HazardAtSideLane

route_scenario.py:
* add global option to turn off traffic 'DEACTIVATE_TRAFFIC', which controls whether to use background traffic.

vehicle_opens_door.py
* add actors that are relevant for the Expert to CarlaDataProvider.active_scenarios

yield_to_emergency_vehicle.py
* add actors that are relevant for the Expert to CarlaDataProvider.active_scenarios