## Features of the DriveLM-Data <a name="features"></a>

- 🛣 Completeness in functionality (covering **Perception**, **Prediction**, and **Planning** QA pairs).


<p align="center">
  <img src="../assets/images/repo/point_1.png">
</p>


- 🔜 Reasoning for future events that have not yet happened.
  - Many **"What If"**-style questions: imagine the future by language.
 

<p align="center">
  <img src="../assets/images/repo/point_2.png" width=70%>
</p>

- ♻ Task-driven decomposition.
  - **One** scene-level description into **many** frame-level trajectories & planning QA pairs.

<p align="center">
  <img src="../assets/images/repo/point_3.png">
</p>

## How about the annotation process? <a name="annotation"></a>

The annotation process is different for DriveLM-nuScenes and DriveLM-CARLA.

<p align="center">
  <img src="../assets/images/repo/paper_data.jpg">
</p>

**For DriveLM-nuScenes**, we divide the annotation process into three steps:

1️⃣ Keyframe selection. Given all frames in one clip, the annotator selects the keyframes that need annotation. The criterion is that those frames should involve changes in ego-vehicle movement status (lane changes, sudden stops, start after a stop, etc.).

2️⃣ Key objects selection. Given keyframes, the annotator needs to pick up key objects in the six surrounding images. The criterion is that those objects should be able to affect the action of the ego vehicle (traffic signals, pedestrians crossing the road, other vehicles that move in the direction of the ego vehicle, etc.).

3️⃣ Question and answer annotation. Given those key objects, we automatically generate questions regarding single or multiple objects about perception, prediction, and planning. More details can be found in our data.

**For DriveLM-CARLA**, we employ an automated annotation approach:

We collect data using CARLA 0.9.14 in the Leaderboard 2.0 framework with a privileged rule-based expert. We set up a series of routes in urban, residential, and rural areas and execute the expert on these routes. During this process, we collect the necessary sensor data, generate relevant QAs based on privileged information about objects and the scene, and organize the logical relationships to connect this series of QAs into a graph.
