<div align="center">
<img src="/assets/logo.png" width=60%>
<br>
<h3>Embodied Multi-role Open-vocabulary Planning with Online Grounding and Execution</h3>
  
<a href="https://www.linkedin.com/in/fra-arg/">Francesco Argenziano</a><sup><span>1</span></sup>,
<a href="https://scholar.google.com/citations?user=sk3SpmUAAAAJ&hl=it&oi=ao/">Michele Brienza</a><sup><span>1</span></sup>,
<a href="https://scholar.google.com/citations?user=Y8LuLfoAAAAJ&hl=it&oi=ao">Vincenzo Suriani</a><sup><span>2</span></sup>,
<a href="https://scholar.google.com/citations?user=xZwripcAAAAJ&hl=it&oi=ao">Daniele Nardi</a><sup><span>1</span></sup>,
<a href="https://scholar.google.com/citations?user=_90LQXQAAAAJ&hl=it&oi=ao">Domenico D. Bloisi</a><sup><span>3</span></sup>
</br>

<sup>1</sup> Department of Computer, Control and Management Engineering, Sapienza University of Rome, Rome, Italy,
<sup>2</sup> School of Engineering, University of Basilicata, Potenza, Italy,
<sup>3</sup> International University of Rome UNINT, Rome, Italy

<div>

[![arxiv paper](https://img.shields.io/badge/Project-Website-blue)](https://sites.google.com/diag.uniroma1.it/empower/home)
[![arxiv paper](https://img.shields.io/badge/arXiv-TBA-red)](https://sites.google.com/diag.uniroma1.it/empower/home)
[![license](https://img.shields.io/badge/License-Apache_2.0-yellow)](LICENSE)

</div>

<img src="/assets/architecture.png" width=100%>

</div>

# Prerequisites
Our framework runs and has been tested on a TIAGo robot. The higher levels of the pipeline (up to the generation of the plan) should be executable in any ROS-based system with RGB-D cameras (camera topics name may be different). 
However, depending on the robot, low-level actions should be designed accordingly since different robot can have different capabilities (mobile vs fixed base, arms vs no-arm etc).
Therefore, the GPT prompts in the ```src/agents.py``` file and the low-level grounders in ```src/low_level_execution.py``` and ```src/primitive_actions.py``` should be adapted to the desired robotic platform.

## Model checkpoints
### YOLO-World
To download the YOLO-World model, visit the official [repository](https://huggingface.co/spaces/stevengrove/YOLO-World), download the .pth weights and put them in the ```config/yolow/``` directory.
### EfficientViT-SAM
To download the pre-trained weights and the encoder and decoder ONNX models follow the instruction on the official [repository](https://github.com/mit-han-lab/efficientvit).
After the download, put the models and weights in the ```config/efficientvitsam/``` directory.
For our tests, we used the ```l2``` model.

## Environment
Create the conda environment from the configuration file
```
conda env create -f environment.yml
```
then activate it with
```
conda activate empower
```

Also, set your OpenAI API key:
```
conda-env config vars set OPENAI_API_KEY=<YOUR API KEY>
```

# Usage
Clone this repo
```
https://github.com/Fra-Tsuna/vlm-grasping.git
```

Build the package with catkin
```
catkin build
```

Before starting, set the following ROS parameters:
```
rosparam set /use_case <name>
```
to name the task you want to solve, and if your robotic platform has speakers, you may wish to enable the speech:
```
rosparam set /speech True
```
otherwise set it to False.

Run:
```
rosrun vlm_grasping create_pcl.py
```
to create the pointcloud and the image of the desired scene.

Modify the file ```src/detection.py``` to include in the task dictionary the task you want to perform. The key should be the name of the task like you set in the rosparamater, while the value is a description in natural language of this task.

In two different terminals run:
```
python3 models_cacher.py <name> 
```
and
```
python3 execute_task.py
```
The models cacher loads the models in memory in such a way that is possible to run multiple time the ```execute_task``` script without the need to reload them, saving up execution time.
Eventually, ```execute_task``` will produce the plan and will dump it in the corresponding folder.

Now, run:
```
rosrun vlm_grasping color_pcl.py
```
to obtain the grounded pointcloud with the segmentation masks projected on it.

Then, run:
```
rosrun vlm_grasping spawn_marker_centroids.py
```
to ground the detections in the RViz scene and to keep it populated.

Finally, in another terminal run:
```
rosrun vlm_grasping low_level_execution.py 
```
to execute the actions needed to achieve the task.
