[x] Problem description
Previous work (including what you used for your method i.e. pretrained models)
Your approach
Datasets
Results
Discussion
What problems did you encounter?
Are there next steps you would take if you kept working on the project?
How does your approach differ from others? Was that beneficial?


Noted that I am ending up finishing a project that is different from what I submit in the project propasal. The following are the overall description of my project. 

# Problem Description
- The problem statement is sovling a navigation task that given a video of the observation of the trajectory, the robot will be able to navigate from its current position to the goal position.

- As the figure 1 shown below, the red point is the current position of the mobile robot and the blue point is the goal position. The green curve is a series of images in the video recorded from the camera of the mobile robot or other agents.
<div align="center">
<img width="408" alt="image" src="https://github.com/YuquanDeng/navigation-model/assets/88260334/4c2197b4-213c-4470-b3b2-d1266533940e">
</div>
<div align="center">
<em>figure 1: Image Navigation Task</em>
</div>

# Previous work
- some previous work including Behavior Cloning. That is, given a demonstration, use a pretrained model to extract the feature, and then train with 3~4 FC layers, finally output the command. This doesn't work well since it needs.
- The pretrained model that I used as a CNN encoder is ResNet50.
- The simulation environment that I used is [Habitat Simulation](https://aihabitat.org/).

# My Approach
What I try is similar to Siamese network. I feed both current observation image and the goal image into pretrained ResNet50, which extract the features, and then concatenate the features of both, feed it back to 4 FC Layers. 

<div align="center">
<img width="608" alt="image" src="https://github.com/YuquanDeng/navigation-model/assets/88260334/bce3f00f-ae00-4a3b-895c-f3d62b00fc1a">
</div>

# Datasets
The Dataset I used is a pair of image and the label (velocity, theta) record from A1 legged robot in off-road environment(around 1 hour) collected by [Yuxiang Yang](https://yxyang.github.io/). 

# Results
I use Adam as optimizer with learning rate 0.001 and batch size 512 to train the model in 80 epochs. The criterion is minimizing MSE Loss between the predicting velocity and theta and the actual velocity and theta.

The following figure is the graph of both train loss and validation loss. We can see that the loss is very low(under 0.05) after 80 epochs. 
<div align="center">
<img width="331" alt="image" src="https://github.com/YuquanDeng/navigation-model/assets/88260334/ab09099f-91b6-479e-8a2d-86b1ba7624b2">
</div>
<div align="center">
<em>Loss</em>
</div>

In order to make sure the model is learning and predicting reasonable velocity and theta. I also plot the true velocity/theta vs. predicted velocity/theta for both the training set and test set.

 Training Set                 | Test Set                  |
:-------------------------:|:-------------------------:|
<img width="400" alt="image" src="https://github.com/YuquanDeng/navigation-model/assets/88260334/0bc0f62c-5b39-4084-8ab6-e4a848618345">|   <img width="400" alt="image" src="https://github.com/YuquanDeng/navigation-model/assets/88260334/6cabc244-75c2-4e69-8295-7de1dda4a4e8">|


 Training Set              | Test Set                  |
:-------------------------:|:-------------------------:|
<img width="400" alt="image" src="https://github.com/YuquanDeng/navigation-model/assets/88260334/501e6712-024d-43d9-abed-023a398588a1">|   <img width="400" alt="image" src="https://github.com/YuquanDeng/navigation-model/assets/88260334/55b60bd2-0f42-4f14-af7f-50a50b5ca307">|

If the scatter points are tend to align as the line y = x, which means predicted velocity/ predicted theta = true velocity/ true theta, then that means the model is very well. Basically, those are another way to analyze the model to get more idea if the model is learning or not. As the figure shown above, both the predicted velocity vs. true velocity for training set and test set plot are tend to follow a line, while it is not perfectly align with y=x, the tend shows the model is predicting reasonable velocity(but not great). For the graph of predicted Theta vs. true Theta, it is not very scale to a line. Most of the data are within the range -0.25 and 0.25. The model gives lots of non-zero prediction for the ground truth theta should be zero. I will give more insight about why that is the case in the later section. 

In order to test its robustness of the model, I deploy the trained model into the Habitat Simulation environment mentioned in the Previous work section. That is a simulation environment that can run a agent in various dataset(Gibson, HM3D, etc). The environment I used is a castle like environment, which is unseen by agent before.

<div align="center">
<img width="458" alt="image" src="https://github.com/YuquanDeng/navigation-model/assets/88260334/8c060d74-2c94-41dc-82b3-2e0fef69ab1d">
</div>

As the figures shown above, the blue point is the start position, the red point is the goal position, the green trajectory is the reference trajectory, and the blue curve is the deomonstrated trajectory by the agent. The agent run by the model deviates from the reference trajectory completely. I will give more analysis why that is the case in the Discussion section.


# Discussion
### Why the trajectory is completely deivate to the left side relative to the reference trajectory?
<p> One observation when I run the model in the habitat simulation is I notice that the theta prediction of the model is always positive, which explain why the agent is always go left (left is the direction of positve heading angle) and never correct back to the reference trajectory since the predicted angle never output negative value.  Remember in the Result section we plot the figure predicted angle vs. true angle and the trend doesn't look good, so the model doesn't learn well in predicting the turning angle. Why that is the case brings us to the second point.</p>

### Why the performance of the model is bad?

#### (1) The dataset is insufficient in size and lacks satisfactory quality.
In order to give a more rigorous explanation, I plot both the steering angle and velocity with respect to the time. 

<div align="center">
<img width="400" alt="image" src="https://github.com/YuquanDeng/navigation-model/assets/88260334/0a20e127-5c45-437f-a8b2-66758eaf266c">
</div>
<div align="center">
<em> velocity vs. Time </em>
</div>

<div align="center">
<img width="400" alt="image" src="https://github.com/YuquanDeng/navigation-model/assets/88260334/60ff0b23-7e8b-44d1-a705-24a0a035db32">
</div>
<div align="center">
<em> Steering Angle vs. Time </em>
</div>


#### (2) Short Goal Image is crucial but challenging to choose.



# What problems did you encounter?
(1) bug in the MLP
(2) alway outputing the same action

# Are there next steps you would take if you kept working on the project?
(1) if want to generalize to more agent, should normalize velocity
(2) try predicting something else, sin, cos of theta, and the distance between observation and goal image
(4) should run it in a more realistic environment 
# How does your approach differ from others? Was that beneficial?
Concanate short goal image is definitely help


citation

change the github repo
record video





