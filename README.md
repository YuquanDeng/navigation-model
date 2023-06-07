Noted that I am ending up finishing a project that is different from what I submit in the project propasal. The following are the overall description of my project. 

# CodeBase Explanation
- The `conf` direcotry contains the configuration that used in training the model and running in the simulation environment.
- `conf_outputs` directory contains the checkpoint of the model as well as the experiment result in the `metric` directory.
- `infrastructure` directory contains all the utils code for training and evaluating models.
- `policies` directory contains the 4 FC layers architecture in `MLP_policy.py`.
- 'scripts' direcotry contains training model pipeline in `run_experiment.py` and evaluation pipeline including running in the simulation environment and logging result in `evaluate_policy.py`.
- `simulation` directory contains the simulation code for kick off the simulatione environment in `simulator.py`. Other files or directory under the `simulation` directory are either the data need for rendering the simulation or tutorial code for learning the simulation environment.
- `test` directory contains the testing code for making sure the components code in the pipeline is correct.
- Noted that Except the `pytorch_utils.py` is taken from UCB cs285 homework 1, I wrote all the rest of the code in the Code base.





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

As the figures shown above, the part 1 and part 2 are the name of the directory of dataset. If we relate the steering angle vs. time figure with the predicted steering angle vs. true steering angle, where the positive value of the steering angle means turning left and the negative value of the steering angle means turning right, we know that the frequency of the turning right is much less than that of turning left. That means there aren't sufficient and valid data in the dataset for training with predicting turning left command. Another conclusion we can make is that we can notice that most of turning angle command is within the range between -0.25 and 0.25, which means most of the time the legged robot only turn slightly and those values are very close to 0. That explains the pattern where the true steering angle is zero but the predicted angle is nonzero.


#### (2) Short Goal Image is crucial but challenging to choose.
When I deploy the model in the simulation environment, Another problem is hot to choose the next short goal image since the short goal image is coming from the prerecorded reference trajectory. The trained model doesn't contain any information about the distance between current observation and the short goal image, so I use the condition that if the output command is small enough(within a threshold), then I update the short goal image to the next one. This heuristics is hard to adapt in different environments especially for the case when the robot deviating from the trajectory and it is rarely to recover from that. That means I need a better mechanism to choose short goal image.


# What problems did you encounter?
<p>
The first problem I encounter is the result I got from the graph predicted steering angle vs. acutal steering angle isn't quite right. Specifically, the prediction is never output negative value steering angle. Eventually I realize there is a bug in my 4 FC layers implementation, I mistakenly apply ReLU in the output, which makes the model only outputs positive value. 
 </p>
 
 <p>
 The second problem I had is after I deploy the model in the simulation environment, I realized the model will only output the same actions, which is very strange. It is also painful to debug since the pipeline is very complex and it is hard to tell which part went wrong. I spent almost two weeks only debugging this and finally figured out I use np.reshape for changing the dimension instead of tranpose, which mess up the dimensions of image feed in to the model and result in the same output actions. 
 </p>

# Are there next steps you would take if you kept working on the project?
By carrying on the points I elaborate in Discussion section, The next steps that I can do are the following:

#### Train the model in a more comprehensive and high quality dataset
<p>
 As we have already discussed in the Result Section and Discussion Section, we can conclude how important the quality of the dataset will affect the performance of model especially in evaluating its robustness. I should train the model with more data. For example, I can try to train the model with GO-Standard dataset. 
 </p>

#### Modify the structure of the model.
<p>
I would like to try predicting different output instead. For example, insteading of predicting velocity and theta, I want the model also learn the distance between current observation and the goal image. That is, predicting velocity, theta, distance. Also, another choice can be predicting sin(theta) and cos(theta) instead of theta directly and comparing the performance between the models with different outputs. For better generalize to other mobile robot, I should also try training the model with normalized velocity.
</p>

#### Use more suitable Simulation Environment
<p>
 The model is too pre-mature to run in a unseen environment and deploy it into another robot. I should choose a better simulation environment to measure the performance of model and makes the evaluation more statistically meaningful.
 </p>


# How does your approach differ from others? Was that beneficial?
Comparing with Behavior Cloning method, goal-conditioned model is very helpful for learn the pattern betwen the current observation and the goal image and output a more reasonable command. Even though the dataset is very small for training the model, it shows some sign the model is working for generalizing to more environments since the output actions is literally close to 0 when the current observation and the goal image share lots of features. 

# Video
Checkout the link: https://drive.google.com/file/d/1HJX3EfSWIU7owJA6ilkUkRK_OODA6H5v/view?usp=drive_link





