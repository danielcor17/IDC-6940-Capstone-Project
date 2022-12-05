Deep Reinforcement Learning Agent for <br /> Autonomous Hydrographic Surveys
=======

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

The objective of this project is to assess the effectiveness of deep reinforcement learning to train an agent
to perform a hydrographic survey within a simulated environment using Rainbow DQN. 
As well, to test whether transfer learning with pretrained models improves learning and performance. 
The overall objective can be divided into four parts:

1.	Design a simulation of the environment where an agent can perform the desired task
2.	Apply a DQN algorithm to operate and train an agent
3.	Assess the performance of the trained agent with the standard architecture 
4.	Compare the performance using instead a pretrained architecture 

The main.py file can be run with `python main.py` <br />
The following command line arguments are available:

--image-file - A path to load environment image file/s from <br />
--max-depth - The maximum depth of the synthetic bathymetry<br />
--sensor-range - Maximum range of the simulated sonar <br />
--sensor-angle - Maximum angle of the simulated sonar <br />
--auxiliary-plots - Whether to show plots with auxiliary information <br />
--id - Name of experiment file to be generated <br />
--seed - Random seed for the experiment <br />
--render - Whether to display the screen <br />
--disable-cuda - Whether to disable CUDA <br />
--enable-cudnn - Whether to enable cuDNN <br />
--T-max - Maximum number of steps during training <br />
--max-episode-length - Maximum number of steps per episode <br />
--learn-start - Number of steps before learning begins <br />
--evaluate - Whether to only evaluate agent <br />
--evaluation-interval - The number of steps before evaluation is triggered <br />
--evaluation-size - The number of transitions for evaluating Q <br />
--model - Pretrained model file <br />
--architecture - Type of Q-network architecture <br />
--hidden-size - Hidden Layer size <br />
--batch-size - Number of batches selected during training <br />
--learning-rate - Learning rate <br />
--adam-eps - Adam epsilon <br />
--norm-clip - Max L2 norm for gradient clipping <br />
--checkpoint-interval - How often to checkpoint <br />
--memory - File to save/load experience replay memory <br />
--replay-frequency - Frequency of sampling from memory <br />
--priority-exponent - PER replay exponent <br />
--priority-weight - Initial PER importance sampling weight <br />
--history-length - Number of consecutive states to process <br />
--target-update - Number of steps to update Q-target <br />
--multi-step - Number of steps for multi-step return <br />
--noisy-std - Initial standard deviation of noisy layers <br />
--atoms - Discretized size of value distribution <br />
--V-min - Minmimum of value distribution <br />
-V-max - Maximum of Value distribution <br />
--discount - Discount factor <br />
--reward-clip - Rewards clipping <br />


Requirements
------------

To install all dependencies with Anaconda run `conda env create -f capstone.yml`.

Acknowledgements
----------------

- [@Kaixhin](https://github.com/kaixhin) for [Rainbow implementation details](https://github.com/Kaixhin/Rainbow/wiki/Matteo's-Notes)

References
----------

[1] [Playing Atari with Deep Reinforcement Learning](http://arxiv.org/abs/1312.5602)  
[2] [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)
  
