# Deep RL for traffic signal control
This repo implements A2C for traffic signal control in SUMO-simulated environments.

Available cooperation levels:
* Centralized: a global agent that makes global control w/ global observation, reward.
* Decentralized: multiple local agents that make local control w/ local observation, global reward.
* Multi-agent: multiple local agents that make local control w/ neighboring observation, spatially discounted global reward.

Available environments:
* A 7-intersection benchmark network w/ designed traffic dynamics. [Ye, Bao-Lin, et al. "A hierarchical model predictive control approach for signal splits optimization in large-scale urban road networks." IEEE Transactions on Intelligent Transportation Systems 17.8 (2016): 2182-2192.](https://ieeexplore.ieee.org/abstract/document/7406703/)
* A 100-intersection large grid. [Chu, Tianshu, Shuhui Qu, and Jie Wang. "Large-scale traffic grid signal control with regional reinforcement learning." American Control Conference (ACC), 2016. IEEE, 2016.](https://ieeexplore.ieee.org/abstract/document/7525014/)

Available policies:
Fully-connected, or LSTM.

## Requirements
* Python3
* [Tensorflow](http://www.tensorflow.org/install)
* [SUMO](http://sumo.dlr.de/wiki/Installing)

## Usages
First define all hyperparameters in a config file under `[config_dir]`, and create the base directory of experiements `[base_dir]`. 

To train a new agent, run
~~~
python3 main.py --base-dir [base_dir] train --config-dir [config_dir] --test-mode no_test
~~~
`no_test` is suggested if no testing is needed during training, since it is time-consuming.

To evaluate and compare trained agents, run
~~~
python3 main.py --base-dir [base_dir] train --agents [agent names] --evaluate-metrics num_arrival_car --evaluate-seeds [seeds]
~~~
Evaluation metrics is the actual objective, and make sure evaluation seeds are different from those used in training/testing.

## Results
Training curves
