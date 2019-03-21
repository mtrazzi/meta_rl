# meta-RL

⚠**DISCLAIMER**⚠
This is the git submodule for the Harlow task for my article [Meta-Reinforcement Learning](https://blog.floydhub.com/author/michaeltrazzi/) on FloydHub.
- For the main repository for the Harlow task (with more information about the task) see [here](https://github.com/mtrazzi/harlow).
- For the two-step task see [here](https://github.com/mtrazzi/two-step-task).

## Getting Started

See instructions in the parent [`README.md`](https://github.com/mtrazzi/harlow#getting-started).

## Directory structure

``` bash
meta-rl
├── harlow.py                 # main file that implements the DeepMind Lab wrapper, processes the frames and run the trainings, initializing a DeepMind Lab environment.
└── meta_rl
    ├── worker.py             # implements the class `Worker`, that contains the method `work` to collect training data and `train` to train the networks on this training data.
    └── ac_network.py         # implements the class `AC_Network`, where we initialize all the networks & the loss function.
```

## Support

- We support Python3.6.

- The branch `master` was tested on FloydHub's instances (using `Tensorflow 1.12` and `CPU`). To change for `GPU`, change `tf.device("/cpu:0")` with `tf.device("/device:GPU:0")` in [`harlow.py`](https://github.com/mtrazzi/meta_rl/blob/master/harlow.py).

## Pip

All the pip packages should be either installed on FloydHub or installed with [`install.sh`](https://github.com/mtrazzi/harlow/blob/master/install.sh).

However, if you want to run this repository on your machine, here are the requirements:
```
numpy==1.16.2
tensorflow==1.12.0
six==1.12.0
scipy==1.2.1
skimage==0.0
setuptools==40.8.0
Pillow==5.4.1
```

Additionally, for the branch `ray` you might need to do (`pip install ray`) and for the branch `multiprocessing` you would need to install `multiprocessing` with (`pip install multiprocessing`).

# Credits

This work uses [awjuliani's Meta-RL implementation](https://github.com/awjuliani/Meta-RL).

I couldn't have done without my dear friend [Kevin Costa](https://github.com/kcosta42), and the additional details provided kindly by [Jane Wang](http://www.janexwang.com/).

