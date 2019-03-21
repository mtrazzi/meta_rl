<p align="center">
  <img src="/img/meta_rl_glitch.gif" alt="Harlow Task">
</p>

⚠**DISCLAIMER**⚠
This is the git submodule for the Harlow task for my article [Meta-Reinforcement Learning](https://blog.floydhub.com/author/michaeltrazzi/) on FloydHub.
- For the main repository for the Harlow task (with more information about the task) see [here](https://github.com/mtrazzi/harlow).
- For the two-step task see [here](https://github.com/mtrazzi/two-step-task).

To get started, check out the parent [`README.md`](https://github.com/mtrazzi/harlow#getting-started).

# Directory structure

``` bash
meta-rl
├── harlow.py                 # main file that implements the DeepMind Lab wrapper, processes the frames and run the trainings, initializing a DeepMind Lab environment.
└── meta_rl
    ├── worker.py             # implements the class `Worker`, that contains the method `work` to collect training data and `train` to train the networks on this training data.
    └── ac_network.py         # implements the class `AC_Network`, where we initialize all the networks & the loss function.
```

# Branches

- [`master`](https://github.com/mtrazzi/meta_rl): for this branch, the frames are pre-processed, on a dataset of 42 pictures of students from 42 (cf. the FloydHub blog for more details). Our model achieved 40% performance on this simplified version of the Harlow task.
<p align="center">
  <img src="/img/reward_cuve_5_seeds_42_images.png" alt="img/reward_cuve_5_seeds_42_images.png">
</p>

- [`dev`](https://github.com/mtrazzi/meta_rl/tree/dev): for this branch, we implemented a stacked LSTM + a convolutional network, to have exactly the same setup as in [Wang et al, 2018 Nature Neuroscience](https://www.nature.com/articles/s41593-018-0147-8). Here is the reward curve we obtained:

<p align="center">
  <img src="/img/conv_plus_stacked_lstm.png" alt="img/conv_plus_stacked_lstm.png">
</p>

- [`monothread2pixel`](https://github.com/mtrazzi/meta_rl/tree/monothread2pixel): here, we used for our dataset only a black image and a white image. We pre-processed those two images so our agent only sees a one-hot, that is either [0,1] or [1,0]. Here is the resulting reward curve after training:
<p align="center">
  <img src="/img/monothread2pixels.png" alt="img/monothread2pixels.png">
</p>

- [`multiprocessing`](https://github.com/mtrazzi/meta_rl/tree/multiprocessing): I implemented multiprocessing using Python’s library multiprocessing. However, it appeared that Tensorflow doesn’t allow to use multiprocessing after having imported tensorflow, so that multiprocessing branch came to a dead end.

- [`ray`](https://github.com/mtrazzi/meta_rl/tree/ray): we also tried multiprocessing with ray  another multiprocessing library. However, it didn’t work out because DeepMind was not pickable, i.e. it couldn’t be serialized using pickle.

# Todo

On branch `master`:
- [ ] train with more episodes (for instance 20-50k) to see if some seeds keep learning.
- [ ] train with different seeds, to see if some seeds can reach > 40% performance.
- [ ] train with more units in the LSTM (for instance > 100 instead of 48), to see if it can keep learning after 10k episodes.
- [ ] train with more images (for instance 1000).

For multi-threading (e.g. in `dev`):
- [ ] support for [distributed tensorflow](https://www.tensorflow.org/guide/distribute_strategy) on multiple GPUs.
- [ ] get rid of CPython's global interpreter lock by connecting Tensorflow's C API with DeepMind Lab C API.

For multiprocessing:
- [ ] in [`multiprocessing`](https://github.com/mtrazzi/meta_rl/tree/multiprocessing) branch, try to import tensorflow _after_ the multiprocessing calls.
- [ ] in [`ray`](https://github.com/mtrazzi/meta_rl/tree/ray), try to make the DeepMind Lab environment pickable (for instance by looking at how OpenAI made their physics engine [mujoco-py](https://github.com/openai/mujoco-py) pickable.


# Support

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

