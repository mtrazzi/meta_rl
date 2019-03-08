# Copyright 2016 Google Inc.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import numpy as np
import six

import deepmind_lab

import os
import tensorflow as tf
from meta_rl.ac_network import AC_Network
from meta_rl.worker import Worker

from datetime import datetime

import threading
import multiprocessing

from skimage import data
from skimage.color import rgb2gray

DATASET_SIZE = 2

# Padding
WIDTH_PAD  = 18 # 19 if we dont want 1 pixel surrounding image
HEIGHT_PAD = 30 # 31 if we dont want 1 pixel surrounding image

MEAN_LEFT_IDX = -1
MEAN_LEFT = np.zeros(DATASET_SIZE)
MEAN_RIGHT_IDX = -1
MEAN_RIGHT = np.zeros(DATASET_SIZE)


def process_obs(obs, true_action=False):
    global MEAN_LEFT_IDX, MEAN_LEFT, MEAN_RIGHT_IDX, MEAN_RIGHT

    black_and_white = rgb2gray(obs)
    left, right = np.hsplit(black_and_white[HEIGHT_PAD:-HEIGHT_PAD, WIDTH_PAD:-WIDTH_PAD], 2)

    one_hot_left = np.zeros(DATASET_SIZE)
    one_hot_right = np.zeros(DATASET_SIZE)

    if not true_action:
      return np.stack([one_hot_left, one_hot_right])

    out = np.where(MEAN_LEFT == left.mean())[0]
    if out.size == 0:
      MEAN_LEFT_IDX += 1
      idx = MEAN_LEFT_IDX
      MEAN_LEFT[idx] = left.mean()
    else:
      idx = out[0]
    one_hot_left[idx] = 1
    out = np.where(MEAN_RIGHT == right.mean())[0]
    if out.size == 0:
      MEAN_RIGHT_IDX += 1
      idx = MEAN_RIGHT_IDX
      MEAN_RIGHT[idx] = right.mean()
    else:
      idx = out[0]
    one_hot_right[idx] = 1

    return np.stack([one_hot_left, one_hot_right])

class WrapperEnv(object):
  """A gym-like wrapper environment for DeepMind Lab.

  Attributes:
      env: The corresponding DeepMind Lab environment.
      length: Maximum number of frames

  Args:
      env (deepmind_lab.Lab): DeepMind Lab environment.

  """
  def __init__(self, env, length, seed):
    self.env = env
    self.length = length
    self.l = []
    self.seed = seed
    self.reset()

  def step(self, action, true_action=False):
    done = not self.env.is_running() or self.env.num_steps() > 3600
    if done:
      self.reset()

    # real step
    obs = self.env.observations()
    reward = self.env.step(action, num_steps=1)

    self.l.append(obs['RGB_INTERLEAVED'])

    if true_action:
      print("\033[34mAction Taken: " + ("Left" if action[0] > 0 else "Right") + "\033[0m")

    if reward > 0:
        print("\033[1;32mTrial reward: " + str(reward) + "\033[0m")
    elif reward < 0:
        print("\033[1;31mTrial reward: " + str(reward) + "\033[0m")

    return process_obs(obs['RGB_INTERLEAVED'], true_action), reward, done, self.env.num_steps()

  def reset(self):
    self.env.reset(seed=self.seed)
    obs = self.env.observations()

    d = np.array(self.l)
    if (len(d) > 2):
        with open("/floyd/home/obs.npy", "bw") as file:
            file.write(d.dumps())

        print("\033[32mModel's Log Saved\033[0m")
    self.l = []

    return process_obs(obs['RGB_INTERLEAVED'])

def run(length, width, height, fps, level, record, demo, demofiles, video):
  """Spins up an environment and runs the random agent."""
  config = {
      'fps': str(fps),
      'width': str(width),
      'height': str(height)
  }
  if record:
    config['record'] = record
  if demo:
    config['demo'] = demo
  if demofiles:
    config['demofiles'] = demofiles
  if video:
    config['video'] = video

  dir_name = "/floyd/home/python/meta_rl/train_" + datetime.now().strftime("%m%d-%H%M%S")

  # Hyperparameters for training/testing
  gamma = .9
  a_size = 2
  n_seeds = 1
  num_episode_train = 1e5
  num_episode_test = 50
  collect_seed_transition_probs = []
  for seed_nb in range(n_seeds):

    # initialize the directories' names to save the models for this particular seed
    model_path = dir_name+'/model_' + str(seed_nb)
    frame_path = dir_name+'/frames_' + str(seed_nb)
    plot_path = dir_name+'/plots_' + str(seed_nb)
    load_model_path = "meta_rl/results/biorxiv/final/model_" + str(seed_nb) + "/model-20000"

    # create the directories
    if not os.path.exists(model_path):
      os.makedirs(model_path)
    if not os.path.exists(frame_path):
      os.makedirs(frame_path)
    if not os.path.exists(plot_path):
      os.makedirs(plot_path)

    # in train don't load the model and set train=True
    # in test, load the model and set train=False
    for train, load_model, num_episodes in [[True, False, num_episode_train]]:

      print ("seed_nb is:", seed_nb)

      tf.reset_default_graph()

      with tf.device("/cpu:0"):
        global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
        trainer = tf.train.RMSPropOptimizer(learning_rate=1e-3)
        master_network = AC_Network(a_size, 'global', None) # Generate global network
        num_workers = 1
        workers = []
        # Create worker classes
        env_list = [deepmind_lab.Lab(level, ['RGB_INTERLEAVED'], config=config) for _ in range(num_workers)]
        for i in range(num_workers):
            env = deepmind_lab.Lab(level, ['RGB_INTERLEAVED'], config=config)
            workers.append(Worker(WrapperEnv(env_list[i], length, seed_nb), i, a_size, trainer, model_path, global_episodes))

        saver = tf.train.Saver(max_to_keep=5)

      config_t = tf.ConfigProto(allow_soft_placement = True)
      with tf.Session(config = config_t) as sess:
        # set the seed
        np.random.seed(seed_nb)
        tf.set_random_seed(seed_nb)

        coord = tf.train.Coordinator()
        if load_model == True:
          print ('Loading Model...')
          ckpt = tf.train.get_checkpoint_state(load_model_path)
          saver.restore(sess,ckpt.model_checkpoint_path)
        else:
          sess.run(tf.global_variables_initializer())

        worker_threads = []
        for worker in workers:
          worker_work = lambda: worker.work(gamma,sess,coord,saver,train,num_episodes)
          thread = threading.Thread(target=(worker_work))
          thread.start()
          worker_threads.append(thread)
        coord.join(worker_threads)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--length', type=int, default=1000,
                      help='Number of steps to run the agent')
  parser.add_argument('--width', type=int, default=80,
                      help='Horizontal size of the observations')
  parser.add_argument('--height', type=int, default=80,
                      help='Vertical size of the observations')
  parser.add_argument('--fps', type=int, default=60,
                      help='Number of frames per second')
  parser.add_argument('--runfiles_path', type=str, default=None,
                      help='Set the runfiles path to find DeepMind Lab data')
  parser.add_argument('--level_script', type=str,
                      default='tests/empty_room_test',
                      help='The environment level script to load')
  parser.add_argument('--record', type=str, default=None,
                      help='Record the run to a demo file')
  parser.add_argument('--demo', type=str, default=None,
                      help='Play back a recorded demo file')
  parser.add_argument('--demofiles', type=str, default=None,
                      help='Directory for demo files')
  parser.add_argument('--video', type=str, default=None,
                      help='Record the demo run as a video')

  args = parser.parse_args()
  if args.runfiles_path:
    deepmind_lab.set_runfiles_path(args.runfiles_path)
  run(args.length, args.width, args.height, args.fps, args.level_script,
      args.record, args.demo, args.demofiles, args.video)
