import tensorflow as tf
import tensorflow.contrib.slim as slim
import threading
import multiprocessing
import numpy as np
from meta_rl.utils import *
from meta_rl.ac_network import AC_Network
import time
import ray

import os
import random
import six

def _action(*entries):
  return np.array(entries, dtype=np.intc)

def deepmind_action_api(action):
  return random.choice(Worker.ACTION_LIST[action:action + 1])

# this function returns 5 things, so num_return_vals=5
@ray.remote(num_return_vals=5)
def episode(reset, step, ac_network, sess, rnn_state):
  episode_buffer = []
  episode_reward = 0
  total_steps = 0
  episode_step_count = 0

  d = False
  r = 0
  a = 0
  t = 0
  s = reset()

  # Allow us to remove noise when starting episode
  for i in range(5):
    _, r_, _, _ = step(np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.intc))

  while d == False:
    #Take an action using probabilities from policy network output.
    a_dist,v,rnn_state_new = sess.run([ac_network.policy,ac_network.value,ac_network.state_out],
      feed_dict={
      ac_network.state:[s],
      ac_network.prev_rewards:[[r]],
      ac_network.timestep:[[t]],
      ac_network.prev_actions:[a],
      ac_network.state_in[0]:rnn_state[0],
      ac_network.state_in[1]:rnn_state[1]})

    a = np.random.choice(a_dist[0],p=a_dist[0])
    a = np.argmax(a_dist == a)
    rnn_state = rnn_state_new
    action = deepmind_action_api(a)

    """Objectif: Reduce action space to speed up training time

    1st Action: No-Op, wait 1 frame to allow pictures to appears
    2nd Action: True Action taken
    3rd Action: Reverse action to go back at the center of the screen
    4th Action: No-Op, to wait 1 frame to allow the cross to appears
    """
    _, r_, _, _ = step(np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.intc))
    s1, r, d, t = step(action, True)
    r += r_
    if not d:
        _, r_, d, _ = step(-action)
        r += r_
        if not d:
            _, r_, d, _ = step(np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.intc))
            r += r_

    episode_buffer.append([s,a,r,t,d,v[0,0]])
    episode_values.append(v[0,0])
    episode_reward += r
    total_steps += 1
    episode_step_count += 1
    s = s1

  return episode_buffer, episode_values, episode_reward, episode_step_count, rnn_state

class Worker():

  ACTIONS = {
    'look_left': _action(-120, 0, 0, 0, 0, 0, 0),
    'look_right': _action(120, 0, 0, 0, 0, 0, 0),
  }

  ACTION_LIST = list(six.viewvalues(ACTIONS))

  def __init__(self, game, name, a_size, trainer, model_path, global_episodes):
    self.name = "worker_" + str(name)
    self.number = name
    self.model_path = model_path
    self.trainer = trainer
    self.global_episodes = global_episodes
    self.increment = self.global_episodes.assign_add(1)
    self.episode_rewards = []
    self.episode_lengths = []
    self.episode_mean_values = []
    self.summary_writer = tf.summary.FileWriter(model_path)

    #Create the local copy of the network and the tensorflow op to copy global paramters to local network
    self.local_AC = AC_Network(a_size, self.name, trainer)
    self.update_local_ops = update_target_graph('global',self.name)
    self.env = game
    self.list_time = []

  def train(self,rollout,sess,gamma,bootstrap_value):
    rollout = np.array(rollout)
    states = rollout[:, 0]
    actions = rollout[:, 1]
    rewards = rollout[:, 2]
    timesteps = rollout[:, 3]
    prev_rewards = [0] + rewards[:-1].tolist()
    prev_actions = [0] + actions[:-1].tolist()
    values = rollout[:, 5]

    self.pr = prev_rewards
    self.pa = prev_actions
    # Here we take the rewards and values from the rollout, and use them to
    # generate the advantage and discounted returns.
    # The advantage function uses "Generalized Advantage Estimation"
    self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
    discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
    self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
    advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
    advantages = discount(advantages,gamma)

    # Update the global network using gradients from loss
    # Generate network statistics to periodically save
    rnn_state = self.local_AC.state_init
    feed_dict = {self.local_AC.target_v:discounted_rewards,
      self.local_AC.state:np.stack(states,axis=0),
      self.local_AC.prev_rewards:np.vstack(prev_rewards),
      self.local_AC.prev_actions:prev_actions,
      self.local_AC.actions:actions,
      self.local_AC.timestep:np.vstack(timesteps),
      self.local_AC.advantages:advantages,
      self.local_AC.state_in[0]:rnn_state[0],
      self.local_AC.state_in[1]:rnn_state[1]}

    v_l,p_l,e_l,g_n,v_n,_ = sess.run([self.local_AC.value_loss,
      self.local_AC.policy_loss,
      self.local_AC.entropy,
      self.local_AC.grad_norms,
      self.local_AC.var_norms,
      self.local_AC.apply_grads],
      feed_dict=feed_dict)
    return v_l / len(rollout),p_l / len(rollout),e_l / len(rollout), g_n,v_n

  def work(self,gamma,sess,coord,saver,train, num_episodes):
    episode_count = sess.run(self.global_episodes)

    # set count to zero when loading a model
    if not train:
      episode_count = 0

    total_steps = 0
    print ("Starting worker " + str(self.number))
    with sess.as_default(), sess.graph.as_default():
      while not coord.should_stop() and episode_count <= num_episodes:
        sess.run(self.update_local_ops)
        episode_buffer = []
        episode_values = []
        episode_frames = []
        episode_reward = 0
        episode_step_count = 0
        rnn_state = self.local_AC.state_init

        start_time = time.time()
      
        # initialize ray
        ray.init()
      
        # creates a task (result_id is a "object ID" in ray)
        result_id = episode.remote(self.env.reset, self.env.step, self.local_AC, sess, rnn_state)
      
        # create a python object from the object ID (or "remote job")
        episode_buffer, episode_values, episode_reward, episode_step_count, rnn_state = ray.get(result_id)
        
        # shutdown ray
        ray.shutdown()

        episode_count += 1
        episode_time = time.time() - start_time
        print("\033[92mWORKER NAME >> " + self.name + " << " + "s\033[0m")
        print("\033[92mEpisode #" + str(episode_count) + " completed in " + str(episode_time) + "s\033[0m")
        self.list_time.append(episode_time)
        if (episode_count % 10 == 0):
          debug_print((episode_count / num_episodes) * 100, "Progress (%):")
          print("\033[34mMean time for the last 10 episodes was " + str(np.mean(self.list_time)) + "s\033[0m")
          self.list_time = []

        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_step_count)
        self.episode_mean_values.append(np.mean(episode_values))

        # Update the network using the experience buffer at the end of the episode.
        if len(episode_buffer) != 0 and train == True:
          print("\033[36mWORKER " + self.name + " started training..." + "\033[0m")
          start_train = time.time()
          v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,0.0)
          time_to_train = time.time() - start_train
          print("\033[31mWORKER " + self.name + " finished training in " + str(time_to_train) + "s" + "\033[0m")

        # Periodically save gifs of episodes, model parameters, and summary statistics.
        if episode_count % 10 == 0 and episode_count != 0:
          if episode_count % 10 == 0 and self.name == 'worker_0':
            if train == True:
              # save model
              os.makedirs(self.model_path+'/model-'+str(episode_count))
              saver.save(sess,self.model_path+'/model-'+str(episode_count)+
                    '/model-'+str(episode_count)+'.cptk')
              print ("Saved Model")

          # only track datapoints for training every 10 episoodes
          if train == True and self.name == 'worker_0':
            # For Tensorboard
            mean_reward = np.mean(self.episode_rewards[-10:])
            mean_length = np.mean(self.episode_lengths[-10:])
            mean_value = np.mean(self.episode_mean_values[-10:])
            summary = tf.Summary()
            summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
            summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
            summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
            if train == True:
              summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
              summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
              summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
              summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
              summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
            self.summary_writer.add_summary(summary, episode_count)

            self.summary_writer.flush()
        if self.name == 'worker_0':
          sess.run(self.increment)

    if not train:
      self.plot(episode_count-1, train)
