import collections
import gym
import numpy as np
import tensorflow as tf
import tqdm

from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple


# Create the environment
env = gym.make("CartPole-v0")

# Set seed for experiment reproducibility
seed = 42
env.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

class ActorCritic(tf.keras.Model):
  """Combined actor-critic network."""

  def __init__(
      self,
      num_actions: int, #可执行行动的数量
      num_hidden_units: int): # 隐藏层单元
    """Initialize."""
    super().__init__()

    self.common = layers.Dense(num_hidden_units, activation="relu") # 不知道这层干得啥
    self.actor = layers.Dense(num_actions)  # 行动器
    self.critic = layers.Dense(1)  # 评价器
  # -> 表示函数返回类型
  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:  # 建立网络？？？
    x = self.common(inputs)
    return self.actor(x), self.critic(x)

num_actions = env.action_space.n  # 2
num_hidden_units = 128

model = ActorCritic(num_actions, num_hidden_units)   #建立模型

# Wrap OpenAI Gym's `env.step` call as an operation in a TensorFlow function.
# This would allow it to be included in a callable TensorFlow graph.

def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Returns state, reward and done flag given an action."""

  state, reward, done, _ = env.step(action)  # 执行动作，得到环境的反馈，并将反馈转化为对应的数据类型
  return (state.astype(np.float32),
          np.array(reward, np.int32),
          np.array(done, np.int32))

# 将Python函数打包成一个TensorFlow op
def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
  return tf.numpy_function(env_step, [action],
                           [tf.float32, tf.int32, tf.int32])

# 运行一次训练
def run_episode(
    initial_state: tf.Tensor,
    model: tf.keras.Model,
    max_steps: int) -> List[tf.Tensor]:
  """Runs a single episode to collect training data."""

  # 行动的概率
  action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  # 状态的价值
  values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  # 行动的回报
  rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

  initial_state_shape = initial_state.shape
  state = initial_state

  for t in tf.range(max_steps):
    # Convert state into a batched tensor (batch size = 1)
    # 将状态转化为一个tensor
    state = tf.expand_dims(state, 0)

    # Run the model and to get action probabilities and critic value
    # 运行模型以得到行动概率和评价值
    action_logits_t, value = model(state)

    # Sample next action from the action probability distribution
    # 从行动概率之中随机采样一个行动
    action = tf.random.categorical(action_logits_t, 1)[0, 0]
    # 对随机采样的行动求概率
    action_probs_t = tf.nn.softmax(action_logits_t)

    # 储存t时刻的评价值
    # Store critic values
    values = values.write(t, tf.squeeze(value))

    # 储存t时刻选择该采样动作的概率
    # Store log probability of the action chosen
    action_probs = action_probs.write(t, action_probs_t[0, action])

    # 执行采样的该动作，然后转移到下一个状态并获得收益
    # Apply action to the environment to get next state and reward
    state, reward, done = tf_env_step(action)
    state.set_shape(initial_state_shape)

    # 储存状态转移之后的收益，即t时刻选择action之后得到的收益
    # Store reward
    rewards = rewards.write(t, reward)

    # 如果训练完成，则结束
    if tf.cast(done, tf.bool):
      break

  # 将所有的行动选择概率组织成数组，其余类推
  action_probs = action_probs.stack()
  values = values.stack()
  rewards = rewards.stack()

  # 返回结果
  return action_probs, values, rewards

# 计算每一时间步的期望收益
def get_expected_return(
    rewards: tf.Tensor,
    gamma: float, # 折扣率y
    standardize: bool = True) -> tf.Tensor:
  """Compute expected returns per timestep."""

  # 计算总的时间步数
  n = tf.shape(rewards)[0]
  returns = tf.TensorArray(dtype=tf.float32, size=n)

  # 从后一个时间步的收益反向累计总收益
  # Start from the end of `rewards` and accumulate reward sums
  # into the `returns` array
  rewards = tf.cast(rewards[::-1], dtype=tf.float32) # 将收益序列
  discounted_sum = tf.constant(0.0)
  discounted_sum_shape = discounted_sum.shape
  for i in tf.range(n):
    reward = rewards[i]
    discounted_sum = reward + gamma * discounted_sum
    discounted_sum.set_shape(discounted_sum_shape)
    returns = returns.write(i, discounted_sum)
  returns = returns.stack()[::-1]

  if standardize:
    returns = ((returns - tf.math.reduce_mean(returns)) /
               (tf.math.reduce_std(returns) + eps))

  return returns

huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

def compute_loss(
    action_probs: tf.Tensor,
    values: tf.Tensor,
    returns: tf.Tensor) -> tf.Tensor:
  """Computes the combined actor-critic loss."""

  advantage = returns - values

  action_log_probs = tf.math.log(action_probs)
  actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

  critic_loss = huber_loss(values, returns)

  return actor_loss + critic_loss

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


@tf.function
def train_step(
    initial_state: tf.Tensor,
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    gamma: float,
    max_steps_per_episode: int) -> tf.Tensor:
  """Runs a model training step."""

  with tf.GradientTape() as tape:

    # Run the model for one episode to collect training data
    action_probs, values, rewards = run_episode(
        initial_state, model, max_steps_per_episode)

    # Calculate expected returns
    returns = get_expected_return(rewards, gamma)

    # Convert training data to appropriate TF tensor shapes
    action_probs, values, returns = [
        tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

    # Calculating loss values to update our network
    loss = compute_loss(action_probs, values, returns)

  # Compute the gradients from the loss
  grads = tape.gradient(loss, model.trainable_variables)

  # Apply the gradients to the model's parameters
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  episode_reward = tf.math.reduce_sum(rewards)

  return episode_reward

# time

max_episodes = 10000
max_steps_per_episode = 1000

# Cartpole-v0 is considered solved if average reward is >= 195 over 100
# consecutive trials
reward_threshold = 195
running_reward = 0

# Discount factor for future rewards
gamma = 0.99

with tqdm.trange(max_episodes) as t:
  for i in t:
    initial_state = tf.constant(env.reset(), dtype=tf.float32)
    episode_reward = int(train_step(
        initial_state, model, optimizer, gamma, max_steps_per_episode))

    running_reward = episode_reward*0.01 + running_reward*.99

    t.set_description(f'Episode {i}')
    t.set_postfix(
        episode_reward=episode_reward, running_reward=running_reward)

    # Show average episode reward every 10 episodes
    if i % 10 == 0:
      pass # print(f'Episode {i}: average reward: {avg_reward}')

    if running_reward > reward_threshold:
        break

print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')

