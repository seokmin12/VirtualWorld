import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from SimulatorEnv import SimulatorEnv
from simulator.DQNAlgorithm.dqn_agent import DQNAgent
from simulator.A2CAlgorithm.a2c_agent import A2CAgent
from simulator.PPOAlgorithm.ppo_agent import PPOAgent
import numpy as np


def train(algorithm='DQN', total_timesteps=100000):
    env = SimulatorEnv()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    if algorithm == 'DQN':
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=0.001,
            gamma=0.95,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            memory_size=10000,
            batch_size=64
        )
    elif algorithm == 'A2C':
        agent = A2CAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=0.001,
            gamma=0.95,
            entropy_coef=0.01
        )
    elif algorithm == 'PPO':
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=0.001,
            gamma=0.95,
            clip_ratio=0.2,
            entropy_coef=0.01
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # 학습
    episode = 0
    total_steps = 0

    while total_steps < total_timesteps:
        state, _ = env.reset()
        episode_reward = 0
        done = False

        # PPO와 A2C를 위한 배치 데이터 수집
        states, actions, rewards, next_states, dones = [], [], [], [], []
        old_log_probs = [] if algorithm == 'PPO' else None

        while not done and total_steps < total_timesteps:
            if algorithm == 'PPO':
                action, log_prob = agent.choose_action(state)
                old_log_probs.append(log_prob)
            else:
                action = agent.choose_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if algorithm == 'DQN':
                agent.train(state, action, reward, next_state, done)
            else:
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)

            state = next_state
            episode_reward += reward
            total_steps += 1

            # DQN의 타겟 네트워크 업데이트
            if algorithm == 'DQN' and total_steps % 1000 == 0:
                agent.update_target_network()

        # PPO와 A2C의 배치 학습
        if algorithm in ['A2C', 'PPO'] and len(states) > 0:
            if algorithm == 'PPO':
                agent.train(states, actions, old_log_probs, rewards, next_states, dones)
            else:  # A2C
                agent.train(states, actions, rewards, next_states, dones)

        episode += 1
        print(f"Episode {episode}: Reward = {episode_reward}, Steps = {total_steps}")

    # 모델 저장
    agent.save(f"./save_model/{algorithm}_simulator")

    # 평가
    evaluate(agent, env)


def evaluate(agent, env, num_episodes=10):
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            if isinstance(agent, PPOAgent):
                action = agent.choose_action(state, training=False)
            else:
                action = agent.choose_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        print(f"Evaluation Episode {episode + 1}: Reward = {episode_reward}")


if __name__ == "__main__":
    train(algorithm='PPO', total_timesteps=100000)
