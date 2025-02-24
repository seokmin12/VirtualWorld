import torch
import os
import numpy as np
from datetime import datetime
from utils.logger import Logger
from tqdm import tqdm
import argparse

"""initialize environment hyperparameters"""
max_timesteps_len = 80 * 365 * 24 + 1  # total steps per episode: 80years * 365days * 24hours + 1hour
episode = 2000

log_freq = 20  # log avg reward in the interval (in num timesteps)

"""PPO hyperparameters"""
# Note: ppo_config keys have been updated to match the new DeepPPOAgent's arguments.
ppo_config = {
    'lr': 3e-4,
    'gamma': 0.999,  # Increased gamma for long-horizon tasks
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'value_coef': 1.0,  # formerly 'c1'
    'entropy_coef': 0.01,  # formerly 'c2'
    'aux_coef': 0.1,
    'batch_size': 64,
    'n_epochs': 10,
    'max_grad_norm': 0.5
}

random_seed = 2048  # set random seed if required (0 = no random seed)

"""logging"""
log_dir = "PPO_logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
# get number of log files in the log directory
current_num_files = next(os.walk(log_dir))[1]
run_num = len(current_num_files)

log_num_dir = log_dir + f'/PPO_{str(run_num)}'
if not os.path.exists(log_num_dir):
    os.makedirs(log_num_dir)
# create new log file for each run
timestep_log_f_name = log_num_dir + "/timestep_log.csv"
result_log_f_name = log_num_dir + "/results.csv"

summary_log_f = Logger(log_num_dir + "/summary_log.log")

summary_log_f.info(
    f"current logging run number : {run_num}\n"
    f"logging at : {timestep_log_f_name}\n"
)

"""checkpointing"""
directory = "PPO_preTrained"
if not os.path.exists(directory):
    os.makedirs(directory)

checkpoint_path = directory + "/PPO_{}.pt".format(run_num)
summary_log_f.info(f"save checkpoint path : {checkpoint_path}\n")


def logging(summary_log_f, state_dim, action_dim):
    summary_log_f.info(
        f"{'-' * 90}\n"
        f"max training timesteps : {max_timesteps_len}\n"
        f"log frequency : {str(log_freq)} timesteps\n"
        f"{'-' * 90}\n"
        f"state space dimension : {state_dim}\n"
        f"action space dimension : {action_dim}\n"
        f"{'-' * 90}\n"
        "Initializing a discrete action space policy\n"
        f"{'-' * 90}\n"
        f"PPO update frequency : {str(max_timesteps_len * 2)} timesteps\n"
        f"PPO K epochs : {ppo_config['n_epochs']}\n"
        f"PPO epsilon clip : {ppo_config['clip_range']}\n"
        f"discount factor (gamma) : {ppo_config['gamma']}\n"
        f"{'-' * 90}\n"
        f"optimizer learning rate actor/critic : {ppo_config['lr']}\n"
        f"{'-' * 90}\n"
        f"setting random seed to {random_seed}\n"
        f"{'-' * 90}\n"
    )
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)


def train():
    from agents.ppo import DeepPPOAgent
    from simulator.SimulatorEnv import Env
    env = Env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    logging(summary_log_f, state_dim, action_dim)

    # Initialize the new PPO agent
    ppo_agent = DeepPPOAgent(state_dim, action_dim, **ppo_config)

    start_time = datetime.now().replace(microsecond=0)

    timestep_log_f = open(timestep_log_f_name, "w+")
    timestep_log_f.write('episode,timestep,reward\n')

    result_log_f = open(result_log_f_name, "w+")
    result_log_f.write('episode,timestep,total_mined,balance,age,day,end_reason\n')

    # Initialize logging and experience buffers
    log_running_reward = 0
    log_running_episodes = 0

    for i_episode in tqdm(range(episode), desc="Total Episode"):
        state, _ = env.reset()
        current_ep_reward = 0
        time_step = 0
        terminated = False

        states = []
        actions = []
        rewards = []
        dones = []
        next_states = []
        old_log_probs = []  # To store the log probabilities returned by select_action
        old_values = []  # To store state-value estimates returned by select_action

        for _ in tqdm(range(max_timesteps_len), desc=f"Episode: {i_episode + 1}", leave=False):
            if terminated:
                break
            action, action_log_prob, value = ppo_agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)

            # Save the transition data along with the extra info needed for update()
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            next_states.append(next_state)
            old_log_probs.append(action_log_prob)
            old_values.append(value)

            state = next_state
            terminated = done or truncated
            time_step += 1
            current_ep_reward += reward

            # Log average reward at specified frequency
            if time_step % log_freq == 0 and log_running_episodes > 0:
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)
                timestep_log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                timestep_log_f.flush()
                log_running_reward = 0
                log_running_episodes = 1  # Reset to 1 instead of 0

        ppo_agent.update(states, actions, rewards, dones, next_states, old_log_probs, old_values)
        ppo_agent.save(checkpoint_path)
        result_log_f.write(
            '{},{},{},{},{},{},{}\n'.format(i_episode, time_step, env.entity.total_mined, env.entity.account.balance,
                                            env.entity.age, env.current_day,
                                            ("Health depleted" if env.entity.health <= 0 else
                                             "Max age reached" if env.entity.age >= 100 else
                                             "Happiness depleted" if env.entity.happiness <= 0.05 else
                                             "Max episode length reached" if time_step >= max_timesteps_len else
                                             "Unknown reason")))
        result_log_f.flush()

        states.clear()
        actions.clear()
        rewards.clear()
        dones.clear()
        next_states.clear()
        old_log_probs.clear()
        old_values.clear()

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        summary_log_f.info(
            f"\nEpisode {i_episode + 1} Summary:\n"
            f"Episode Reward: {current_ep_reward:.2f}\n"
            "\nCharacter Status:\n"
            f"Age: {env.entity.age} years\n"
            f"Survival Time: {env.current_day} days\n"
            f"Health: {env.entity.health:.2f}/100\n"
            f"Happiness: {env.entity.happiness:.2f}/1.0\n"
            f"Current Condition: {env.entity.current_condition}\n"

            "\nSkills & Attributes:"
            f"Mining Power: {env.entity.mining_power:.2f}\n"

            "\nAchievements:\n"
            f"Total Mined: {env.entity.total_mined}\n"
            f"Total Traded: {env.entity.total_traded}\n"
            f"Final Balance: ${env.entity.account.balance:.2f}\n"
            f"\nEpisode ended due to: " +
            ("Health depleted" if env.entity.health <= 0 else
             "Max age reached" if env.entity.age >= 100 else
             "Happiness depleted" if env.entity.happiness <= 0.05 else
             "Max episode length reached" if time_step >= max_timesteps_len else
             "Unknown reason") + "\n"
                                 f"Current Time: Day {env.current_day}\n"
                                 f"Current Timestep: {time_step}\n"
                                 f"{'-' * 90}\n"
        )

    timestep_log_f.close()
    result_log_f.close()

    end_time = datetime.now().replace(microsecond=0)
    summary_log_f.info(
        f"{'-' * 90}\n"
        f"Started training at (GMT) : {start_time}\n"
        f"Finished training at (GMT) : {end_time}\n"
        f"Total training time  : {end_time - start_time}\n"
        f"{'-' * 90}"
    )
    print("============================================================================================")
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


def train_multi_entity(num_entities: int):
    from agents.ppo2 import DeepPPOAgent
    from simulator.MultiEntityEnv import MultiEntityEnv
    env = MultiEntityEnv(num_entities)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    logging(summary_log_f, state_dim, action_dim)

    agents = [DeepPPOAgent(state_dim, action_dim, num_entities=1, **ppo_config) for _ in range(num_entities)]

    # Open log files before starting the episodes loop
    timestep_log_f = open(timestep_log_f_name, "w+")
    timestep_log_f.write('episode,agent,timestep,reward\n')

    result_log_f = open(result_log_f_name, "w+")
    result_log_f.write('episode,agent,timestep,total_mined,balance,age,day,end_reason\n')

    for i_episode in tqdm(range(episode), desc="Total Episode"):
        states, _ = env.reset()

        buffers = [{
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'next_states': [],
            'old_log_probs': [],
            'old_values': []
        } for _ in range(num_entities)]
        episode_rewards = [0.0] * num_entities

        time_step = 0
        terminated_flags = [False] * num_entities

        for _ in tqdm(range(max_timesteps_len), desc=f"Episode {i_episode + 1}", leave=False):
            if all(terminated_flags):
                break
            actions = []
            # For each agent, select its action if not yet terminated.
            for idx, agent in enumerate(agents):
                if terminated_flags[idx]:
                    actions.append(0)
                else:
                    action, log_prob, value = agent.select_action(states[idx])
                    actions.append(action)
                    buffers[idx]['states'].append(states[idx])
                    buffers[idx]['actions'].append(action)
                    buffers[idx]['old_log_probs'].append(log_prob)
                    buffers[idx]['old_values'].append(value)

            next_states, rewards, dones, infos = env.step(actions)
            for idx in range(num_entities):
                if not terminated_flags[idx]:
                    buffers[idx]['rewards'].append(rewards[idx])
                    buffers[idx]['dones'].append(dones[idx])
                    buffers[idx]['next_states'].append(next_states[idx])
                    episode_rewards[idx] += rewards[idx]

            states = next_states
            for idx in range(num_entities):
                if dones[idx]:
                    terminated_flags[idx] = True

            time_step += 1

            if time_step % log_freq == 0:
                for idx in range(num_entities):
                    log_avg_reward = round(episode_rewards[idx], 4)
                    timestep_log_f.write('{},{},{},{}\n'.format(i_episode, idx, time_step, log_avg_reward))
                timestep_log_f.flush()

        agents_checkpoint = {}
        for idx, agent in enumerate(agents):
            agent.update(
                buffers[idx]['states'],
                buffers[idx]['actions'],
                buffers[idx]['rewards'],
                buffers[idx]['dones'],
                buffers[idx]['next_states'],
                buffers[idx]['old_log_probs'],
                buffers[idx]['old_values']
            )

            last_info = infos[idx]
            end_reason = (
                "Health depleted" if last_info['health'] <= 0 else
                "Max age reached" if last_info['current_day'] >= 365 * 100 else
                "Happiness depleted" if last_info['happiness'] < 0.05 else
                "Max episode length reached"
            )
            result_log_f.write('{},{},{},{},{},{},{},{}\n'.format(
                i_episode,
                idx,
                time_step,
                last_info['total_mined'],
                last_info['balance'],
                last_info['age'],
                last_info['current_day'],
                end_reason
            ))
            result_log_f.flush()

            summary_log_f.info(
                f"\nEpisode {i_episode + 1} - Agent {idx} Summary:\n"
                f"Age: {env.entities[idx].age}\n"
                f"Health: {env.entities[idx].health}/100.0\n"
                f"Happiness: {env.entities[idx].happiness}\n"
                f"Day: {env.entities[idx].current_day}\n"
                f"Balance: ${env.entities[idx].account.balance}\n"
                f"Episode Reward: {episode_rewards[idx]:.2f}\n"
                f"End Reason: {end_reason}"
            )

            if hasattr(agent, 'actor'):
                agents_checkpoint[f'agent_{idx}_actor'] = agent.actor.state_dict()
            if hasattr(agent, 'critic'):
                agents_checkpoint[f'agent_{idx}_critic'] = agent.critic.state_dict()

        checkpoint_filename = os.path.join(directory,
                                           "PPO_{}_entity{}.pt".format(run_num, num_entities))
        torch.save(agents_checkpoint, checkpoint_filename)
        summary_log_f.info(
            f"{'=' * 50}\n"
        )

    # Close log files after all episodes are finished.
    timestep_log_f.close()
    result_log_f.close()

    end_time = datetime.now().replace(microsecond=0)
    summary_log_f.info(
        f"{'-' * 90}\n"
        f"Training Finished at (GMT): {end_time}\n"
        f"{'-' * 90}"
    )
    print("============================================================================================")
    print("Finished training at (GMT): ", end_time)
    print("============================================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PPO Agent")
    parser.add_argument("--env", type=str, required=True, default="single", help="Select env single or multi")
    parser.add_argument("--num_entities", type=int, default=1, help="Num of entities")
    args = parser.parse_args()

    if args.env == "single":
        train()
    else:
        train_multi_entity(args.num_entities)
