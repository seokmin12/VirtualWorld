from SimulatorEnv import Env
from simulator.PPOAlgorithm.ppo_agent import DeepPPOAgent
import torch
import os
import numpy as np
from datetime import datetime
from utils.logger import Logger
from tqdm import tqdm


def train():
    """initialize environment hyperparameters"""
    has_continuous_action_space = False

    max_timesteps_len = 100 * 365 * 24  # total steps per episode: 100years * 365days * 24hours
    episode = 2000

    print_freq = 4  # print avg reward in the interval (in num timesteps)
    log_freq = 20  # log avg reward in the interval (in num timesteps)
    save_model_freq = 20  # save module frequency (in num timesteps)

    action_std = 0.6  # starting std for action distribution (for continuous case)
    action_std_decay_rate = 0.05  # linearly decay action_std
    min_action_std = 0.1  # minimum action_std
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency

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

    random_seed = 0  # set random seed if required (0 = no random seed)

    env = Env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    """logging"""
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # get number of log files in the log directory
    current_num_files = next(os.walk(log_dir))[2]
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

    checkpoint_path = directory + "/PPO_{}_{}.pth".format(random_seed, run_num)
    summary_log_f.info(f"save checkpoint path : {checkpoint_path}\n")

    summary_log_f.info(
        f"{'-' * 90}\n"
        f"max training timesteps : {max_timesteps_len}\n"
        f"model saving frequency : {str(save_model_freq)} timesteps\n"
        f"log frequency : {str(log_freq)} timesteps\n"
        f"printing average reward over episodes in last : {str(print_freq)} timesteps\n"
        f"{'-' * 90}\n"
        f"state space dimension : {state_dim}\n"
        f"action space dimension : {action_dim}\n"
        f"{'-' * 90}\n"
    )

    if has_continuous_action_space:
        summary_log_f.info(
            "Initializing a continuous action space policy\n"
            f"{'-' * 90}\n"
            f"starting std of action distribution : {action_std}\n"
            f"decay rate of std of action distribution : {action_std_decay_rate}\n"
            f"minimum std of action distribution : {min_action_std}\n"
            f"decay frequency of action distribution : {str(action_std_decay_freq)} timesteps\n"
        )
    else:
        summary_log_f.info("Initializing a discrete action space policy\n")
    summary_log_f.info(
        f"{'-' * 90}\n"
        f"PPO update frequency : {str(max_timesteps_len * 2)} timesteps\n"
        f"PPO K epochs : {ppo_config['n_epochs']}\n"
        f"PPO epsilon clip : {ppo_config['clip_range']}\n"
        f"discount factor (gamma) : {ppo_config['gamma']}\n"
        f"{'-' * 90}\n"
        f"optimizer learning rate actor/critic : {ppo_config['lr']}\n"
    )
    if random_seed:
        summary_log_f.info(
            f"{'-' * 90}\n"
            f"setting random seed to {random_seed}\n"
        )
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    summary_log_f.info(f"{'-' * 90}\n")

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

    update_interval = 20  # Number of timesteps before triggering a PPO update
    global_step = 0  # Global timestep counter

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
            global_step += 1
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
            f"Trading Skill: {env.entity.trading_skill:.2f}\n"
            f"Work Ethic: {env.entity.work_ethic:.2f}\n"
            f"Risk Tolerance: {env.entity.risk_tolerance:.2f}\n"

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


if __name__ == "__main__":
    train()
