from SimulatorEnv import Env
from simulator.PPOAlgorithm.ppo_agent import PPO
import torch
import os
import numpy as np
from tqdm import tqdm


def train():
    """initialize environment hyperparameters"""
    has_continuous_action_space = False

    max_timesteps_len = 100 * 365 * 24  # 총 스텝 수 100년 * 365일 * 24시간

    print_freq = max_timesteps_len * 4  # print avg reward in the interval (in num timesteps)
    log_freq = max_timesteps_len * 2  # log avg reward in the interval (in num timesteps)
    save_model_freq = int(5e5)  # save model frequency (in num timesteps)

    action_std = 0.6  # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)

    """PPO hyperparameters"""
    update_timestep = max_timesteps_len * 2  # 더 자주 업데이트
    K_epochs = 80  # 에포크 수 증가
    eps_clip = 0.2
    gamma = 0.99

    lr_actor = 0.0003
    lr_critic = 0.001

    random_seed = 0  # set random seed if required (0 = no random seed)

    env = Env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    """logging"""
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    # create new log file for each run
    log_f_name = log_dir + '/PPO' + "_log_" + str(run_num) + ".csv"
    print("current logging run number" + " : ", run_num)
    print("logging at : " + log_f_name)

    """checkpointing"""
    run_num_pretrained = 0  # change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + "/PPO_{}_{}.pth".format(random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)

    # print all hyperparameters

    print("--------------------------------------------------------------------------------------------")

    print("max training timesteps : ", max_timesteps_len)
    # print("max timesteps per episode : ", max_ep_len)

    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")

    print("--------------------------------------------------------------------------------------------")

    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)

    print("--------------------------------------------------------------------------------------------")

    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")

    else:
        print("Initializing a discrete action space policy")

    print("--------------------------------------------------------------------------------------------")

    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)

    print("--------------------------------------------------------------------------------------------")

    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)

    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    print("============================================================================================")

    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                    action_std)

    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')

    # Initialize these variables before the episode loop
    log_running_reward = 0
    log_running_episodes = 0

    episode = 20
    for i_episode in range(episode):
        state, _ = env.reset()
        current_ep_reward = 0
        time_step = 0
        terminated = False

        while not terminated and time_step < max_timesteps_len:
            action = ppo_agent.select_action(state)
            state, reward, done, truncated, info = env.step(action)

            terminated = done or truncated

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if update_timestep % time_step == 0:
                ppo_agent.update()

            # if continuous action space, then decay action std of output action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # update logging only if we have completed at least one episode
            if log_freq % time_step == 0 and log_running_episodes > 0:
                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 1  # Reset to 1 instead of 0

            # save model weights
            if save_model_freq % time_step == 0:
                ppo_agent.save(checkpoint_path)

            # break; if the episode is over
            if done:
                break

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

        print(f"\nEpisode {i_episode} Summary:")
        print(f"Episode Reward: {current_ep_reward:.2f}")
        print("\nCharacter Status:")
        print(f"Age: {env.entity.age} years")
        print(f"Survival Time: {env.current_day} days")
        print(f"Health: {env.entity.health:.2f}/100")
        print(f"Happiness: {env.entity.happiness:.2f}/1.0")
        print(f"Current Condition: {env.entity.current_condition}")

        print("\nSkills & Attributes:")
        print(f"Mining Power: {env.entity.mining_power:.2f}")
        print(f"Trading Skill: {env.entity.trading_skill:.2f}")
        print(f"Work Ethic: {env.entity.work_ethic:.2f}")
        print(f"Risk Tolerance: {env.entity.risk_tolerance:.2f}")

        print("\nAchievements:")
        print(f"Total Mined: {env.entity.total_mined}")
        print(f"Total Traded: {env.entity.total_traded}")
        print(f"Final Balance: ${env.entity.account.balance:.2f}")

        print(f"\nEpisode ended due to: " +
              ("Health depleted" if env.entity.health <= 0 else
               "Max age reached" if env.entity.age >= 100 else
               "Happiness depleted" if env.entity.happiness <= 0.05 else
               "Max episode length reached" if time_step >= max_timesteps_len else
               "Unknown reason"))
        print(f"Current Time: Day {env.current_day}")
        print(f"Current Timestep: {time_step}")
        print("-" * 50)

    log_f.close()


if __name__ == "__main__":
    train()
