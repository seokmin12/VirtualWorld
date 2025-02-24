import argparse
from tqdm import tqdm

# Import the PPO agent and the environment.
from agents.ppo import DeepPPOAgent
from simulator.SimulatorEnv import Env


def test(checkpoint_path: str, num_episodes: int = 10, max_steps: int = 700801):
    """
    Test the trained PPO module for a specified number of episodes.
    
    Args:
        checkpoint_path (str): Path to the saved checkpoint.
        num_episodes (int): Number of episodes to run.
        max_steps (int): Maximum steps per episode.
    """
    env = Env()

    # Create the agent with the same configuration used during training.
    agent = DeepPPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dim=128,
        lr=3e-4,
        gamma=0.999,
        gae_lambda=0.95,
        clip_range=0.2,
        value_coef=1.0,
        entropy_coef=0.01,
        aux_coef=0.1,
        batch_size=64,
        n_epochs=10,
        max_grad_norm=0.5
    )

    # Load the trained module checkpoint.
    agent.load(checkpoint_path)
    print(f"Loaded checkpoint from {checkpoint_path}")

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0.0
        terminated = False
        step = 0

        for _ in tqdm(range(max_steps), desc=f"Episode: {episode}", leave=False):
            if terminated:
                break
            # Get the action from the agent.
            action, log_prob, value = agent.select_action(state)

            # Take a step in the environment.
            next_state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            state = next_state
            terminated = done or truncated
            step += 1

        print(f"Episode {episode}/{num_episodes}: Total Balance: ${env.entity.account.balance} in {step} steps Age: {env.entity.age}")


def main():
    parser = argparse.ArgumentParser(description="Test a Trained PPO Agent")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--episodes", type=int, default=10, help="Number of test episodes")
    args = parser.parse_args()

    test(
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes
    )


if __name__ == "__main__":
    main()
