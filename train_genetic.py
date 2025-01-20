from GeneticSimulatorEnv import GeneticSimulatorEnv
from DQNAgent import DQNAgent
import numpy as np
import torch


def train_genetic():
    env = GeneticSimulatorEnv(population_size=100)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    generations = 50
    episodes_per_generation = 20
    batch_size = 32

    for generation in range(generations):
        generation_reward = 0

        for episode in range(episodes_per_generation):
            state = env.reset()
            episode_reward = 0

            while True:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)

                if done:
                    break

            generation_reward += episode_reward

        # 세대 종료 후 진화
        env.evolve_population()

        # 결과 출력
        avg_reward = generation_reward / episodes_per_generation
        best_fitness = env.population[0].fitness_score
        print(f"Generation {generation + 1}")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Best Fitness: {best_fitness:.2f}")
        print(f"Best Entity Traits: Risk={env.population[0].risk_tolerance:.2f}, "
              f"Work={env.population[0].work_ethic:.2f}, "
              f"Trade={env.population[0].trading_skill:.2f}")
        print("-" * 50)

        # 모델 저장
        if generation % 5 == 0:
            torch.save({
                'model_state_dict': agent.model.state_dict(),
                'best_entity': env.population[0].__dict__
            }, f'./save_model/genetic_model_gen{generation}.pth')


if __name__ == "__main__":
    train_genetic()
