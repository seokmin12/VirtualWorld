<div align="center">
    <h3>
        🎮 Virtual World Simulator
    </h3>
</div>

---
<div align="center">
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/License-MIT-yellow.svg">
    </a>
    <br>
    <img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=ffffff">
    <img src="https://img.shields.io/badge/pytorch-EE4C2C?style=for-the-badge&logo=python&logoColor=ffffff">
    <br>
    <a href="./README.md">
        <img alt="README in English" src="https://img.shields.io/badge/English-d9d9d9">
    </a>
    <a href="./README_KR.md">
        <img alt="README in Korean" src="https://img.shields.io/badge/한국어-d9d9d9">
    </a>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
        <a href="#introduction">Introduction</a>
    </li>
    <li>
        <a href="#getting-started">Getting Started</a>
        <ul>
            <li><a href="#dependencies">Dependencies</a></li>
        </ul>
    </li>
    <li>
        <a href="#usage">Usage</a>
            <ul>
                <li><a href="#run-training">Run Training</a></li>
                <li><a href="#run-testing">Run Testing</a></li>
                <li><a href="#run-fine-tuning">Run Fine Tuning</a></li>
            </ul>
    </li>
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#model-results">Model Results</a></li>
  </ol>
</details>

## Introduction
The **Simulation Hypothesis** proposes that the universe we perceive may not be a real, physical world, but rather a vast virtual simulation. This concept was introduced by philosopher **Nick Bostrom** in his 2003 paper [*"Are You Living In a Computer Simulation?"*](https://simulation-argument.com/simulation/), arguing that an advanced civilization may have created simulations indistinguishable from reality.

I found this hypothesis fascinating and began exploring whether a similar system could be implemented. Observing how economic systems develop in the real world, I noticed that with the introduction of currency and financial systems, markets emerged, leading to the formation and evolution of nations. If such principles can be applied to digital environments, then the creation of virtual economies and states should also be feasible. In fact, cryptocurrencies like Bitcoin have established borderless financial systems, and concepts of independent economies and virtual states are emerging within the **metaverse**.

Building on these ideas, I have defined an entity (a human) and constructed an interactive environment in which this entity performs specific actions, learns optimal survival strategies, and strives to live as efficiently and as long as possible. Through this approach, I aim to explore the potential for autonomous evolution within a simulation.

## Getting Started

### Dependencies
```
pip install -r requirements.txt
```

## Usage

### Run Training
``` 
# Single Entity
python train.py --env single

# multi Entity
python train.py --env multi --num_entities <num of entities>
```

### Run Testing
```
python test.py --checkpoint <path_to_trained_model> --episodes <num of test episodes>
```

### Run Fine Tuning
```
python fine_tune.py --checkpoint <path_to_trained_model>
```

## Project Structure
### Module
- **[module/Entity.py:](./module/Entity.py)** Defines entities and their actions (mining, resting, leisure, religious activities, transactions).
- **[module/Account.py:](./module/Account.py)** Defines entity accounts and encryption.
- **[module/Bank.py:](./module/Bank.py)** Defines a virtual bank and blockchain-based transactions between accounts.
- **[module/Transaction.py:](./module/Transaction.py)** Manages transaction records and encryption.
- **[module/StockMarket.py:](./module/StockMarket.py)** Defines a virtual stock market.
- **[module/Invest/Stock.py:](./module/Invest/Stock.py)** Defines virtual stocks.

### Simulator
- **[simulator/SimulatorEnv.py:](./simulator/SimulatorEnv.py)** Implements a single-entity simulation environment.
- **[simulator/MultiEntityEnv.py:](./simulator/MultiEntityEnv.py)** Implements a multi-entity simulation environment.

### Training
- **[train.py:](./train.py)** Agent training.
- **[fine_tune.py:](./fine_tune.py)** Supports fine-tuning of the trained model.

## Model Results
[Results](./PPO_logs/PPO_3/results.csv)
| Episode | Timestep | Total Mined | Balance | Age | Day | End Reason                  |
|---------|----------|-------------|---------|-----|-----|-----------------------------|
| 0       | 73       | 17          | 850     | 20  | 3   | Health depleted             |
| 1       | 27       | 6           | 300     | 20  | 1   | Happiness depleted          |
| 2       | 46       | 11          | 550     | 20  | 1   | Health depleted             |
| 3       | 50       | 9           | 450     | 20  | 2   | Health depleted             |
| 4       | 200      | 17          | 850     | 20  | 8   | Health depleted             |
| 5       | 81       | 10          | 500     | 20  | 3   | Health depleted             |
| ...     | ...      | ...         | ...     | ... | ... | ...                         |
| 26      | 26       | 13          | 650     | 20  | 1   | Happiness depleted          |
| 27      | 700800   | 1966        | 98300   | 99  | 29199| Max episode length reached  |