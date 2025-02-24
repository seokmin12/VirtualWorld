<div align="center">
    <h3>
        🎮 가상 세상 시뮬레이터
    </h3>
    <p>
        Virtual World Simulator
    </p>
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
**시뮬레이션 가설(Simulation Hypothesis)** 은 우리가 인식하는 이 우주가 실재하는 세계가 아니라, 거대한 가상 시뮬레이션 속에서 구현된 것일 수 있다는 가설이다. 이는 철학자 **닉 보스트롬(Nick Bostrom)** 이 2003년 논문 ＜당신은 컴퓨터 시뮬레이션 속에 살고 있는가?(Are You Living In a Computer Simulation?)＞에서 제시한 개념으로, 고도로 발달한 문명이 현실과 구별할 수 없는 시뮬레이션을 생성했을 가능성을 논증한다.

나는 이 가설을 흥미롭게 받아들이며, 유사한 시스템을 구현할 수 있지 않을까 하는 고민을 해왔다. 현실 세계에서 경제 시스템이 형성되는 과정을 살펴보면, 화폐와 금융 시스템이 등장하면서 시장이 발달했고, 이는 국가의 형성과 발전으로 이어졌다. 이러한 원리가 디지털 환경에서도 적용될 수 있다면, 가상의 경제와 국가 또한 충분히 구축할 수 있을 것이다. 실제로 비트코인과 같은 암호화폐는 국경 없는 금융 시스템을 구축하고 있으며, 메타버스에서는 점차 독립적인 경제와 국가 개념이 자리 잡아가고 있다.

이러한 개념을 바탕으로, 나는 하나의 개체(인간)를 정의하고 그 개체가 상호작용할 수 있는 환경을 구축하였다. 해당 환경 내에서 개체는 특정 행동을 수행하며 생존을 위한 최적의 전략을 학습하고, 수명이 다할 때까지 최대한 효율적으로 살아남도록 설계되었다. 이를 통해, 시뮬레이션 내에서 자율적으로 진화하는 시스템을 구현하는 가능성을 탐구하고자 한다.

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
- **[module/Entity.py:](./module/Entity.py)** 개체 정의 및 행동 (채굴, 휴식, 여가, 종교 활동, 거래) 정의 
- **[module/Account.py:](./module/Account.py)** 개체의 계좌 정의 및 암호화
- **[module/Bank.py:](./module/Bank.py)** 가상 은행 정의 및 계좌들 간 블록체인화
- **[module/Transaction.py:](./module/Transaction.py)** 계좌의 거래내역 정의 및 암호화
- **[module/StockMarket.py:](./module/StockMarket.py)** 가상 주식 시장 정의
- **[module/Invest/Stock.py:](./module/Invest/Stock.py)** 가상 주식 정의

### Simulator
- **[simulator/SimulatorEnv.py:](./simulator/SimulatorEnv.py)** 단일 개체 시뮬레이션 환경 구현
- **[simulator/MultiEntityEnv.py:](./simulator/MultiEntityEnv.py)** 다중 개체 시뮬레이션 환경 구현

### Train
- **[train.py:](./train.py)** 에이전트 훈련
- **[fine_tune.py:](./fine_tune.py)** 미세조정 학습 지원

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
