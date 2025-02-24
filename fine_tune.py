import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import argparse

from simulator.SimulatorEnv import Env
from agents.ppo import ActorCriticNetwork


# Dataset 클래스 (변경 없음)
class FineTuneDataset(Dataset):
    """
    A simple dataset that collects samples by interacting with the environment.
    Each sample consists of:
      - current state (a normalized 7-dim vector),
      - target balance (next state's index 0, as a 1-dim array),
      - target health (next state's index 1, as a 1-dim array)
    """

    def __init__(self, num_samples: int = 1000):
        self.data = []
        env = Env()
        state, _ = env.reset()
        for _ in range(num_samples):
            # use a random action to generate a new sample
            action = env.action_space.sample()
            next_state, reward, done, truncated, info = env.step(action)
            # Save current state and next state's balance and health as targets.
            self.data.append((
                np.array(state, dtype=np.float32),
                np.array(next_state[0:1], dtype=np.float32),
                np.array(next_state[1:2], dtype=np.float32)
            ))
            if done:
                state, _ = env.reset()
            else:
                state = next_state

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        state, balance_target, health_target = self.data[index]
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(balance_target, dtype=torch.float32),
            torch.tensor(health_target, dtype=torch.float32)
        )


class PPOFineTuner(pl.LightningModule):
    """
    Lightning module for fine tuning the PPO agent's ActorCriticNetwork.
    In this example, we fine tune on an auxiliary regression task:
    predicting the next state's normalized balance and health.
    """

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, lr: float = 1e-5):
        super().__init__()
        self.model = ActorCriticNetwork(state_dim, hidden_dim, action_dim)
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        states, balance_targets, health_targets = batch
        # Forward pass: obtain auxiliary predictions from the module.
        _, _, aux_balance, aux_health, _ = self.model(states)
        loss_balance = F.smooth_l1_loss(aux_balance, balance_targets)
        loss_health = F.smooth_l1_loss(aux_health, health_targets)
        loss = loss_balance + loss_health
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss_balance", loss_balance, on_step=False, on_epoch=True)
        self.log("train_loss_health", loss_health, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        states, balance_targets, health_targets = batch
        # 검증 단계 forward
        _, _, aux_balance, aux_health, _ = self.model(states)
        loss_balance = F.smooth_l1_loss(aux_balance, balance_targets)
        loss_health = F.smooth_l1_loss(aux_health, health_targets)
        loss = loss_balance + loss_health
        # 검증 손실 로그
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss_balance", loss_balance, on_step=False, on_epoch=True)
        self.log("val_loss_health", loss_health, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # 검증 손실 개선이 없으면 LR을 감소시킴
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }


def fine_tune(path):
    # 환경 초기화로 상태 차원 및 행동 차원 획득.
    env = Env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hidden_dim = 128  # PPO 에이전트 구성과 동일하게 설정.
    # 미세조정을 위한 Lightning module 생성.
    fine_tuner = PPOFineTuner(state_dim, hidden_dim, action_dim, lr=1e-5)

    # 사전 학습된 가중치 불러오기.
    checkpoint_path = path  # 실제 checkpoint 경로로 업데이트 필요.
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    fine_tuner.model.load_state_dict(checkpoint["network_state_dict"])
    print("Loaded pre-trained weights from:", checkpoint_path)

    # Fine-tuning dataset 생성 후 학습/검증 데이터셋으로 분리.
    dataset = FineTuneDataset(num_samples=5000)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Increase num_workers to reduce potential bottlenecks.
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=9,
        persistent_workers=True  # This flag keeps workers alive across epochs
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=9,
        persistent_workers=True
    )

    # Early stopping 콜백 추가 (검증 손실 개선이 없으면 종료).
    from pytorch_lightning.callbacks import EarlyStopping
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=3,
        verbose=True,
        mode='min'
    )

    # PyTorch Lightning Trainer 초기화.
    trainer = pl.Trainer(
        max_epochs=100,  # 더 정교한 미세조정을 위해 에포크 수를 증가.
        log_every_n_steps=10,
        callbacks=[early_stop_callback],
        gradient_clip_val=0.5  # Gradient clipping을 통한 학습 안정성 강화.
    )

    # 미세조정 진행 (학습 및 검증 루프 제공).
    trainer.fit(fine_tuner, train_dataloader, val_dataloader)

    # 미세조정된 모델과 함께 optimizer 및 scheduler 상태 저장.
    # trainer.optimizers는 리스트이므로 첫 번째 optimizer를 가져옵니다.
    optimizer = trainer.optimizers[0]
    # 추출된 lr_scheduler는 trainer.lr_scheduler_configs에 저장되어 있습니다.
    scheduler = trainer.lr_scheduler_configs[0].scheduler

    torch.save({
        "network_state_dict": fine_tuner.model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, "PPO_preTrained/PPO_fine_tuned.pt")
    print("Fine-tuned module (with optimizer and scheduler) saved to PPO_preTrained/PPO_fine_tuned.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine tune a Trained PPO Agent")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained module checkpoint")
    args = parser.parse_args()

    fine_tune(
        path=args.checkpoint
    )
