"""AMP（Adversarial Motion Priors）関連の処理。

第19章・第22章に対応する。

- スタイル報酬 r^S の計算（式(6)）
- 識別器 D の更新（式(3)〜(5)、勾配ペナルティ込み）

「シーン条件付き」にするのは、ここに入る amp_obs に物体特徴が
含まれているかどうかだけの違いである（第36章）。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .models import Discriminator


def style_reward(discriminator: Discriminator, amp_obs: torch.Tensor) -> torch.Tensor:
    """スタイル報酬 r^S(s_t, s_{t+1}) = -log(1 - D(s_t, s_{t+1}))  （式(6)）

    識別器が「本物っぽい（D→1）」と判定するほど大きくなる。
    """
    with torch.no_grad():
        logits = discriminator(amp_obs)
        d = torch.sigmoid(logits)
        # 数値安定のため clamp（log(0) を避ける）
        d = d.clamp(1e-4, 1 - 1e-4)
        return -torch.log(1.0 - d)


class AMPUpdater:
    """識別器の更新を担う（式(3)〜(5)）。"""

    def __init__(self, amp_obs_dim: int, cfg, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self.discriminator = Discriminator(amp_obs_dim, cfg.hidden_dims).to(device)
        self.optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=cfg.disc_learning_rate
        )
        self.bce = nn.BCEWithLogitsLoss()

    @torch.no_grad()
    def predict_style_reward(self, amp_obs: torch.Tensor) -> torch.Tensor:
        """ロールアウト中に使う軽量版（勾配不要）。"""
        return style_reward(self.discriminator, amp_obs)

    def update(self, policy_amp_obs: torch.Tensor, dataset_amp_obs: torch.Tensor) -> dict:
        """識別器を1回更新する。

        - policy_amp_obs : 方策のロールアウトから得た遷移（偽物, ラベル0）
        - dataset_amp_obs: モーションデータセットの遷移（本物, ラベル1）
        """
        cfg = self.cfg
        n = min(cfg.disc_batch_size, policy_amp_obs.shape[0], dataset_amp_obs.shape[0])

        idx_p = torch.randperm(policy_amp_obs.shape[0], device=self.device)[:n]
        idx_d = torch.randperm(dataset_amp_obs.shape[0], device=self.device)[:n]
        fake = policy_amp_obs[idx_p]
        real = dataset_amp_obs[idx_d]

        # --- 識別器の損失（式(3)〜(5)）---
        # 本物を1に、偽物を0にするクロスエントロピー。
        logits_real = self.discriminator(real)
        logits_fake = self.discriminator(fake)
        loss_real = self.bce(logits_real, torch.ones_like(logits_real))
        loss_fake = self.bce(logits_fake, torch.zeros_like(logits_fake))

        # 勾配ペナルティ（式(5)）: 入力に対する出力勾配の二乗和に罰則。
        real_gp = real.clone().detach().requires_grad_(True)
        logits_gp = self.discriminator(real_gp)
        gp = Discriminator.gradient_penalty(logits_gp, real_gp)

        loss = loss_real + loss_fake + cfg.gradient_penalty_weight * gp

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.discriminator.parameters(), cfg.max_grad_norm)
        self.optimizer.step()

        # 参考指標: 識別器の正解率（本物を本物、偽物を偽物と判定した割合）
        with torch.no_grad():
            acc_real = (torch.sigmoid(logits_real) > 0.5).float().mean().item()
            acc_fake = (torch.sigmoid(logits_fake) < 0.5).float().mean().item()
        return {
            "disc_loss": loss.item(),
            "disc_acc_real": acc_real,
            "disc_acc_fake": acc_fake,
            "grad_penalty": gp.item(),
        }
