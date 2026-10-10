"""DDPM のノイズスケジュールと前方/逆過程（第08・09章）。

- beta_t       : 各ステップで加えるノイズの分散
- alpha_t      : 1 - beta_t
- alpha_bar_t  : alpha の累積積（ノイズの残存率）
"""

from __future__ import annotations

import torch


class NoiseSchedule:
    def __init__(self, num_timesteps: int = 200, beta_start: float = 1e-4,
                 beta_end: float = 0.02, device: str = "cpu"):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    # --- 前方過程（学習不要、数式で決まる）---
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """x_t = sqrt(abar) x0 + sqrt(1-abar) eps（第08章の式）。"""
        a_bar = self.alpha_bars[t].view(-1, *([1] * (x0.dim() - 1)))
        return a_bar.sqrt() * x0 + (1.0 - a_bar).sqrt() * noise

    # --- 逆過程（1ステップのノイズ除去）---
    def p_sample_step(self, model, x_t: torch.Tensor, t: torch.Tensor,
                      condition: torch.Tensor | None = None) -> torch.Tensor:
        """x_t から x_{t-1} を推定する（第08章の逆過程）。"""
        beta_t = self.betas[t].view(-1, *([1] * (x_t.dim() - 1)))
        alpha_t = self.alphas[t].view(-1, *([1] * (x_t.dim() - 1)))
        a_bar = self.alpha_bars[t].view(-1, *([1] * (x_t.dim() - 1)))

        eps_pred = model(x_t, t, condition)
        mean = (1.0 / alpha_t.sqrt()) * (x_t - (beta_t / (1.0 - a_bar).sqrt()) * eps_pred)
        if int(t.max().item()) == 0:
            return mean
        return mean + beta_t.sqrt() * torch.randn_like(x_t)

    def to(self, device: str) -> "NoiseSchedule":
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self
