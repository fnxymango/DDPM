"""DDPM forward/reverse process (Ho et al., 2020).

Index convention: code t in [0, T-1] corresponds to paper t in [1, T].
"""
import torch
import torch.nn.functional as F


class Diffusion:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        # Sec. 4: linear schedule beta_1=1e-4 .. beta_T=0.02, T=1000
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    @staticmethod
    def _at(a, t):
        return a[t].view(-1, 1, 1, 1)

    def q_sample(self, x0, t, noise=None):
        """Eq. (4): q(x_t | x_0) = N(sqrt(abar_t) x_0, (1 - abar_t) I)."""
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self._at(self.alpha_bars, t)
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise

    def loss(self, model, x0):
        """Eq. (14) L_simple / Algorithm 1."""
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        return F.mse_loss(model(xt, t), noise)

    @torch.no_grad()
    def p_sample(self, model, xt, t_int):
        """Algorithm 2, one step. sigma_t^2 = beta_t (Sec. 3.2)."""
        t = torch.full((xt.shape[0],), t_int, device=xt.device, dtype=torch.long)
        beta, alpha, ab = self.betas[t_int], self.alphas[t_int], self.alpha_bars[t_int]
        eps = model(xt, t)
        mean = (xt - beta / (1.0 - ab).sqrt() * eps) / alpha.sqrt()
        if t_int == 0:
            return mean
        return mean + beta.sqrt() * torch.randn_like(xt)

    @torch.no_grad()
    def sample(self, model, shape, device, keep=()):
        x = torch.randn(shape, device=device)
        traj = {self.T: x.clone()} if self.T in keep else {}
        for t_int in reversed(range(self.T)):
            x = self.p_sample(model, x, t_int)
            if t_int in keep:
                traj[t_int] = x.clone()
        return x.clamp(-1, 1), traj
