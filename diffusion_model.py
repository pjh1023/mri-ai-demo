# -*- coding: utf-8 -*-
"""
노이즈에서 MRI를 그려내는 diffusion(DDPM) 모델.
- DiffusionUNet : "노이즈 낀 사진 + 몇 단계인지(t)" 를 받아 "섞여 있는 노이즈" 를 맞히는 U-Net (128x128)
- NoiseSchedule : 사진에 노이즈를 몇 % 섞을지 정하는 표 (t = 0 깨끗함 ~ T 완전 노이즈)
노트북에서는 이 파일을 import 해서 쓰고, 학습 루프와 생성 루프는 노트북 안에 직접 적는다.
"""
import math, torch, torch.nn as nn, torch.nn.functional as F


def time_embedding(t, dim):
    "단계 번호 t(정수)를 sin/cos 파동 섞음으로 바꿔서 신경망이 읽기 쉽게 만든다"
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=1)


class ResBlock(nn.Module):
    "돋보기(3x3 conv) 2개 + '지금 몇 단계인지' 정보를 섞어 주는 블록"
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.norm1, self.conv1 = nn.GroupNorm(8, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.norm2, self.conv2 = nn.GroupNorm(8, cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.temb = nn.Linear(tdim, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(temb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Attention(nn.Module):
    "사진의 멀리 떨어진 부분끼리도 서로 참고하게 하는 층 (가장 작은 해상도에서만 사용)"
    def __init__(self, c, heads=4):
        super().__init__()
        self.heads, self.norm = heads, nn.GroupNorm(8, c)
        self.qkv, self.proj = nn.Conv2d(c, 3 * c, 1), nn.Conv2d(c, c, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        out = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.proj(out.transpose(-1, -2).reshape(b, c, h, w))


class DiffusionUNet(nn.Module):
    "128→64→32→16 으로 줄이며 보고 다시 키우는 U-Net. 각 크기마다 ResBlock 2개, 16x16에서는 Attention 추가"
    def __init__(self, base=64, mults=(1, 2, 4, 8), tdim=128):
        super().__init__()
        self.tdim = tdim
        self.tmlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        chs = [base * m for m in mults]                                  # 예: 32, 64, 128, 256
        self.inp = nn.Conv2d(1, chs[0], 3, padding=1)
        self.enc, self.dec, skips = nn.ModuleList(), nn.ModuleList(), []
        c = chs[0]
        for i, co in enumerate(chs):                                     # 줄이며 보기
            blocks = nn.ModuleList([ResBlock(c, co, tdim), ResBlock(co, co, tdim)])
            self.enc.append(blocks); skips += [co, co]; c = co
        self.mid = nn.ModuleList([ResBlock(c, c, tdim), Attention(c), ResBlock(c, c, tdim)])
        for i, co in reversed(list(enumerate(chs))):                     # 키우며 그리기 (줄이기 전 모습과 합쳐서)
            blocks = nn.ModuleList([ResBlock(c + skips.pop(), co, tdim), ResBlock(co + skips.pop(), co, tdim)])
            self.dec.append(blocks); c = co
        self.attn_enc = Attention(chs[-1]); self.attn_dec = Attention(chs[-1])
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, x, t):
        temb = self.tmlp(time_embedding(t, self.tdim))
        h, skips = self.inp(x), []
        for i, blocks in enumerate(self.enc):
            if i > 0: h = F.avg_pool2d(h, 2)
            for blk in blocks:
                h = blk(h, temb); skips.append(h)
            if i == len(self.enc) - 1: h = self.attn_enc(h)
        h = self.mid[2](self.mid[1](self.mid[0](h, temb)), temb)
        for i, blocks in enumerate(self.dec):
            for blk in blocks:
                h = blk(torch.cat([h, skips.pop()], 1), temb)
            if i == 0: h = self.attn_dec(h)
            if i < len(self.dec) - 1: h = F.interpolate(h, scale_factor=2, mode="nearest")
        return self.out(h)


class NoiseSchedule:
    "t단계에서 사진을 얼마나 남기고(√ᾱ) 노이즈를 얼마나 섞을지(√(1-ᾱ)) 정해 둔 표"
    def __init__(self, T=1000):
        self.T = T
        betas = torch.linspace(1e-4, 0.02, T)
        self.alpha_bar = torch.cumprod(1 - betas, 0)        # 단계가 갈수록 0에 가까워짐

    def add_noise(self, x0, t, noise):
        "깨끗한 사진 x0에 t단계만큼 노이즈를 섞는다"
        ab = self.alpha_bar[t][:, None, None, None]
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    @torch.no_grad()
    def sample(self, model, n, size=128, steps=50, keep_every=None, generator=None):
        "완전한 노이즈에서 출발해 steps번에 걸쳐 조금씩 노이즈를 걷어낸다 (DDIM 방식)"
        dev = next(model.parameters()).device
        x = torch.randn(n, 1, size, size, generator=generator).to(dev)
        ts = torch.linspace(self.T - 1, 0, steps).long()
        frames = []
        for i, t in enumerate(ts):
            ab = self.alpha_bar[t]
            ab_prev = self.alpha_bar[ts[i + 1]] if i + 1 < steps else torch.tensor(1.0)
            pred_noise = model(x, t.repeat(n).to(dev))               # 지금 섞여 있는 노이즈를 맞힌다
            x0 = ((x - (1 - ab).sqrt() * pred_noise) / ab.sqrt()).clamp(-1, 1)   # 그걸 빼면 깨끗한 사진의 추정치
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * pred_noise          # 한 단계 덜 노이즈 낀 상태로
            if keep_every and (i % keep_every == 0 or i == steps - 1):
                frames.append(x0.clone())
        return (x.clamp(-1, 1), frames) if keep_every else x.clamp(-1, 1)
