import torch
import torch.nn as nn

__all__ = ['C3k2_DFF']

class DFF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_atten = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.conv_redu = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.nonlin = nn.Sigmoid()

    def forward(self, x, skip):
        output = torch.cat([x, skip], dim=1)
        att = self.conv_atten(self.avg_pool(output))
        output = output * att
        output = self.conv_redu(output)
        att = self.conv1(x) + self.conv2(skip)
        att = self.nonlin(att)
        output = output * att
        return output

class Bottleneck_DFF(nn.Module):
    def __init__(self, c1, c2, use_dff=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = use_dff and c1 == c2
        self.DFF = DFF(c2)

    def forward(self, x):
        if self.add:
            results = self.DFF(x, self.cv2(self.cv1(x)))
        else:
            results = self.cv2(self.cv1(x))
        return results

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, use_dff=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = use_dff and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, use_dff=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, use_dff, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class C3(nn.Module):
    def __init__(self, c1, c2, n=1, use_dff=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, use_dff, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class C3k(C3):
    def __init__(self, c1, c2, n=1, use_dff=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, use_dff, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, use_dff, g, k=(k, k), e=1.0) for _ in range(n)))

class C3kDFF(C3):
    def __init__(self, c1, c2, n=1, use_dff=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, use_dff, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_DFF(c_, c_, use_dff, g, k=(k, k), e=1.0) for _ in range(n)))

class C3k2_DFF(C2f):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, use_dff=True, k=3):
        super().__init__(c1, c2, n, use_dff, g, e)
        self.m = nn.ModuleList(
            C3kDFF(self.c, self.c, 2, use_dff, g, e, k=k) if c3k else Bottleneck_DFF(self.c, self.c, use_dff, g, k=(k, k), e=1.0)
            for _ in range(n)
        )
