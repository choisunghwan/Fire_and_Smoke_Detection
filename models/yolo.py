# 필요한 패키지 import
import torch # Torch(머신러닝 라이브러리) 모듈
import torch.nn as nn # 신경망(Neural Networks) 모델 정의
from models.common import Conv

class Model(nn.Module):
    def forward(self, x):
        y = []

        for m in self.model:
            if m.f != -1:
                x = [x if j == -1 else y[j] for j in m.f]

            x = m(x)
            y.append(x if m.i in self.save else None)
        
        return x

    def fuse(self):
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m._non_persistent_buffers_set = set()

                fusedconv = nn.Conv2d(m.conv.in_channels,
                        m.conv.out_channels,
                        kernel_size=m.conv.kernel_size,
                        stride=m.conv.stride,
                        padding=m.conv.padding,
                        groups=m.conv.groups,
                        bias=True).requires_grad_(False).to(m.conv.weight.device)

                w_conv = m.conv.weight.clone().view(m.conv.out_channels, -1)
                w_bn = torch.diag(m.bn.weight.div(torch.sqrt(m.bn.eps + m.bn.running_var)))
                fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))
                
                b_conv = torch.zeros(m.conv.weight.size(0), device=m.conv.weight.device) if m.conv.bias is None else m.conv.bias
                b_bn = m.bn.bias - m.bn.weight.mul(m.bn.running_mean).div(torch.sqrt(m.bn.running_var + m.bn.eps))
                fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
                
                m.conv = fusedconv
                delattr(m, 'bn')
                m.forward = m.fuseforward
        
        # Model 정보 출력
        print('Model 정보 : {0} Layers, {1} parameters'.format(len(list(self.parameters())), sum(x.numel() for x in self.parameters())))
        
        return self

class Detect(nn.Module):
    def forward(self, x):
        z = []

        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
                self.grid[i] = torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

            y = x[i].sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
            z.append(y.view(bs, -1, self.no))

        return torch.cat(z, 1), x
