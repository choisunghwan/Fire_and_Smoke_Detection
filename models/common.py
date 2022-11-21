# 필요한 패키지 import
import torch # Torch(머신러닝 라이브러리) 모듈
import torch.nn as nn # 신경망(Neural Networks) 모델 정의

# batch_size, channel, width, height 값 변경
# - x(b, c, w, h) -> y(b, 4c, w/2, h/2)
class Focus(nn.Module):
    def forward(self, x):
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))

  
# Convolution + Batch Normalization Layer
# - 활성화 함수 : Hard Swish
class Conv(nn.Module):
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


# BottleNeck Layer
# - Short-cut Connection
class Bottleneck(nn.Module):
    # Standard bottleneck
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# BottleNeck CSP(Cross Stage Partial) Layer
# - cv1(Conv1) : Convolution + Batch Normalization Layer
# - cv2(Conv2) : Convolution Layer
# - cv3(Conv3) : Convolution Layer
# - cv4(Conv4) : Convolution + Batch Normalization Layer
# - y1 : Short-Connection으로 연결된 cv1 -> cv3 연산값
# - y2 : cv2 연산값
# - 출력값 : y1 + y2 -> cv4 연산값
class BottleneckCSP(nn.Module):
    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)

        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


# Spatial Pyramid Pooling Layer
# - YOLOv3-SPP에서 사용
# - 5*5, 9*9, 13*13 feature map 사용
# - 출력값 : (5 + 9 + 13) = 27의 크기로 고정된 1차원 형태의 배열(Fully Connected Layer의 입력)
class SPP(nn.Module):
    def forward(self, x):
        x = self.cv1(x)

        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


# Concatenate Layer
# - 2개의 Convolution Layer 연산값을 합침
class Concat(nn.Module):
    def forward(self, x):
        return torch.cat(x, self.d)
