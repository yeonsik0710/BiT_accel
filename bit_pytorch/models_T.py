# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Bottleneck ResNet v2 with GroupNorm and Weight Standardization."""

#   grad_out에서 한 channel 전체를 0로,
#   grad_out에 채널의 합이 작은 순서대로

from collections import OrderedDict  # pylint: disable=g-importing-member

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import random
import numpy as np

# 글로벌 변수로 각 레이어별 accum_c_max_values 선언
global_accum_c_max_values = {}

# grad_out의 zero ratio 모니터링을 위한 변수
output_gradient = {}

def get_output_gradient():
   global output_gradient

   return output_gradient

def get_max_tensor():
  global global_accum_c_max_values

  # # 딕셔너리의 각 텐서를 CPU로 이동하고 NumPy 배열로 변환
  # for layer_id in global_accum_c_max_values:
  #     global_accum_c_max_values[layer_id] = global_accum_c_max_values[layer_id].cpu().numpy()

  return global_accum_c_max_values

# new
class CustomConv2dFunction(Function):
    def __init__(self, zero_ratio):
        super(CustomConv2dFunction, self).__init__()
        self.accum_c_max_values = None

    def forward(ctx, input, weight, bias, stride, padding, dilation, groups, T, steps, layer_id):
        # 입력 및 파라미터 저장
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.T = T
        ctx.steps = steps
        ctx.layer_id = layer_id

        # Conv2d 연산 수행
        output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
        return output
    
    def backward(ctx, grad_output):
        # 저장된 입력 및 파라미터 불러오기
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        steps = ctx.steps
        layer_id = ctx.layer_id
        T = ctx.T
        
        # gradient 계산
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups)

        # 변수 정의
        abs_grad_output = torch.abs(grad_output)

        if steps < 2_000:
            # 각 레이어별 global_accum_c_max_values 초기화
            if layer_id not in global_accum_c_max_values:
                global_accum_c_max_values[layer_id] = torch.tensor([], device=grad_output.device)

            # # 모니터링
            # if (steps != 0) and ((steps % 10) == 0):
            #    print(f"Layer {layer_id}: {global_accum_c_max_values[layer_id].shape}")

            # 4차원 grad_out에서 channel 차원에 대해 max value 뽑기
            max_value, _ = torch.max(abs_grad_output.view(abs_grad_output.size(0), abs_grad_output.size(1), -1), dim=2)
            max_value, _ = max_value.max(dim=0)

            max_value = max_value.unsqueeze(0) # 1차원에서 2차원으로 변경

            # 현재 step의 max value 저장
            global_accum_c_max_values[layer_id] = torch.cat([max_value, global_accum_c_max_values[layer_id]], dim=0)

            
        else:
            if T > 0:
                # 1. 각 channel index에 대해 max 값을 구합니다 (batch와 spatial 차원(height, width)을 모두 무시하고, channel 별 최대값)
                max_per_channel, _ = torch.max(abs_grad_output.view(abs_grad_output.size(0), abs_grad_output.size(1), -1), dim=2)
                max_per_channel, _ = max_per_channel.max(dim=0)  # batch 차원에서 최대값을 가져옵니다 (channel 크기만큼 1D 텐서)

                # 2. max_per_channel이 T보다 작은지 여부를 확인 (True이면 그 channel을 모두 0으로 만듦)
                mask = max_per_channel < T

                # 3. mask가 True인 channel index에 대해 grad_output의 해당 channel 전체를 0으로 설정
                grad_output = grad_output.permute(1, 0, 2, 3)  # channel을 첫번째 차원으로 이동 (C, N, H, W)
                grad_output[mask, :, :, :] = 0  # T보다 작은 channel index에 대해 해당 channel의 값을 모두 0으로 설정
                grad_output = grad_output.permute(1, 0, 2, 3)  # 다시 원래 차원 순서 (N, C, H, W)로 복원
                # print(f'{layer_id}s Threshold is {T}.')
        
        
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 2, 3))

        output_gradient[layer_id] = grad_output
        
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None

# 기존 틀은 그대로인데, WS(Weight Standaraization만 추가)
class CustomConv2d(nn.Conv2d):
  def __init__(self, *args, layer_id=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.steps = 0
        self.T = 0
        self.layer_id = layer_id

  def forward(self, x):
    
    w = self.weight
    v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
    w = (w - m) / torch.sqrt(v + 1e-10)

    output = CustomConv2dFunction.apply(x, w, self.bias, self.stride, self.padding,
                    self.dilation, self.groups, self.T, self.steps, self.layer_id)


    return output


# 기존 StdConv2d
class StdConv2d(nn.Conv2d):
  def forward(self, x):
    w = self.weight
    v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
    w = (w - m) / torch.sqrt(v + 1e-10)
    return F.conv2d(x, w, self.bias, self.stride, self.padding,
                    self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, layer_id=None, bias=False):
  return CustomConv2d(cin, cout, kernel_size=3, stride=stride,
                   padding=1, bias=bias, groups=groups, layer_id=layer_id)


def conv1x1(cin, cout, stride=1, zero_ratio=0, layer_id=None, bias=False):
  return CustomConv2d(cin, cout, kernel_size=1, stride=stride,
                   padding=0, bias=bias, layer_id=layer_id)


# tensorflow 2 torch
def tf2th(conv_weights):
  """Possibly convert HWIO to OIHW."""
  if conv_weights.ndim == 4:
    conv_weights = conv_weights.transpose([3, 2, 0, 1])
  return torch.from_numpy(conv_weights)


class PreActBottleneck(nn.Module):
  """Pre-activation (v2) bottleneck block.

  Follows the implementation of "Identity Mappings in Deep Residual Networks":
  https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua

  Except it puts the stride on 3x3 conv when available.
  """

  def __init__(self, cin, cout=None, cmid=None, stride=1, layer_id=None):
    super().__init__()
    cout = cout or cin
    cmid = cmid or cout//4
    self.layer_id = layer_id

    self.gn1 = nn.GroupNorm(32, cin)
    self.conv1 = conv1x1(cin, cmid, layer_id=(self.layer_id + '_conv1'))
    self.gn2 = nn.GroupNorm(32, cmid)
    self.conv2 = conv3x3(cmid, cmid, stride=stride, layer_id=(self.layer_id + '_conv2'))  # Original code has it on conv1!!
    self.gn3 = nn.GroupNorm(32, cmid)
    self.conv3 = conv1x1(cmid, cout, layer_id=(self.layer_id + '_conv3'))
    self.relu = nn.ReLU(inplace=True)

    if (stride != 1 or cin != cout):
      # Projection also with pre-activation according to paper.
      self.downsample = StdConv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=False)
      # self.downsample = conv1x1(cin, cout, stride=stride, threshold=self.T, zero_ratio=self.zero_ratio)

  def forward(self, x):
    out = self.relu(self.gn1(x))

    # Residual branch
    residual = x
    if hasattr(self, 'downsample'):
      residual = self.downsample(out)

    # Unit's branch
    out = self.conv1(out)
    out = self.conv2(self.relu(self.gn2(out)))
    out = self.conv3(self.relu(self.gn3(out)))

    return out + residual

  def load_from(self, weights, prefix=''):
    convname = 'standardized_conv2d'
    with torch.no_grad():
      self.conv1.weight.copy_(tf2th(weights[f'{prefix}a/{convname}/kernel']))
      self.conv2.weight.copy_(tf2th(weights[f'{prefix}b/{convname}/kernel']))
      self.conv3.weight.copy_(tf2th(weights[f'{prefix}c/{convname}/kernel']))
      self.gn1.weight.copy_(tf2th(weights[f'{prefix}a/group_norm/gamma']))
      self.gn2.weight.copy_(tf2th(weights[f'{prefix}b/group_norm/gamma']))
      self.gn3.weight.copy_(tf2th(weights[f'{prefix}c/group_norm/gamma']))
      self.gn1.bias.copy_(tf2th(weights[f'{prefix}a/group_norm/beta']))
      self.gn2.bias.copy_(tf2th(weights[f'{prefix}b/group_norm/beta']))
      self.gn3.bias.copy_(tf2th(weights[f'{prefix}c/group_norm/beta']))
      if hasattr(self, 'downsample'):
        w = weights[f'{prefix}a/proj/{convname}/kernel']
        self.downsample.weight.copy_(tf2th(w))


class ResNetV2(nn.Module):
  """Implementation of Pre-activation (v2) ResNet mode."""

  def __init__(self, block_units, width_factor, head_size=21843, zero_head=False):
    super().__init__()
    wf = width_factor  # shortcut 'cause we'll use it a lot.

    # The following will be unreadable if we split lines.
    # pylint: disable=line-too-long
    self.root = nn.Sequential(OrderedDict([
        ('conv', StdConv2d(3, 64*wf, kernel_size=7, stride=2, padding=3, bias=False)),
        ('pad', nn.ConstantPad2d(1, 0)),
        ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0)),
        # The following is subtly not the same!
        # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
    ]))

    self.body = nn.Sequential(OrderedDict([
        ('block1', nn.Sequential(OrderedDict(
            [('unit01', PreActBottleneck(cin=64*wf, cout=256*wf, cmid=64*wf, layer_id='b1_u01'))] +
            [(f'unit{i:02d}', PreActBottleneck(cin=256*wf, cout=256*wf, cmid=64*wf, layer_id=f'b1_u{i:02d}')) for i in range(2, block_units[0] + 1)],
        ))),
        ('block2', nn.Sequential(OrderedDict(
            [('unit01', PreActBottleneck(cin=256*wf, cout=512*wf, cmid=128*wf, stride=2, layer_id='b2_u01'))] +
            [(f'unit{i:02d}', PreActBottleneck(cin=512*wf, cout=512*wf, cmid=128*wf, layer_id=f'b2_u{i:02d}')) for i in range(2, block_units[1] + 1)],
        ))),
        ('block3', nn.Sequential(OrderedDict(
            [('unit01', PreActBottleneck(cin=512*wf, cout=1024*wf, cmid=256*wf, stride=2, layer_id='b3_u01'))] +
            [(f'unit{i:02d}', PreActBottleneck(cin=1024*wf, cout=1024*wf, cmid=256*wf, layer_id=f'b3_u{i:02d}')) for i in range(2, block_units[2] + 1)],
        ))),
        ('block4', nn.Sequential(OrderedDict(
            [('unit01', PreActBottleneck(cin=1024*wf, cout=2048*wf, cmid=512*wf, stride=2, layer_id='b4_u01'))] +
            [(f'unit{i:02d}', PreActBottleneck(cin=2048*wf, cout=2048*wf, cmid=512*wf, layer_id=f'b4_u{i:02d}')) for i in range(2, block_units[3] + 1)],
        ))),
    ]))
    # pylint: enable=line-too-long

    self.zero_head = zero_head
    self.head = nn.Sequential(OrderedDict([
        ('gn', nn.GroupNorm(32, 2048*wf)),
        ('relu', nn.ReLU(inplace=True)),
        ('avg', nn.AdaptiveAvgPool2d(output_size=1)),
        ('conv', nn.Conv2d(2048*wf, head_size, kernel_size=1, bias=True)),
    ]))

  def forward(self, x):
    x = self.head(self.body(self.root(x)))
    assert x.shape[-2:] == (1, 1)  # We should have no spatial shape left.
    return x[...,0,0]

  def load_from(self, weights, prefix='resnet/'):
    with torch.no_grad():
      self.root.conv.weight.copy_(tf2th(weights[f'{prefix}root_block/standardized_conv2d/kernel']))  # pylint: disable=line-too-long
      self.head.gn.weight.copy_(tf2th(weights[f'{prefix}group_norm/gamma']))
      self.head.gn.bias.copy_(tf2th(weights[f'{prefix}group_norm/beta']))
      if self.zero_head:
        nn.init.zeros_(self.head.conv.weight)
        nn.init.zeros_(self.head.conv.bias)
      else:
        self.head.conv.weight.copy_(tf2th(weights[f'{prefix}head/conv2d/kernel']))  # pylint: disable=line-too-long
        self.head.conv.bias.copy_(tf2th(weights[f'{prefix}head/conv2d/bias']))

      for bname, block in self.body.named_children():
        for uname, unit in block.named_children():
          unit.load_from(weights, prefix=f'{prefix}{bname}/{uname}/')


KNOWN_MODELS = OrderedDict([
    ('BiT-M-R50x1', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 1, *a, **kw)),
    ('BiT-M-R50x3', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 3, *a, **kw)),
    ('BiT-M-R101x1', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 1, *a, **kw)),
    ('BiT-M-R101x3', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 3, *a, **kw)),
    ('BiT-M-R152x2', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 2, *a, **kw)),
    ('BiT-M-R152x4', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 4, *a, **kw)),
    ('BiT-S-R50x1', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 1, *a, **kw)),
    ('BiT-S-R50x3', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 3, *a, **kw)),
    ('BiT-S-R101x1', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 1, *a, **kw)),
    ('BiT-S-R101x3', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 3, *a, **kw)),
    ('BiT-S-R152x2', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 2, *a, **kw)),
    ('BiT-S-R152x4', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 4, *a, **kw)),
])
