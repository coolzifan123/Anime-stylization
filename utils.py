import torch
import torch.functional as F
import torch.nn as nn
import numpy as np
import cv2

def variation_loss(image: torch.Tensor, ksize=1):
  """
  A smooth loss in fact. Like the smooth prior in MRF.
  V(y) = || y_{n+1} - y_n ||_2
  """
  dh = image[:, :, :-ksize, :] - image[:, :, ksize:, :]
  dw = image[:, :, :, :-ksize] - image[:, :, :, ksize:]
  return (torch.mean(torch.abs(dh)) + torch.mean(torch.abs(dw)))


def rgb2yuv(rgb: torch.Tensor) -> torch.Tensor:
  """ rgb2yuv NOTE rgb image value range must in 0~1
  Args:
      rgb (torch.Tensor): 4D tensor , [b,c,h,w]
  Returns:
      torch.Tensor: 4D tensor, [b,h,w,c]
  """
  kernel = torch.tensor([[0.299, -0.14714119, 0.61497538],
                         [0.587, -0.28886916, -0.51496512],
                         [0.114, 0.43601035, -0.10001026]],
                        dtype=torch.float32, device=rgb.device)
  rgb = F.tensordot(rgb, kernel, [[1], [0]])
  return rgb


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def custom_blur_demo(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
    dst = cv2.filter2D(image, -1, kernel=kernel)
    return dst

def preprocessing(img, size):
    h, w = img.shape[:2]
    if h <= size[0]:
        h = size[0]
    else:
        x = h % 32
        h = h - x

    if w < size[1]:
        w = size[1]
    else:
        y = w % 32
        w = w - y
    # the cv2 resize func : dsize format is (W ,H)
    img = cv2.resize(img, (w, h))
    return img/127.5 - 1.0

