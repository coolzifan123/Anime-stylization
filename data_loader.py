import torch
import numpy as np
import os
from PIL import Image
import glob
import torchvision
from torchvision import transforms
from torch.utils.data import RandomSampler, Sampler, DataLoader, TensorDataset, random_split, ConcatDataset
from typing import List, Sequence, Tuple

class Dataset(torch.utils.data.Dataset):
    def __init__(self,photo_path,cartoon_path,transform1=None,transform2=None):
        super(Dataset, self).__init__()
        self.transform1=transform1
        self.transform2=transform2

        self.photo = list(sorted(glob.glob(photo_path+ '/*.*')))
        self.cartoon= list(sorted(glob.glob(cartoon_path+ '/*.*')))
    def __len__(self):
        return len(self.cartoon)
        # return 8

    def __getitem__(self, item):
        cartoon_len=len(self.cartoon)
        photo_path=self.photo[item]
        cartoon_path=self.cartoon[item]
        photo = Image.open(photo_path).convert("RGB")
        cartoon = Image.open(cartoon_path).convert("RGB")
        if self.transform1 is not None:
            photo = self.transform1(photo)
        if self.transform2 is not None:
            cartoon = self.transform2(cartoon)
        return photo,cartoon

class photo_dataset(torch.utils.data.Dataset):
    def __init__(self,photo_path,transform):
        super(photo_dataset, self).__init__()
        self.transform=transform

        self.photo = list(sorted(glob.glob(photo_path+ '/*.*')))
    def __len__(self):
        return len(self.photo)

    def __getitem__(self, item):
        photo_path=self.photo[item]
        photo = Image.open(photo_path).convert("RGB")
        if self.transform is not None:
            photo = self.transform(photo)
        return photo


class ImageFolder(torch.utils.data.Dataset):
  def __init__(self, root, transform=None):
    super().__init__()
    self.transform=transform
    self.samples = list(sorted(glob.glob(root+ '/*.*')))

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(self, index: int):
    path = self.samples[index]
    sample = Image.open(path).convert("RGB")
    if self.transform is not None:
      sample = self.transform(sample)

    return sample

  def size(self, idx):
    return len(self.samples)


class MergeDataset(Dataset):
  def __init__(self, *tensors):
    """Merge two dataset to one Dataset
    """
    self.tensors = tensors
    self.sizes = [len(tensor) for tensor in tensors]

  def __getitem__(self, indexs: List[int]):
    return tuple(tensor[idx] for idx, tensor in zip(indexs, self.tensors))

  def __len__(self):
    return max(self.sizes)

class MultiRandomSampler(RandomSampler):
  def __init__(self, data_source: MergeDataset, replacement=True, num_samples=None, generator=None):
    """ a Random Sampler for MergeDataset. NOTE will padding all dataset to same length
    Args:
        data_source (MergeDataset): MergeDataset object
        replacement (bool, optional): shuffle index use replacement. Defaults to True.
        num_samples ([type], optional): Defaults to None.
        generator ([type], optional): Defaults to None.
    """
    self.data_source: MergeDataset = data_source
    self.replacement = replacement
    self._num_samples = num_samples
    self.generator = generator
    self.maxn = len(self.data_source)

  @property
  def num_samples(self):
    # dataset size might change at runtime
    if self._num_samples is None:
      self._num_samples = self.data_source.sizes
    return self._num_samples

  def __iter__(self):
    rands = []
    for size in self.num_samples:
      if self.maxn == size:
        rands.append(torch.randperm(size).tolist())
      else:
        rands.append(torch.randint(high=size, size=(self.maxn,),
                                   dtype=torch.int64, generator=self.generator).tolist())
    return zip(*rands)

  def __len__(self):
    return len(self.data_source)