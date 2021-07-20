import torchvision.transforms as transforms
import torchvision, os, torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.utils import save_image
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, WeightedRandomSampler
from torch.utils.data import random_split
from PIL import Image
import numpy as np

class MapDataset(torch.utils.data.Dataset):
  def __init__(self, dataset, transformation):
    self.dataset = dataset
    self.transformation = transformation

  def __getitem__(self, index):
    x = self.transformation(self.dataset[index][0])
    y = self.dataset[index][1]
    return x, y

  def __len__(self):
    return len(self.dataset)



class LoadDataset():
  def __init__(self, input_dim, batch_size_test, savePath_idx_dataset=None, normalization=True):
    self.input_dim = input_dim
    self.batch_size_test = batch_size_test
    self.savePath_idx_dataset = savePath_idx_dataset

    mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
    std = [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

    transformation_valid_list = [transforms.Resize(330), 
                                          transforms.CenterCrop(300), 
                                          transforms.ToTensor()]
    
    if (normalization):
      transformation_train_list.append(transforms.Normalize(mean = mean, std = std))
      transformation_valid_list.append(transforms.Normalize(mean = mean, std = std))

        
    self.transformations_valid = transforms.Compose(transformation_valid_list)



  def caltech(self, root_path, split_train=0.8):

    dataset = datasets.ImageFolder(root_path)

    val_dataset = MapDataset(dataset, self.transformations_valid)


    if (self.savePath_idx_dataset is not None):
      data = np.load(self.savePath_idx_dataset, allow_pickle=True)
      train_idx, valid_idx = data[0], data[1]
      indices = list(range(len(valid_idx)))
      split = int(np.floor(0.5 * len(valid_idx)))
      valid_idx, test_idx = valid_idx[:split], valid_idx[split:]

    else:
      nr_samples = len(dataset)
      indices = list(range(nr_samples))
      split = int(np.floor(split_train * nr_samples))
      np.random.shuffle(indices)
      rain_idx, test_idx = indices[:split], indices[split:]


    test_data = torch.utils.data.Subset(val_dataset, indices=test_idx)

    testLoader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size_test, 
                                              num_workers=4)


    return testLoader
