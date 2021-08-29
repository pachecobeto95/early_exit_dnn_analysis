import torchvision.transforms as transforms
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.utils import save_image
import os, sys, time, math, os
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, WeightedRandomSampler
from torch.utils.data import random_split
from PIL import Image
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import product
import pandas as pd
import torchvision.models as models
from torchvision import datasets, transforms
from pthflops import count_ops
from torch import Tensor
from typing import Callable, Any, Optional, List
import functools

class LoadDataset():
  def __init__(self, input_dim, batch_size_train, batch_size_test, save_idx, model_id, seed=42):
    self.input_dim = input_dim
    self.batch_size_train = batch_size_train
    self.batch_size_test = batch_size_test
    self.seed = seed
    self.save_idx = save_idx
    self.model_id = model_id

    #To normalize the input images data.
    mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
    std = [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

    # Note that we apply data augmentation in the training dataset.
    self.transformations_train = transforms.Compose([transforms.Resize((input_dim, input_dim)),
                                                     transforms.RandomChoice([
                                                                              transforms.ColorJitter(brightness=(0.80, 1.20)),
                                                                              transforms.RandomGrayscale(p = 0.25)]),
                                                     transforms.RandomHorizontalFlip(p = 0.25),
                                                     transforms.RandomRotation(25),
                                                     transforms.ToTensor(), 
                                                     transforms.Normalize(mean = mean, std = std),
                                                     ])

    # Note that we do not apply data augmentation in the test dataset.
    self.transformations_test = transforms.Compose([
                                                     transforms.Resize(input_dim), 
                                                     transforms.ToTensor(), 
                                                     transforms.Normalize(mean = mean, std = std),
                                                     ])

  def cifar_10(self, root_path, split_ratio):
    # This method loads Cifar-10 dataset. 
    
    # saves the seed
    torch.manual_seed(self.seed)

    # This downloads the training and test CIFAR-10 datasets and also applies transformation  in the data.
    train_set = datasets.CIFAR10(root=root_path, train=True, download=True, transform=self.transformations_train)
    test_set = datasets.CIFAR10(root=root_path, train=False, download=True, transform=self.transformations_test)

    classes_list = train_set.classes

    # This line defines the size of validation dataset.
    val_size = int(split_ratio*len(train_set))

    # This line defines the size of training dataset.
    train_size = int(len(train_set) - val_size)

    #This line splits the training dataset into train and validation, according split ratio provided as input.
    train_dataset, val_dataset = random_split(train_set, [train_size, val_size])

    #This block creates data loaders for training, validation and test datasets.
    train_loader = DataLoader(train_dataset, self.batch_size_train, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, self.batch_size_test, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, self.batch_size_test, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader

  def cifar_100(self, root_path, split_ratio):
    # This method loads Cifar-100 dataset
    root = "cifar_100"
    torch.manual_seed(self.seed)
    #troquei o savePath_idx pelo root_path, conferir dps
    #conferfir--------------------------------------------------------------------------------------------------
    if (os.path.exists(os.path.join(root_path, "train"))):#ing_idx_caltech256_id_%s.npy"%(self.model_id)))):
      train_idx = np.load(os.path.join(root_path, 'train'))#"training_idx_caltech256_id_%s.npy"%(self.model_id)))
      #val_idx = np.load(os.path.join(savePath_idx, "validation_idx_caltech256_id_%s.npy"%(self.model_id)))
      test_idx = np.load(os.path.join(root_path, 'test'))#"test_idx_caltech256_id_%s.npy"%(self.model_id)))
    #-----------------------------------------------------------------------------------------------------------
    else:
      # This downloads the training and test Cifar-100 datasets and also applies transformation  in the data.
      train_set = datasets.CIFAR100(root=root_path, train=True, download=True, transform=self.transformations_train)
      test_set = datasets.CIFAR100(root=root_path, train=False, download=True, transform=self.transformations_train)

    classes_list = train_set.classes

    # This line defines the size of validation dataset.
    val_size = int(split_ratio*len(train_set))

    # This line defines the size of training dataset.
    train_size = int(len(train_set) - val_size)

    #This line splits the training dataset into train and validation, according split ratio provided as input.
    train_dataset, val_dataset = random_split(train_set, [train_size, val_size])

    #This block creates data loaders for training, validation and test datasets.
    train_loader = DataLoader(train_dataset, self.batch_size_train, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, self.batch_size_test, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, self.batch_size_test, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader
  
  def get_indices(self, dataset, split_ratio):
    nr_samples = len(dataset)
    indices = list(range(nr_samples))
    
    train_size = nr_samples - int(np.floor(split_ratio * nr_samples))

    np.random.shuffle(indices)

    train_idx, test_idx = indices[:train_size], indices[train_size:]

    return train_idx, test_idx

  def caltech_256(self, root_path, split_ratio, savePath_idx):
    # This method loads the Caltech-256 dataset.

    torch.manual_seed(self.seed)
    np.random.seed(seed=self.seed)

    # This block receives the dataset path and applies the transformation data. 
    train_set = datasets.ImageFolder(root_path, transform=self.transformations_train)

    val_set = datasets.ImageFolder(root_path, transform=self.transformations_test)
    test_set = datasets.ImageFolder(root_path, transform=self.transformations_test)

    if (os.path.exists(os.path.join(savePath_idx, "training_idx_caltech256_id_%s.npy"%(self.model_id)))):
      
      train_idx = np.load(os.path.join(savePath_idx, "training_idx_caltech256_id_%s.npy"%(self.model_id)))
      val_idx = np.load(os.path.join(savePath_idx, "validation_idx_caltech256_id_%s.npy"%(self.model_id)))
      test_idx = np.load(os.path.join(savePath_idx, "test_idx_caltech256_id_%s.npy"%(self.model_id)))

    else:

      # This line get the indices of the samples which belong to the training dataset and test dataset. 
      train_idx, test_idx = self.get_indices(train_set, split_ratio)

      # This line mounts the training and test dataset, selecting the samples according indices. 
      train_data = torch.utils.data.Subset(train_set, indices=train_idx)
      ##essa linha parecia estar faltando. copiei da versão anterior##

      # This line gets the indices to split the train dataset into training dataset and validation dataset.
      train_idx, val_idx = self.get_indices(train_data, split_ratio)

      np.save(os.path.join(savePath_idx, "traning_idx_caltech256_id_%s.npy"%(self.model_id)), train_idx)
      np.save(os.path.join(savePath_idx, "validation_idx_caltech256_id_%s.npy"%(self.model_id)), val_idx)
      np.save(os.path.join(savePath_idx, "test_idx_caltech256_id_%s.npy"%(self.model_id)), test_idx)

    # This line mounts the training and test dataset, selecting the samples according indices. 
    train_data = torch.utils.data.Subset(train_set, indices=train_idx)
    val_data = torch.utils.data.Subset(val_set, indices=val_idx)
    test_data = torch.utils.data.Subset(test_set, indices=test_idx)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size_train, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size_test, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size_test, num_workers=4)

    return train_loader, val_loader, test_loader 

  def getDataset(self, root_path, dataset_name, split_ratio, savePath_idx):
    self.dataset_name = dataset_name
    def func_not_found():
      print("No dataset %s is found"%(self.dataset_name))

    func_name = getattr(self, self.dataset_name, func_not_found)
    train_loader, val_loader, test_loader = func_name(root_path, split_ratio, savePath_idx)
    return train_loader, val_loader, test_loader


def load_early_exit_dnn_model(model, model_path, device):
  
  model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])

  return model

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
  """3x3 convolution with padding"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

class BasicBlock(nn.Module):
  """Basic Block defition.
  Basic 3X3 convolution blocks for use on ResNets with layers <= 34.
  Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
  """
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                      bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        self.out_channels = out_planes


# necessary for backwards compatibility
ConvBNReLU = ConvBNActivation


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class EarlyExitBlock(nn.Module):
  """
  This EarlyExitBlock allows the model to terminate early when it is confident for classification.
  """
  def __init__(self, input_shape, n_classes, exit_type, device):
    super(EarlyExitBlock, self).__init__()
    self.input_shape = input_shape

    _, channel, width, height = input_shape
    self.expansion = width * height if exit_type == 'plain' else 1

    self.layers = nn.ModuleList()

    if (exit_type == 'bnpool'):
      self.layers.append(nn.BatchNorm2d(channel))

    if (exit_type != 'plain'):
      self.layers.append(nn.AdaptiveAvgPool2d(1))
    
    #This line defines the data shape that fully-connected layer receives.
    current_channel, current_width, current_height = self.get_current_data_shape()

    self.layers = self.layers.to(device)

    #This line builds the fully-connected layer
    self.classifier = nn.Sequential(nn.Linear(current_channel*current_width*current_height, n_classes)).to(device)

    self.softmax_layer = nn.Softmax(dim=1)


  def get_current_data_shape(self):
    _, channel, width, height = self.input_shape
    temp_layers = nn.Sequential(*self.layers)

    input_tensor = torch.rand(1, channel, width, height)
    _, output_channel, output_width, output_height = temp_layers(input_tensor).shape
    return output_channel, output_width, output_height
        
  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    x = x.view(x.size(0), -1)
    output = self.classifier(x)
    #confidence = self.softmax_layer()
    return output

def conv1x1(in_planes, out_planes, stride=1):
  """1x1 convolution"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Early_Exit_DNN(nn.Module):
  def __init__(self, model_name: str, n_classes: int, 
               pretrained: bool, n_branches: int, input_shape:tuple, 
               exit_type: str, device, distribution="linear"):
    super(Early_Exit_DNN, self).__init__()

    """
    This classes builds an early-exit DNNs architectures
    Args:

    model_name: model name 
    n_classes: number of classes in a classification problem, according to the dataset
    pretrained: 
    n_branches: number of branches (early exits) inserted into middle layers
    input_shape: shape of the input image
    exit_type: type of the exits
    distribution: distribution method of the early exit blocks.
    device: indicates if the model will processed in the cpu or in gpu
    
    Note: the term "backbone model" refers to a regular DNN model, considering no early exits.

    """
    self.model_name = model_name
    self.n_classes = n_classes
    self.pretrained = pretrained
    self.n_branches = n_branches
    self.input_shape = input_shape
    self.exit_type = exit_type
    self.distribution = distribution
    self.device = device
    self.channel, self.width, self.height = input_shape
    self._temperature_branches = None



    build_early_exit_dnn = self.select_dnn_architecture_model()

    build_early_exit_dnn()

  def select_dnn_architecture_model(self):
    """
    This method selects the backbone to insert the early exits.
    """

    architecture_dnn_model_dict = {"alexnet": self.early_exit_alexnet,
                                   "mobilenet": self.early_exit_mobilenet,
                                   "resnet18": self.early_exit_resnet18,
                                   "resnet34": self.early_exit_resnet34}

    return architecture_dnn_model_dict.get(self.model_name, self.invalid_model)

  def select_distribution_method(self):
    """
    This method selects the distribution method to insert early exits into the middle layers.
    """
    distribution_method_dict = {"linear":self.linear_distribution,
                                "pareto":self.paretto_distribution,
                                "fibonacci":self.fibo_distribution}
    return distribution_method_dict.get(self.distribution, self.invalid_distribution)
    
  def linear_distribution(self, i):
    """
    This method defines the Flops to insert an early exits, according to a linear distribution.
    """
    flop_margin = 1.0 / (self.n_branches+1)
    return self.total_flops * flop_margin * (i+1)

  def paretto_distribution(self, i):
    """
    This method defines the Flops to insert an early exits, according to a pareto distribution.
    """
    return self.total_flops * (1 - (0.8**(i+1)))

  def fibo_distribution(self, i):
    """
    This method defines the Flops to insert an early exits, according to a fibonacci distribution.
    """
    gold_rate = 1.61803398875
    return total_flops * (gold_rate**(i - self.num_ee))

  def verifies_nr_exits(self, backbone_model):
    """
    This method verifies if the number of early exits provided is greater than a number of layers in the backbone DNN model.
    """
    
    total_layers = len(list(backbone_model.children()))
    if (self.n_branches >= total_layers):
      raise Exception("The number of early exits is greater than number of layers in the DNN backbone model.")

  def countFlops(self, model):
    """
    This method counts the numper of Flops in a given full DNN model or intermediate DNN model.
    """
    input = torch.rand(1, self.channel, self.width, self.height)
    flops, all_data = count_ops(model, input, print_readable=False, verbose=False)
    return flops

  def where_insert_early_exits(self):
    """
    This method defines where insert the early exits, according to the dsitribution method selected.
    Args:

    total_flops: Flops of the backbone (full) DNN model.
    """
    threshold_flop_list = []
    distribution_method = self.select_distribution_method()

    for i in range(self.n_branches):
      threshold_flop_list.append(distribution_method(i))

    return threshold_flop_list

  def invalid_model(self):
    raise Exception("This DNN model has not implemented yet.")
  def invalid_distribution(self):
    raise Exception("This early-exit distribution has not implemented yet.")

  def is_suitable_for_exit(self):
    """
    This method answers the following question. Is the position to place an early exit?
    """
    intermediate_model = nn.Sequential(*(list(self.stages)+list(self.layers)))
    current_flop = self.countFlops(intermediate_model)
    return self.stage_id < self.n_branches and current_flop >= self.threshold_flop_list[self.stage_id]

  def add_exit_block(self):
    """
    This method adds an early exit in the suitable position.
    """
    input_tensor = torch.rand(1, self.channel, self.width, self.height)

    self.stages.append(nn.Sequential(*self.layers))

    feature_shape = nn.Sequential(*self.stages)(input_tensor).shape

    self.exits.append(EarlyExitBlock(feature_shape, self.n_classes, self.exit_type, self.device).to(self.device))
    self.layers = nn.ModuleList()
    self.stage_id += 1    

  def set_device(self):
    """
    This method sets the device that will run the DNN model.
    """

    self.stages.to(self.device)
    self.exits.to(self.device)
    self.layers.to(self.device)
    self.classifier.to(self.device)


  def early_exit_alexnet(self):
    """
    This method inserts early exits into a Alexnet model
    """

    self.stages = nn.ModuleList()
    self.exits = nn.ModuleList()
    self.layers = nn.ModuleList()
    self.cost = []
    self.stage_id = 0

    # Loads the backbone model. In other words, Alexnet architecture provided by Pytorch.
    backbone_model = models.alexnet(self.pretrained)

    # It verifies if the number of early exits provided is greater than a number of layers in the backbone DNN model.
    self.verifies_nr_exit_alexnet(backbone_model.features)
    
    # This obtains the flops total of the backbone model
    self.total_flops = self.countFlops(backbone_model)

    # This line obtains where inserting an early exit based on the Flops number and accordint to distribution method
    self.threshold_flop_list = self.where_insert_early_exits()

    for layer in backbone_model.features:
      self.layers.append(layer)
      if (isinstance(layer, nn.ReLU)) and (self.is_suitable_for_exit()):
        self.add_exit_block()

    
    
    self.layers.append(nn.AdaptiveAvgPool2d(output_size=(6, 6)))
    self.stages.append(nn.Sequential(*self.layers))

    
    self.classifier = backbone_model.classifier
    self.classifier[6] = nn.Linear(in_features=4096, out_features=self.n_classes, bias=True)
    self.softmax = nn.Softmax(dim=1)
    self.set_device()

  def verifies_nr_exit_alexnet(self, backbone_model):
    """
    This method verifies if the number of early exits provided is greater than a number of layers in the backbone DNN model.
    In AlexNet, we consider a convolutional block composed by: Convolutional layer, ReLU and he Max-pooling layer.
    Hence, we consider that it makes no sense to insert side branches between these layers or only after the convolutional layer.
    """

    count_relu_layer = 0
    for layer in backbone_model:
      if (isinstance(layer, nn.ReLU)):
        count_relu_layer += 1

    if (count_relu_layer > self.n_branches):
      raise Exception("The number of early exits is greater than number of layers in the DNN backbone model.")

  def early_exit_resnet18(self):
    """
    This method inserts early exits into a Resnet18 model
    """

    self.stages = nn.ModuleList()
    self.exits = nn.ModuleList()
    self.layers = nn.ModuleList()
    self.cost = []
    self.stage_id = 0

    self.inplanes = 64

    n_blocks = 4

    backbone_model = models.resnet18(self.pretrained)

    # It verifies if the number of early exits provided is greater than a number of layers in the backbone DNN model.
    self.verifies_nr_exits(backbone_model)

    # This obtains the flops total of the backbone model
    self.total_flops = self.countFlops(backbone_model)

    # This line obtains where inserting an early exit based on the Flops number and accordint to distribution method
    self.threshold_flop_list = self.where_insert_early_exits()

    building_first_layer = ["conv1", "bn1", "relu", "maxpool"]
    for layer in building_first_layer:
      self.layers.append(getattr(backbone_model, layer))

    if (self.is_suitable_for_exit()):
      self.add_exit_block()

    for i in range(1, n_blocks+1):
      
      block_layer = getattr(backbone_model, "layer%s"%(i))

      for l in block_layer:
        self.layers.append(l)

        if (self.is_suitable_for_exit()):
          self.add_exit_block()
    
    self.layers.append(nn.AdaptiveAvgPool2d(1))
    self.classifier = nn.Sequential(nn.Linear(512, self.n_classes))
    self.stages.append(nn.Sequential(*self.layers))
    self.softmax = nn.Softmax(dim=1)
    self.set_device()

  def early_exit_resnet34(self):
    return True
  

  def early_exit_mobilenet(self):
    """
    This method inserts early exits into a Mobilenet V2 model
    """

    self.stages = nn.ModuleList()
    self.exits = nn.ModuleList()
    self.layers = nn.ModuleList()
    self.cost = []
    self.stage_id = 0

    last_channel = 1280
    
    # Loads the backbone model. In other words, Mobilenet architecture provided by Pytorch.
    backbone_model = models.mobilenet_v2(self.pretrained)

    # It verifies if the number of early exits provided is greater than a number of layers in the backbone DNN model.
    self.verifies_nr_exits(backbone_model.features)
    
    # This obtains the flops total of the backbone model
    self.total_flops = self.countFlops(backbone_model)

    # This line obtains where inserting an early exit based on the Flops number and accordint to distribution method
    self.threshold_flop_list = self.where_insert_early_exits()

    for i, layer in enumerate(backbone_model.features.children()):
      
      self.layers.append(layer)    
      if (self.is_suitable_for_exit()):
        self.add_exit_block()

    self.layers.append(nn.AdaptiveAvgPool2d(1))
    self.stages.append(nn.Sequential(*self.layers))
    

    self.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(last_channel, self.n_classes),)

    self.set_device()
    self.softmax = nn.Softmax(dim=1)

  @property
  def temperature_branches(self):
    return self._temperature_branches
  

  @temperature_branches.setter  
  def temperature_branches(self, temp_branches):
    self._temperature_branches = temp_branches
  
  def forwardTrain(self, x):
    """
    This method is used to train the early-exit DNN model
    """
    
    output_list, conf_list, class_list  = [], [], []

    for i, exitBlock in enumerate(self.exits):
      
      x = self.stages[i](x)
      output_branch = exitBlock(x)
      output_list.append(output_branch)

      #Confidence is the maximum probability of belongs one of the predefined classes and inference_class is the argmax
      conf, infered_class = torch.max(self.softmax(output_branch), 1)
      conf_list.append(conf)
      class_list.append(infered_class)

    x = self.stages[-1](x)

    x = torch.flatten(x, 1)

    output = self.classifier(x)
    infered_conf, infered_class = torch.max(self.softmax(output), 1)
    output_list.append(output)
    conf_list.append(infered_conf)
    class_list.append(infered_class)

    return output_list, conf_list, class_list

  def temperature_scale_overall(self, logits, temp_overall):
    temperature = temp_overall.unsqueeze(1).expand(logits.size(0), logits.size(1)).to(self.device)
    return logits / temperature

  def temperature_scale_branches(self, logits, exit_branch):
    temperature = self._temperature_branches[exit_branch].unsqueeze(1).expand(logits.size(0), logits.size(1)).to(self.device)
    return logits / temperature

  def forward_inference_calib_overall(self, x, p_tar, temp_overall):
    """
    This method is used to experiment of early-exit DNNs with overall calibration.
    """
    output_list, conf_list, class_list  = [], [], []
    n_exits = self.n_branches + 1
    exit_branches = np.zeros(n_exits)
    wasClassified = False

    for i, exitBlock in enumerate(self.exits):
      x = self.stages[i](x)
      output_branch = exitBlock(x)
      output_branch = self.temperature_scale_overall(output_branch, temp_overall)

      conf_branch, infered_class_branch = torch.max(self.softmax(output_branch), 1)
      conf_list.append(conf_branch.item()), class_list.append(infered_class_branch)

      if (conf_branch.item() >= p_tar):
        exit_branches[i] = 1

        if (not wasClassified):
          actual_exit_branch = i
          actual_conf = conf_branch.item()
          actual_inferred_class = infered_class_branch
          wasClassified = True

    x = self.stages[-1](x)
    
    x = torch.flatten(x, 1)

    output = self.classifier(x)
    output = self.temperature_scale_overall(output, temp_overall)

    conf, infered_class = torch.max(self.softmax(output), 1)
    conf_list.append(conf.item()), class_list.append(infered_class)

    exit_branches[-1] = 1

    if (conf.item() <  p_tar):
      max_conf = np.argmax(conf_list)
      conf_list[-1] = conf_list[max_conf]
      class_list[-1] = class_list[max_conf]

    if (not wasClassified):
      actual_exit_branch = self.n_branches
      actual_conf = conf_list[-1]
      actual_inferred_class = class_list[-1]

    return actual_conf, actual_inferred_class, actual_exit_branch, conf_list, class_list, exit_branches

  def forward_inference_calib_branches(self, x, p_tar, temp_branches):
    """
    This method is used to experiment of early-exit DNNs with calibration in all the branches.
    """

    output_list, conf_list, class_list  = [], [], []
    n_exits = self.n_branches + 1
    exit_branches = np.zeros(n_exits)
    wasClassified = False

    for i, exitBlock in enumerate(self.exits):
      x = self.stages[i](x)
      output_branch = exitBlock(x)
      output_branch = self.temperature_scale_branches(output_branch, temp_branches, i)

      conf_branch, infered_class_branch = torch.max(self.softmax(output_branch), 1)
      conf_list.append(conf_branch.item()), class_list.append(infered_class_branch)

      if (conf_branch.item() >= p_tar):
        exit_branches[i] = 1

        if (not wasClassified):
          actual_exit_branch = i
          actual_conf = conf_branch.item()
          actual_inferred_class = infered_class_branch
          wasClassified = True

    x = self.stages[-1](x)
    
    x = torch.flatten(x, 1)

    output = self.classifier(x)
    output = self.temperature_scale_branches(output, temp_branches, -1)
    conf, infered_class = torch.max(self.softmax(output), 1)
    conf_list.append(conf.item()), class_list.append(infered_class)

    exit_branches[-1] = 1

    if (conf.item() <  p_tar):
      max_conf = np.argmax(conf_list)
      conf_list[-1] = conf_list[max_conf]
      class_list[-1] = class_list[max_conf]

    if (not wasClassified):
      actual_exit_branch = self.n_branches
      actual_conf = conf_list[-1]
      actual_inferred_class = class_list[-1]

    return actual_conf, actual_inferred_class, actual_exit_branch, conf_list, class_list, exit_branches

  def forward_inference_test(self, x, p_tar=0.5):
    """
    This method is used to experiment of early-exit DNNs.
    """
    output_list, conf_list, class_list  = [], [], []
    n_exits = self.n_branches + 1
    exit_branches = np.zeros(n_exits)
    wasClassified = False

    for i, exitBlock in enumerate(self.exits):
      x = self.stages[i](x)

      output_branch = exitBlock(x)
      conf_branch, infered_class_branch = torch.max(self.softmax(output_branch), 1)
      conf_list.append(conf_branch.item()), class_list.append(infered_class_branch)

      if (conf_branch.item() >= p_tar):
        exit_branches[i] = 1

        if (not wasClassified):
          actual_exit_branch = i
          actual_conf = conf_branch.item()
          actual_inferred_class = infered_class_branch
          wasClassified = True

    x = self.stages[-1](x)
    
    x = torch.flatten(x, 1)

    output = self.classifier(x)
    conf, infered_class = torch.max(self.softmax(output), 1)
    conf_list.append(conf.item()), class_list.append(infered_class)

    exit_branches[-1] = 1

    if (conf.item() <  p_tar):
      max_conf = np.argmax(conf_list)
      conf_list[-1] = conf_list[max_conf]
      class_list[-1] = class_list[max_conf]

    if (not wasClassified):
      actual_exit_branch = self.n_branches
      actual_conf = conf_list[-1]
      actual_inferred_class = class_list[-1]

    return actual_conf, actual_inferred_class, actual_exit_branch, conf_list, class_list, exit_branches


  def forwardEval(self, x, p_tar):
    """
    This method is used to train the early-exit DNN model
    """
    output_list, conf_list, class_list  = [], [], []

    for i, exitBlock in enumerate(self.exits):
      x = self.stages[i](x)

      output_branch = exitBlock(x)
      conf, infered_class = torch.max(self.softmax(output_branch), 1)

      # Note that if confidence value is greater than a p_tar value, we terminate the dnn inference and returns the output
      if (conf.item() >= p_tar):
        return output_branch, conf.item(), infered_class, i

      else:
        output_list.append(output_branch)
        conf_list.append(conf.item())
        class_list.append(infered_class)

    x = self.stages[-1](x)
    
    x = torch.flatten(x, 1)

    output = self.classifier(x)
    conf, infered_class = torch.max(self.softmax(output), 1)
    
    # Note that if confidence value is greater than a p_tar value, we terminate the dnn inference and returns the output
    # This also happens in the last exit
    if (conf.item() >= p_tar):
      return output, conf.item(), infered_class, self.n_branches
    else:

      # If any exit can reach the p_tar value, the output is give by the more confidence output.
      # If evaluation, it returns max(output), max(conf) and the number of the early exit.

      conf_list.append(conf.item())
      class_list.append(infered_class)
      output_list.append(output)
      max_conf = np.argmax(conf_list)
      return output_list[max_conf], conf_list[max_conf], class_list[max_conf], self.n_branches


  def forward(self, x, p_tar=0.5, training=True):
    """
    This implementation supposes that, during training, this method can receive a batch containing multiple images.
    However, during evaluation, this method supposes an only image.
    """
    if (training):
      return self.forwardTrain(x)
    else:
      return self.forwardEval(x, p_tar)



def getTemperatureBranches(temperature_path, n_exits, p_tar=0.8):
  df_temp = pd.read_csv(temperature_path)
  df_temp = df_temp.loc[:, ~df_temp.columns.str.contains('^Unnamed')]
  df_temp = df_temp[df_temp.p_tar==p_tar]
  temp_list = [nn.Parameter(torch.tensor([df_temp["temperature_branch_%s"%(i)].unique().item()])) for i in range(1, n_exits + 1)]
  return temp_list

def measuring_processing_time_branches(model, x, p_tar):
  processing_time_branches = np.zeros(len(model.exits) + 1)
  conf_list, class_list = [], []

  for i, exitBlock in enumerate(model.exits):
    start = time.time()
    x = model.stages[i](x)
    output_branch = model.temperature_scale_branches(exitBlock(x), i)

    conf, infered_class = torch.max(model.softmax(output_branch), 1)
    
    if (conf.item() >= p_tar):
      proc_time = time.time() - start
      processing_time_branches[i] = proc_time
      return processing_time_branches, i
    
    else:
      proc_time = time.time() - start
      processing_time_branches[i] = proc_time
      conf_list.append(conf.item())
      class_list.append(infered_class)

  start = time.time()
  x = model.stages[-1](x)
    
  x = torch.flatten(x, 1)
  output = model.temperature_scale_branches(model.classifier(x), -1)

  conf, infered_class = torch.max(model.softmax(output), 1)
    
  # Note that if confidence value is greater than a p_tar value, we terminate the dnn inference and returns the output
  # This also happens in the last exit
  if (conf.item() >= p_tar):
    proc_time = time.time() - start
    processing_time_branches[-1] = proc_time
    return processing_time_branches, model.n_branches
    
  else:

    # If any exit can reach the p_tar value, the output is give by the more confidence output.
    # If evaluation, it returns max(output), max(conf) and the number of the early exit.

    conf_list.append(conf.item())
    class_list.append(infered_class)
    max_conf = np.argmax(conf_list)
    proc_time = time.time() - start
    processing_time_branches[-1] = proc_time
    return processing_time_branches, model.n_branches




def measuring_processing_time_total(model, x, p_tar):
  conf_list, class_list = [], []
  for i, exitBlock in enumerate(model.exits):
    start = time.time()

    x = model.stages[i](x)
    output_branch = model.temperature_scale_branches(exitBlock(x), i)


    conf, infered_class = torch.max(model.softmax(output_branch), 1)
    
    if (conf.item() >= p_tar):
      proc_time_total = time.time() - start
      return proc_time_total
    else:
      conf_list.append(conf.item())
      class_list.append(infered_class)


  x = model.stages[-1](x)
    
  x = torch.flatten(x, 1)

  output = model.temperature_scale_branches(model.classifier(x), -1)
  conf, infered_class = torch.max(model.softmax(output), 1)
    
  # Note that if confidence value is greater than a p_tar value, we terminate the dnn inference and returns the output
  # This also happens in the last exit
  if (conf.item() >= p_tar):
    proc_time_total = time.time() - start
    return proc_time_total
    
  else:

    # If any exit can reach the p_tar value, the output is give by the more confidence output.
    # If evaluation, it returns max(output), max(conf) and the number of the early exit.

    conf_list.append(conf.item())
    class_list.append(infered_class)
    max_conf = np.argmax(conf_list)
    proc_time_total = time.time() - start
    return proc_time_total


def measuring_processing_time_branches_acc(model, x, p_tar):
  processing_time_acc = np.zeros(len(model.exits) + 1)  
  conf_list, class_list = [], []
  proc_acc = 0
  for i, exitBlock in enumerate(model.exits):
    start = time.time()
    x = model.stages[i](x)
    output_branch = model.temperature_scale_branches(exitBlock(x), i)

    conf, infered_class = torch.max(model.softmax(output_branch), 1)

    if (conf.item() >= p_tar):
      proc_acc += time.time() - start
      processing_time_acc[i] = proc_acc
      return processing_time_acc

    else:
      proc_acc += time.time() - start
      processing_time_acc[i] = proc_acc
      conf_list.append(conf.item())
      class_list.append(infered_class)

  start = time.time()
  x = model.stages[-1](x)
    
  x = torch.flatten(x, 1)

  output = model.temperature_scale_branches(model.classifier(x), -1)
  conf, infered_class = torch.max(model.softmax(output), 1)
    
  # Note that if confidence value is greater than a p_tar value, we terminate the dnn inference and returns the output
  # This also happens in the last exit
  if (conf.item() >= p_tar):
    proc_acc += time.time() - start
    processing_time_acc[-1] = proc_acc

    return processing_time_acc
    
  else:

    # If any exit can reach the p_tar value, the output is give by the more confidence output.
    # If evaluation, it returns max(output), max(conf) and the number of the early exit.

    conf_list.append(conf.item())
    class_list.append(infered_class)
    max_conf = np.argmax(conf_list)

    return processing_time_acc

def experiment_proc_time(model, test_loader, p_tar, n_branches, device):
  n_exits = n_branches + 1
  processing_time_branches_list, processing_time_total_list, processing_time_acc_list = [], [], []

  model.eval()
  with torch.no_grad():
    for i, (data, target) in enumerate(test_loader, 1):
      print("Imagem: %s/%s"%(i, len(test_loader)))
      data, target = data.to(device), target.float().to(device)
      
      processing_time_branches, exit_branch = measuring_processing_time_branches(model, data, p_tar)
      processing_total = measuring_processing_time_total(model, data, p_tar)
      processing_time_acc = measuring_processing_time_branches_acc(model, data, p_tar)
      
      processing_time_branches_list.append(processing_time_branches)
      processing_time_total_list.append(processing_total)
      processing_time_acc_list.append(processing_time_acc)

      del data, target
      torch.cuda.empty_cache()

  result_samples = {"p_tar":len(processing_time_total_list)*[p_tar], 
                    "processing_time_total": processing_time_total_list}

  processing_time_branches_list = np.array(processing_time_branches_list)
  processing_time_acc_list = np.array(processing_time_acc_list)

  for i in range(n_exits):
    result_samples.update({"tp_%s"%(i+1): processing_time_branches_list[:, i],
                           "tp_acc_%s"%(i+1): processing_time_acc_list[:, i]})
  return result_samples

def exp_proc_time(model, test_loader, p_tar_list, n_branches, device, save_path, temperature_path):
  df_result_samples = pd.DataFrame()
  n_exits = n_branches + 1

  for p_tar in p_tar_list:
    print("P_tar: %s"%(p_tar))
    temperature_branches = getTemperatureBranches(temperature_path, n_exits)
    model.temperature_branches = temperature_branches
    result_samples = experiment_proc_time(model, test_loader, p_tar, n_branches, device)
    df_result = pd.read_csv(save_path) if (os.path.exists(save_path)) else pd.DataFrame()
    df_result = df_result.append(pd.Series(result_samples), ignore_index=True)
    df_result.to_csv(save_path)
        

model_name = "mobilenet"
dataset_name = "caltech256"
model_id = 2
img_dim = 300
input_dim = 300
batch_size_train, batch_size_test = 64, 1
split_ratio = 0.1
save_idx = False

exp_dir_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(os.path.dirname(exp_dir_path), "datasets", "256_ObjectCategories") 
save_idx_path = os.path.join(exp_dir_path, "dataset_indices")

dataset = LoadDataset(img_dim, batch_size_train, batch_size_test, save_idx, model_id)
_, val_loader, test_loader = dataset.caltech_256(dataset_path, split_ratio, save_idx_path)

n_classes = 258
pretrained = True
n_branches = 5
img_dim = 300
input_dim = 300
model_id = 2
n_exits = n_branches + 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_shape = (3, input_dim, input_dim)
distribution = "linear"
exit_type = "bnpool"
model_name = "mobilenet"


# this line indicates the path to trained early-exit DNN model. You must change it to yours trained model
#model_path = "./drive/MyDrive/project_quality_magazine/caltech256/mobilenet/models/pristine_model_mobilenet_caltech256_3_5_b.pth"
#model_path = "./drive/MyDrive/project_quality_magazine/caltech256/mobilenet/pristine_model_mobilenet_caltech256_3_5_b.pth"
model_path = os.path.join(os.path.dirname(exp_dir_path), "appEdge", "api", "services", "models", "%s_%s_%s_branches_%s.pth"%(model_name, dataset_name, model_id, n_branches)) #caltech

early_exit_model = Early_Exit_DNN(model_name, n_classes, pretrained, n_branches, input_shape, exit_type, device, distribution=distribution)
early_exit_model = early_exit_model.to(device)
early_exit_model.exits.to(device)

# this line loads the trained model to the early_exit_model.
early_exit_model = load_early_exit_dnn_model(early_exit_model, model_path, device)

temperature_path = os.path.join(os.path.dirname(exp_dir_path), "appEdge", "api", "services", "temperature", "branches_temp_scaling_branches_%s_2.csv"%(n_branches))
save_path = os.path.join(os.path.dirname(exp_dir_path), "appEdge", "api", "services", "results", "processing_time_eval_dataset_branches_%s.csv"%(n_branches))

p_tar_list = [0.7, 0.75, 0.8, 0.85, 0.9]

exp_proc_time(early_exit_model, test_loader, p_tar_list, n_branches, device, save_path, temperature_path)