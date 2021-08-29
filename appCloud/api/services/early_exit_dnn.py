import torch
import torch.nn as nn
#from utils import ExitBlock
from pthflops import count_ops
import torchvision.models as models
import numpy as np
from torch import Tensor
from typing import Callable, Any, Optional, List
import functools


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
    self.temp_overall = None


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

  @property
  def temperature_branches(self):
    return self._temperature_branches
  

  @temperature_branches.setter  
  def temperature_branches(self, temp_branches):
    self._temperature_branches = temp_branches

  
  def temperature_scale_branches(self, logits, exit_branch):
    temperature = self._temperature_branches[exit_branch].unsqueeze(1).expand(logits.size(0), logits.size(1)).to(self.device)
    return logits / temperature

  def forwardCloudInference(self, x, conf_list, class_list, p_tar, nr_branch_edge):

    for i, exitBlock in enumerate(self.exits[int(nr_branch_edge):], int(nr_branch_edge)):
        x = self.stages[i](x)

    x = self.stages[-1](x)
    
    x = torch.flatten(x, 1)

    output = self.classifier(x)
    output = self.temperature_scale_branches(output, -1)

    conf, infered_class = torch.max(self.softmax(output), 1)
    
    # Note that if confidence value is greater than a p_tar value, we terminate the dnn inference and returns the output
    # This also happens in the last exit
    if (conf.item() >= p_tar):
      return conf.item(), infered_class
    else:

      # If any exit can reach the p_tar value, the output is give by the more confidence output.
      # If evaluation, it returns max(output), max(conf) and the number of the early exit.

      conf_list.append(conf.item())
      class_list.append(infered_class)
      max_conf = np.argmax(conf_list)
      return conf_list[max_conf], class_list[max_conf]

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
