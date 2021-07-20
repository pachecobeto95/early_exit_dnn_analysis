import torchvision.transforms as transforms
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.utils import save_image
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, WeightedRandomSampler
from torch.utils.data import random_split
from PIL import Image
import torch, os
import numpy as np


class LoadDataset():
    def __init__(self, input_dim, batch_size_train, batch_size_test, seed=42):
        self.input_dim = input_dim
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.seed = seed

        #To normalize the input images data.
        mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
        std = [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

        # Note that we apply data augmentation in the training dataset.
        #You can change as you want.
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
            transforms.CenterCrop(input_dim), 
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

    def caltech_256(self, root_path, split_ratio):
        # This method loads the Caltech-256 dataset.

        #Setting the seeds.
        torch.manual_seed(self.seed)
        np.random.seed(seed=self.seed)

        # This block receives the dataset path and applies the transformation data. 
        train_set = datasets.ImageFolder(root_path, transform=self.transformations_train)

        val_set = datasets.ImageFolder(root_path, transform=self.transformations_test)
        test_set = datasets.ImageFolder(root_path, transform=self.transformations_test)

        # This line get the indices of the samples which belong to the training dataset and test dataset. 
        train_idx, test_idx = self.get_indices(train_set, split_ratio)

        # This line mounts the training and test dataset, selecting the samples according indices. 
        train_data = torch.utils.data.Subset(train_set, indices=train_idx)
        test_data = torch.utils.data.Subset(test_set, indices=test_idx)

        # This line gets the indices to split the train dataset into training dataset and validation dataset.
        train_idx, val_idx = self.get_indices(train_data, split_ratio)

        train_data = torch.utils.data.Subset(train_set, indices=train_idx)
        val_data = torch.utils.data.Subset(val_set, indices=val_idx)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size_train, shuffle=True, num_workers=4)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size_test, num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size_test, num_workers=4)

        return train_loader, val_loader, test_loader 

    def getDataset(self, root_path, dataset_name, split_ratio):
        self.dataset_name = dataset_name
        def func_not_found():
            print("No dataset %s is found"%(self.dataset_name))

        func_name = getattr(self, self.dataset_name, func_not_found)
        train_loader, val_loader, test_loader = func_name(root_path, split_ratio)
        return train_loader, val_loader, test_loader

parser = argparse.ArgumentParser(description='Evaluating DNNs perfomance using distorted image: blur ou gaussian noise')
parser.add_argument('--input_dim', type=int, help='Input width and height of the images')
parser.add_argument('--batch_size_train', type=int, help='The number of images within a batch used for training.')
parser.add_argument('--batch_size_test', type=int, default=1, 
    help='The number of images within a batch used for validation and test dataset.')
parser.add_argument('--split_ratio', type=float, help='The ratio to split training and validation datasets')
parser.add_argument('--dataset_root_path', type=float, 
    help='Path to the dataset. When cifar, dataset_root_path must be ./dataset_name')
parser.add_argument('--dataset_name', type=str, choices=["cifar_10", "cifar_100", "caltech_256"], 
    help='Name of the dataset')

args = parser.parse_args()
dataset = LoadDataset(args.input_dim, args.batch_size_train, args.batch_size_test)
dataset.getDataset(args.dataset_root_path, args.dataset_name, args.split_ratio)