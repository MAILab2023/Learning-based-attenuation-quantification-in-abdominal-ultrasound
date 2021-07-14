import torch
import argparse
from torch.utils.data import Dataset
from copy import deepcopy
from utils import mydataset, Experiment

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args("")
    dim = parser.parse_args("")
    pin_memory = True

    args.main_dir = r'/'
    args.data_dir = r'/'
    args.train_dir = r'dataset_train'
    args.test_dir = r'dataset_test'
    args.savefilename = 'RAN_ENV_AB'
    args.numberOfWorkers = 16
    args.train_batch_size = 24
    args.test_batch_size = 24
    args.epoch = 250

    device = torch.device("cuda:0")

    args.lr = 5e-6
    args.l2 = args.lr * 1e-2
    args.lambda1 = 0.0
    args.lambda2 = 5e-8

    print('Loading Dataset')
    trainset = mydataset.create_dataset(args.data_dir, args.train_dir)
    trainset, valset = torch.utils.data.random_split(trainset, [len(trainset) - int(400), int(400)])
    testset = mydataset.create_dataset(args.data_dir, args.test_dir)
    partition = {'train': trainset, 'val': valset, 'test': testset}

    print('Start Training')
    net, setting, result = Experiment.run_experiment(partition, deepcopy(args))

