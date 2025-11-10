#!/usr/bin/env python3

import time 
import logging
import os
import random
import torch
import torch.utils.data
from diff_utils.helpers import * 
import fnmatch
import random

import pandas as pd 
import numpy as np
import csv, json

from tqdm import tqdm

class ModulationLoader(torch.utils.data.Dataset):
    def __init__(self, data_path, context_path=None, split_file=None, pc_size=None):
        super().__init__()

        self.conditional = context_path is not None 

        if self.conditional:
            self.modulations, self.condition_paths = self.load_modulations(data_path, context_path, split_file)
        else:
            self.modulations = self.unconditional_load_modulations(data_path, split_file)
        #self.modulations = self.modulations[0:8]
        #context_paths = context_paths[0:8]

        print("data shape, dataset len: ", self.modulations[0].shape, len(self.modulations))
        #assert args.batch_size <= len(self.modulations)

        if self.conditional:
            assert len(self.condition_paths) == len(self.modulations)
        
        
    def __len__(self):
        return len(self.modulations)

    def __getitem__(self, index):

        if self.conditional:
            context = self.condition_paths[index]
            c = random.choice(context)
            return {
                "context" : c,
                "latent" : self.modulations[index]         
            }
        else:
            return {
                "context" : None,
                "latent" : self.modulations[index]         
            }
        

    def load_modulations(self, data_source, pc_source, split, f_name="latent.txt", add_flip_augment=False, return_filepaths=True):
        files = []
        filepaths = {}
        length = len(os.listdir(data_source))
        for idx in range(length):
            instance_filename = os.path.join(data_source, str(idx), f_name)
            files.append(torch.from_numpy(np.loadtxt(instance_filename)).float())
            tmp = []
            for filename in os.listdir(os.path.join(data_source, str(idx))):
                if fnmatch.fnmatch(filename, "text*"):
                    condition_filename = os.path.join(data_source, str(idx), filename)
                    tmp.append(torch.from_numpy(np.loadtxt(condition_filename)).unsqueeze(0).float())
            filepaths[idx] = tmp
        return files, filepaths
        

    def unconditional_load_modulations(self, data_source, split, f_name="latent.txt", add_flip_augment=False):
        files = []
        length = len(os.listdir(data_source))
        for idx in range(length):
            instance_filename = os.path.join(data_source, str(idx), f_name)
            files.append( torch.from_numpy(np.loadtxt(instance_filename)).float())
        return files