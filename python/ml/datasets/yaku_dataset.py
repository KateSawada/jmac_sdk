# -*- coding: utf-8 -*-

# Copyright 2023 KateSawada
# Copyright 2022 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Dataset modules.

References:
    - https://github.com/kan-bayashi/ParallelWaveGAN
    - https://github.com/bigpon/QPPWG

"""
import os
from logging import getLogger
from multiprocessing import Manager
import random

import numpy as np
from hydra.utils import to_absolute_path
from torch.utils.data import Dataset
from mjx import State

from ml.utils import read_txt

# A logger for this file
logger = getLogger(__name__)


class YakuDataset(Dataset):
    """PyTorch compatible mahjong yaku dataset."""

    def __init__(
        self,
        data_list,
        return_filename=False,
        allow_cache=False,
    ):
        """Initialize dataset.

        Args:
            data_list (str): Filename of the list of data directories.
                They contains wins.npy and four player's obs & action npy.
            return_filename (bool): Whether to return the filename with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.
        """
        self.data_list = read_txt(data_list)

        self.return_filename = return_filename
        self.allow_cache = allow_cache

        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(data_list))]


    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_filename = True).
            ndarray: obs feature array (shape=(133, 34))
            ndarray: achieved yaku one-hot array (shape=55)
        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        data_path = os.path.abspath(self.data_list[idx])


        yaku = np.load(os.path.join(data_path, "wins.npy"))
        won_yaku = yaku[np.any(yaku == 1, axis=1)][0]  # extract winner's yaku array. only one winner is chosen

        with open(os.path.join(data_path, "state.json")) as f:
            states = State(f.read()).past_decisions()

        obs, act = random.choice(states)
        obs = obs.to_features(feature_name="mjx-large-v0")

        if self.return_filename:
            items = (self.data_list[idx], obs, won_yaku)
        else:
            items = (obs, won_yaku)

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.data_list)
