"""
Some utils related to the dataset.
"""

import os
import json
import math

import torch
from torch.utils.data import Dataset
import torchaudio

import pandas as pd
import numpy as np

from utils.anc_utils import add_white_noise


def minmaxscaler(data):
    min = data.min()
    max = data.max()
    return (data) / (max - min)


def minmaxscaler_frames(data):
    for i in range(data.shape[0]):
        min = data[i].min()
        max = data[i].max()
        data[i] = data[i] / (max - min)
    return data


class MyNoiseDataset(Dataset):

    def __init__(self, folder, annotations_file):
        self.folder = folder
        self.annotations_file = pd.read_csv(annotations_file)

    def __len__(self):
        return len(self.annotations_file)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, _ = torchaudio.load(os.path.join(self.folder, audio_sample_path))
        signal = minmaxscaler(signal)
        return signal, label

    def _get_audio_sample_path(self, index):
        path = self.annotations_file.iloc[index, 1]
        return path

    def _get_audio_sample_label(self, index):
        label = self.annotations_file.iloc[index, 2]
        label = json.loads(label)  # transform str to numpy float32
        label = np.array(label)
        label = label.astype(np.float32)
        return label


class MyNoiseDatasetFrames(Dataset):
    """
    The dataset for the ANC with the real and sequential data.

    In this class, we read the files from the folder that each file will contain a frame, including 10 seconds.
    The labels are annotated as <file_name(i), label(i)>, where i is the index of the second i.
    """

    def __init__(self, folder, annotations_file, snr=30):
        self.folder = folder
        self.annotations_file = pd.read_csv(annotations_file)
        self.file_names = os.listdir(folder)
        self.snr = snr

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, item):
        # get the name of the file at the index
        audio_frame_path = self.file_names[item]
        # read the 10 seconds frame from the file
        signal, _ = torchaudio.load(os.path.join(self.folder, audio_frame_path))
        # normalize the signal for each frame
        signal = signal.reshape(10, 16000)
        signal = minmaxscaler_frames(signal)

        # add white noise to the signal
        snr = math.exp(self.snr / 10)
        for i in range(signal.shape[0]):
            signal[i] = add_white_noise(signal=signal[i].unsqueeze(0), snr=snr)[0, :]

        # get the labels for each second of the file at the index
        label_tags = [f"{audio_frame_path.split('.')[0]}({i}).wav" for i in range(10)]
        labels = self._get_audio_sample_label(label_tags)

        return signal, labels

    def _get_audio_sample_label(self, tags):
        # stack the labels into a tensor
        labels = torch.zeros((10, 15))
        for tag in tags:
            # get the line index of the tag in the "File_path" column
            # get the index for the label as the number in () of the tag
            tag_index = int(tag.split('(')[1].split(')')[0])
            tag = self.annotations_file[self.annotations_file["File_path"] == tag].index[0]

            label = self.annotations_file.iloc[tag, 2]
            label = json.loads(label)  # transform str to numpy float32
            label = torch.tensor(label, dtype=torch.float32)
            labels[tag_index] = label
        return labels
