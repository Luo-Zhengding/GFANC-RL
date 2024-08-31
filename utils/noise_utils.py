"""
Some utils to process the noise.
"""

import os
import torch
import torchaudio

import numpy as np

from scipy import signal
import scipy.io as sio


def loading_real_wave_noise(noise_path):
    """
    load the noise
    :param folder_name:
    :param sound_name:
    :return:
    """
    waveform, sample_rate = torchaudio.load(noise_path)
    return waveform, sample_rate


def loading_paths_from_MAT(Primay_path_file, Secondary_path_file):
    Pri_dfs, Secon_dfs = sio.loadmat(Primay_path_file), sio.loadmat(Secondary_path_file)
    Pri_path, Secon_path = Pri_dfs['Pz1'].squeeze(), Secon_dfs['S'].squeeze()
    return Pri_path, Secon_path


def disturbance_generation_from_real_noise(wave_form, fs, Pri_path, Sec_path, Repet=0):
    wave = wave_form[0, :].numpy()
    wavec = wave
    for ii in range(Repet):
        wavec = np.concatenate((wavec, wave), axis=0)  # add the length of the wave_form through repetition

    # Constructing the desired signal
    Dir, Fx = signal.lfilter(Pri_path, 1, wavec), signal.lfilter(Sec_path, 1, wavec)

    N = len(Dir)
    N_z = N // fs
    Dir, Fx = Dir[0:N_z * fs], Fx[0:N_z * fs]

    return torch.from_numpy(Dir).type(torch.float), torch.from_numpy(Fx).type(torch.float), torch.from_numpy(
        wavec).type(torch.float)
