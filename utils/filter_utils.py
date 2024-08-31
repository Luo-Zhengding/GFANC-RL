"""
Some utils to process the filters.
"""

import numpy as np

import torch

import scipy.io as sio


def load_weights(model, pretrained_path, device):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_path, map_location=device)
    for k, v in model_dict.items():
        model_dict[k] = pretrained_dict[k]
    model.load_state_dict(model_dict)


def load_pretrained_filters(mat_file):
    mat_contents = sio.loadmat(mat_file)
    Wc_vectors = mat_contents['Wc_v']
    return torch.from_numpy(Wc_vectors).type(torch.float)


def cast_multiple_time_length_of_primary_noise(primary_noise, fs=16000):
    # multiple length of samples
    assert primary_noise.shape[0] == 1, 'The dimension of the primary noise should be [1 x samples]'
    cast_len = primary_noise.shape[1] - primary_noise.shape[1] % fs
    return primary_noise[:, :cast_len]  # make the length of primary_noise is an integer multiple of fs


def make_noise_frames(primary_noise, fs=16000):
    assert primary_noise.shape[0] == 1, 'The dimension of the primary noise should be [1 x samples]'
    assert primary_noise.shape[1] % fs == 0, 'The length of the primary noise is not an integral multiple of fs.'

    time_len = int(primary_noise.shape[1] / fs)

    # build the matrices of the primary noise (time_len, 1, fs)
    primary_noise_frames = primary_noise.reshape(time_len, fs).unsqueeze(1)
    return primary_noise_frames


def make_frames(re_ori, dis_ori, fx_ori, fs=16000):
    time_len = int(re_ori.shape[1] / fs) if re_ori.ndim == 2 else int(re_ori.shape[0] / fs)

    re_frames = re_ori.reshape(time_len, fs).unsqueeze(1)
    dis_frames = dis_ori.reshape(time_len, fs)
    fx_frames = fx_ori.reshape(time_len, fs)

    return re_frames, dis_frames, fx_frames


def predict_hard_weights(primary_noise_frames, device, model, threshold=0.5):
    noise_frames = primary_noise_frames.to(device)
    noise_frames = noise_frames / (noise_frames.max() - noise_frames.min())
    prediction = model(noise_frames)
    hard_weights = prediction.detach().cpu().numpy() >= threshold
    hard_weights = hard_weights.astype(int)
    return hard_weights
