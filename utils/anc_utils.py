"""
Some utils that are related to the anc.
"""

import numpy as np

import torch
from scipy import signal

import math


def compute_error(Dis, Fx, control_filter, fs=16000):
    """
    The function to eliminate noise in TEST, which means that the first frame will not be filtered.
    """
    filter_length = control_filter.shape[1]
    Xd = torch.zeros(1, filter_length, dtype=torch.float64)
    current_filter = torch.zeros(1, filter_length, dtype=torch.float64)

    error = torch.zeros(Dis.shape[0])

    j = 0
    for i, dis in enumerate(Dis):
        Xd = torch.roll(Xd, 1, 1)
        Xd[0, 0] = Fx[i]
        yt = current_filter @ Xd.t()
        e = dis - yt
        error[i] = e.item()
        if (i + 1) % fs == 0:
            current_filter = control_filter[j]
            j += 1
    return error


def compute_single_error(Dis, Fx, control_filter):
    filter_length = control_filter.shape[0]
    Xd = torch.zeros(1, filter_length, dtype=torch.float64)

    error = torch.zeros(Dis.shape[0])

    for i, dis in enumerate(Dis):
        Xd = torch.roll(Xd, 1, 1)
        Xd[0, 0] = Fx[i]
        yt = control_filter @ Xd.t()
        e = dis - yt
        error[i] = e.item()

    error = torch.sum(error ** 2).item()
    return error


def compute_frame_error(Dis, Fx, control_filter):
    yt = signal.lfilter(control_filter, 1, Fx)
    e = Dis - yt
    error = e ** 2
    error = torch.sum(error).item()
    return error


def compute_frame_error_vector(Dis, Fx, control_filter):
    yt = signal.lfilter(control_filter, 1, Fx)
    e = Dis - yt
    return e


def add_white_noise(signal, snr):
    signal_power = signal.norm(p=2)
    length = signal.shape[1]
    additional_noise = np.random.randn(length)
    additional_noise = torch.from_numpy(additional_noise).type(torch.float32).unsqueeze(0)
    noise_power = additional_noise.norm(p=2)
    # snr = math.exp(snr_db / 10)
    scale = snr * noise_power / signal_power
    noisy_signal = signal + additional_noise / scale
    return noisy_signal


def compute_frame_error_new(Dis, Fx, control_filter, forget_vector):
    Wc = control_filter  # The generate filter [1, 1024]
    # torch.Size([1, 1023])
    Pad = torch.zeros((control_filter.shape[0] - 1), dtype=torch.float32).to(control_filter.device)
    Fx = torch.cat((Pad, Fx), dim=0)  # torch.Size([, 17023])
    # Shift filtered-x by the control filter window torch.Size([1, 16000, 1024])
    Fx_vector = Fx.unfold(0, control_filter.shape[0], 1)
    Fx_v = torch.flip(Fx_vector, dims=[1]).type(torch.float64)  # torch.Size([1, 16000, 1024])
    Y_anti = torch.matmul(Wc, Fx_v.T)  # torch.Size([1, 16000])
    Er_v = Dis - Y_anti  # torch.Size([1, 16000])

    Er_2 = torch.pow(Er_v, 2)  # torch.Size([1, 16000])
    error = torch.matmul(forget_vector[0], Er_2)
    # error = Er_v ** 2
    # error = torch.sum(error).item()
    return error


def noise_cancellation(sub_filters, predict_weights, Disturbance, filter_ref, BATCH_SIZE):
    """ Noise cancelation based on the generate filter
    Args:
        sub_filters(float32 tensor)    : The pre-trained control filter group, whose size shoule be [15 x 1024]
        predict_weights(float32 tensor) : The output lable of the CNN, and its size should be [1 x 15]
        Disturbance(float32 tensor)     : The disturbance signal, whose size is [1 x len]
        filter_ref(float32 tensor)      : The filtered reference signal, whose size is [1 x len]

    Returns:
        float32 tensor : [1 x Len] the error signal
    """
    B = BATCH_SIZE
    Wc = torch.matmul(predict_weights, sub_filters)[0]  # !!! The generate filter [1024]
    Pad = torch.zeros((B, sub_filters.shape[1] - 1), dtype=torch.float32)  # # torch.Size([B, 1023])
    Fx = torch.cat((Pad, filter_ref), dim=1)  # torch.Size([B, 17023])
    # Shift filtered-x by the control filter window torch.Size([B, 16000, 1024])
    Fx_vector = Fx.unfold(1, sub_filters.shape[1], 1)
    Fx_v = torch.flip(Fx_vector, dims=[2])  # torch.Size([B, 16000, 1024])
    Y_anti = torch.matmul(Wc, Fx_v.transpose(1, 2))  # torch.Size([1, 16000])
    Er_v = Disturbance - Y_anti  # torch.Size([1, 16000])

    return Er_v


def forget_factor_generation(Len_c=1024, Len_data=16000, Lambda=0.999, B=1):
    """ Generate forget_factor to reduce the effect of the initial value
    Args:
        Len_c (float32)   : the length of the control filter
        Len_data(float32) : the lenght of the each frame
        Lambda(float32)   : the forgeting factor

    Returns:
        the forgeting factor [1 x Len]
    """
    Forget_v = []
    for ii in range(Len_c):
        Forget_v.append(Lambda ** ii)
    Forget = np.tile(Forget_v, (B, 1))  # (B, 1024)
    For_v = torch.flip(torch.FloatTensor(Forget), dims=[1])
    Pad_1 = torch.ones((B, Len_data - Len_c), dtype=torch.float64)
    Forget_vector = torch.cat((For_v, Pad_1), dim=1)  # torch.Size([1, 16000])
    return Forget_vector


# BATCH_SIZE = 1
def loss_function(Er_v, Forget_vector):
    Er_2 = torch.pow(Er_v, 2)  # torch.Size([1, 16000])
    # ! deleted the Er_2.T
    loss = torch.matmul(Forget_vector, Er_2)  # torch.Size([1])
    return loss


def relative_loss_function_log(Dis, error):
    # power_dis = torch.var(Dis)
    power_dis = 10 * torch.log10(torch.var(Dis))
    # power_err = torch.var(error)
    power_err = 10 * torch.log10(torch.var(error))
    # nr_level = power_dis / power_err
    nr_level = power_dis - power_err
    # nr_level = 10 * torch.log10(power_dis / power_err)
    return nr_level


def relative_loss_function(Dis, error):
    power_dis = torch.var(Dis)
    # power_dis = 10 * torch.log10(torch.var(Dis))
    power_err = torch.var(error)
    # power_err = 10 * torch.log10(torch.var(error))
    nr_level = power_dis / power_err
    # nr_level = power_dis - power_err
    # nr_level = 10 * torch.log10(power_dis / power_err)
    return nr_level
