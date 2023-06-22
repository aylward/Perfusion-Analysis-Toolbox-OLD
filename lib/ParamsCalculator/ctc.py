import os
import torch
import logging
import numpy as np
from scipy import signal
import itk
from builtins import object

# Concentration time curve computation

def mrp_s0(signal, config, device):
    '''
    Calculate the MRP bolus arrival time (bat) and corresponding S0: averaged over signals before bat
    return: s0 # (n_slice, n_row, n_column)
    '''
    sig_avg = torch.zeros([signal.size()[3]], device = device, dtype = torch.float, requires_grad = False)
    for t in range(signal.size()[3]):
        sig_avg[t] = torch.mean(signal[..., t])
    flag = True
    bat  = 0
    while flag:
        s0_avg = torch.mean(sig_avg[:bat + 1])
        if torch.abs(s0_avg - sig_avg[bat + 1]) / s0_avg < config.mrp_s0_threshold:
            bat += 1
        else:
            flag = False
            bat -= 1
        if bat == signal.size()[3] - 1:
            flag = False
            bat -= 1
    print('  Bolus arrival time (start from 0):', bat)
    s0 = torch.mean(signal[..., :bat], dim = 3) # time dimension == 3
    
    return s0, bat


def ctp_s0(signal, config, device):
    '''
    Calculate the CTP bolus arrival time (bat) and corresponding S0: averaged over signals before bat
    return: s0 # (n_slice, n_row, n_column)
    '''
    sig_avg = torch.zeros([signal.size()[3]], device = device, dtype = torch.float, requires_grad = False)
    for t in range(signal.size()[3]):
        sig_avg[t] = torch.mean(signal[..., t])
    threshold = config.ctp_s0_threshold * (torch.max(sig_avg) - torch.min(sig_avg))
    flag = True
    bat  = 1
    while flag:
        if sig_avg[bat] - sig_avg[bat - 1] >= threshold and sig_avg[bat + 1] > sig_avg[bat]:
            flag = False
        else:
            bat += 1
        if bat == signal.size()[3]:
            flag = False
    print('  Bolus arrival time (start from 0):', bat - 1)
    s0 = torch.mean(signal[..., :bat], dim = 3) # time dimension == 3
    
    return s0, bat



def mr2ctc(signal, config, device):

    # TODO: use mask if needed

    s0, _ = mrp_s0(signal, config, device)
    ctc = torch.zeros(signal.size(), device = device, dtype = torch.float, requires_grad = False)
    
    for t in range(signal.size()[3]):
        ctc[..., t] = - config.k_mr/config.TE * torch.log(signal[..., t] / s0)

    # Check computed CTC: should have no NaN value
    if not len(torch.nonzero(torch.isnan(ctc))) == 0:
        raise ValueError('Computed CTC contains NaN value, check out!')

    return ctc


def ct2ctc(signal, config, device):

    s0, _ = ctp_s0(signal, config, device)
    ctc = torch.zeros(signal.size(), device = device, dtype = torch.float, requires_grad = False)
    for t in range(signal.size()[3]):
        ctc[..., t] = config.k_ct * (signal[..., t] - s0)

    # Check computed CTC: should have no NaN value
    if not len(torch.nonzero(torch.isnan(ctc))) == 0:
        raise ValueError('Computed CTC contains NaN value, check out!')

    return ctc

def cal(raw_perf, itk_info, config, device):
    print('Calculating Concentration Time Curve ...')
    if config.image_type == 'CTP':
        ctc = ct2ctc(raw_perf, config, device) 
        ctc_raw_nda = ctc.cpu()
        ctc_raw_nda = ctc_raw_nda.numpy()
        ctc_raw = itk.GetImageFromArray(ctc_raw_nda)
        ctc_raw.SetOrigin(itk_info[0])
        ctc_raw.SetSpacing(itk_info[1])
        ctc_raw.SetDirection(itk_info[2])
        ctcname = os.path.join(itk_info[3], 'CTC.nii')
        print('  Save calculated ctc as:', os.path.basename(ctcname))
        itk.imwrite(ctc_raw, ctcname)
        return ctc
