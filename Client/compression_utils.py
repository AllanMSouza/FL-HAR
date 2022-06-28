from collections import OrderedDict
from typing import Dict, List, Tuple
import os

import numpy as np
import torch

PRINT_BITS = 1 #os.environ['PRINT_BITS']
DEVICE     = 'cpu'
device     = 'cpu'


def print_bits(param, compressed, method, aprox):
    bits_compressed = 0
    normal_bits     = 0

    for layer in range(len(compressed)):
        bits_compressed += get_bits(compressed[layer], method, aprox)
        normal_bits     += get_bits(torch.from_numpy(param[layer]), method, aprox)

    print(f'Normal weights    : {normal_bits}')
    print(f'Compressed weights: {bits_compressed}')

def DGC(param, p, aprox):
    '''
    "Deep Gradient Compression: Reducing the communication Bandwidth for Distributed Training, Lin et al."
    '''

    compressed = []

    for layer in range(len(param)):
        compressed.append(torch.zeros(param[layer].shape).to(device))
        
    
    for l in range(len(param)):
        T     = torch.from_numpy(param[l])
        T_abs = torch.abs(T)
        v, _  = approx_v(T_abs, p, aprox)
        out   = torch.where(T_abs >= v, T, torch.Tensor([0.0]).to(device))
        compressed[l] = out

    #     T[l][x] = T[l][x].cpu().detach().numpy()
    if PRINT_BITS:
        print_bits(param, compressed, 'dgc', aprox)
        
        
    return compressed



def STC(param, p, aprox):
    '''
    "Sparse Binary Compression: Towards Distributed Deep Learning with minimal Communication, Sattler et al."
    '''

    compressed = []

    for layer in range(len(param)):
        compressed.append(torch.zeros(param[layer].shape).to(device))
        
    
    for l in range(len(param)):
        T        = torch.from_numpy(param[l])
        T_abs    = torch.abs(T)
        v, topk  = approx_v(T_abs, p, aprox)
        mean     = torch.mean(topk)
        out_     = torch.where(T >= v, mean, torch.Tensor([0.0]).to(device))
        out      = torch.where(T <= -v, -mean, out_)
        compressed[l] = out


    if PRINT_BITS:
        print_bits(param, compressed, 'stc', aprox)
    return compressed

def SSGD(param, aprox):
    """
      signSGD: Compressed Optimisation for non-convex Problems, Bernstein et al.

    """
    compressed = []

    for layer in range(len(param)):
        compressed.append(torch.zeros(param[layer].shape).to(device))
        

    for l in range(len(param)):
        T             = torch.from_numpy(param[l])
        compressed[l] = T.sign()

    if PRINT_BITS:
        print_bits(param, compressed, 'sgd', aprox)

    return compressed

def get_bits(T, compression_method, approx=False):
    """
    Returns the number of bits that are required to communicate the Tensor T, which was compressed with compresion_method
    """

    B_val = {"none" : 32, "dgc" : 32, "stc" : 1, "sgd" : 1}[compression_method]

    # dense methods
    if compression_method in ["none", "sgd"]:
        k = T.numel()
        B_pos = 0

    # sparse methods non-optimal encoding
    elif compression_method in ["dgc"]:
        k = torch.sum(T!=0.0).item()
        B_pos = 16

    # sparse methods golomb encoding
    elif compression_method in ["stc"]:
        k = torch.sum(T!=0.0).item()
        N = T.numel()

        q = (k+1)/(N+1)
        golden = (np.sqrt(5)+1)/2

        if q == 1:
            return B_val*T.numel()
        if q == 0:
            return 0

        b_star = 1 + np.floor(np.log2(np.log(golden-1)/np.log(1-q)))

        if approx:
            B_pos = b_star + 1/(1-(1-q)**(2**b_star)) + 1
        else:
            idc = torch.nonzero(T.view(-1))
            distances = idc[:]-torch.cat([torch.Tensor([[-1]]).long().to(device),idc[:-1]])
            B_pos = torch.mean(torch.ceil(distances.float()/2**b_star)).item()+(b_star+1)

    
    b_total = (B_pos+B_val)*k
    # f = open("bitsEnviados.txt", "a")
    #f.write(str(b_total)+"\n")
    #f.close()
    return b_total

def approx_v(T, p, frac):
    if frac < 1.0:
        n_elements = T.numel()
        n_sample = min(int(max(np.ceil(n_elements * frac), np.ceil(100/p))), n_elements)
        n_top = int(np.ceil(n_sample*p))

        if n_elements == n_sample:
            i = 0
        else:
            i = np.random.randint(n_elements-n_sample)

        topk, _ = torch.topk(T.flatten()[i:i+n_sample], n_top)
        if topk[-1] == 0.0 or topk[-1] == T.max():
            return approx_v(T, p, 1.0)
    else:
        n_elements = T.numel()
        n_top = int(np.ceil(n_elements*p))
        topk, _ = torch.topk(T.flatten(), n_top)

    return topk[-1], topk