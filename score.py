# Import some general modules
import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
from glob import glob

# Select which GPU to use (if available)
gpu = 0
if torch.cuda.is_available():
    torch.cuda.device(gpu)
    print("Using GPU %d" % gpu)
else:
    gpu = None
    print("Using CPU only")


# Check if dataset is present
import os.path
dataset_path = "/workspace/dataset/GOLD_XYZ_OSC.0001_1024.hdf5"
os.path.isfile(dataset_path)


# Prepare data loader
from torch.utils.data import Dataset, DataLoader
import h5py

class radioml_18_dataset(Dataset):
    def __init__(self, dataset_path, load_into_ram=False):
        super(radioml_18_dataset, self).__init__()
        h5_file = h5py.File(dataset_path,'r')
        if load_into_ram:
            self.data = h5_file['X'][:]
        else:
            self.data = h5_file['X']
        self.mod = np.argmax(h5_file['Y'], axis=1) # comes in one-hot encoding
        self.snr = h5_file['Z'][:,0]
        self.len = self.data.shape[0]

        self.mod_classes = ['OOK','4ASK','8ASK','BPSK','QPSK','8PSK','16PSK','32PSK',
        '16APSK','32APSK','64APSK','128APSK','16QAM','32QAM','64QAM','128QAM','256QAM',
        'AM-SSB-WC','AM-SSB-SC','AM-DSB-WC','AM-DSB-SC','FM','GMSK','OQPSK']
        self.snr_classes = np.arange(-20., 32., 2) # -20dB to 30dB

        # do not touch this seed to ensure the prescribed train/test split!
        np.random.seed(2018)
        train_indices = []
        test_indices = []
        for mod in range(0, 24): # all modulations (0 to 23)
            for snr_idx in range(0, 26): # all SNRs (0 to 25 = -20dB to +30dB)
                # 'X' holds frames strictly ordered by modulation and SNR
                start_idx = 26*4096*mod + 4096*snr_idx
                indices_subclass = list(range(start_idx, start_idx+4096))
                
                # 90%/10% training/test split, applied evenly for each mod-SNR pair
                split = int(np.ceil(0.1 * 4096)) 
                np.random.shuffle(indices_subclass)
                train_indices_subclass = indices_subclass[split:]
                test_indices_subclass = indices_subclass[:split]
                
                # you could train on a subset of the data, e.g. based on the SNR
                # here we use all available training samples
                if snr_idx >= 0:
                    train_indices.extend(train_indices_subclass)
                test_indices.extend(test_indices_subclass)
                
        self.train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        self.test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

    def __getitem__(self, idx):
        # transpose frame into Pytorch channels-first format (NCL = -1,2,1024)
        return self.data[idx].transpose(), self.mod[idx], self.snr[idx]

    def __len__(self):
        return self.len

dataset = radioml_18_dataset(dataset_path, load_into_ram=True)

from torch import nn
import brevitas.nn as qnn
from brevitas.quant import IntBias
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.defaults import Int8ActPerTensorFloatMinMaxInit

# Adjustable hyperparameters
input_bits = 4
a_bits = 4
w_bits = 4
filters_conv = 64
filters_dense = 128

# Setting seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

class InputQuantizer(Int8ActPerTensorFloatMinMaxInit):
    bit_width = input_bits
    min_val = -2.0
    max_val = 2.0
    scaling_impl_type = ScalingImplType.CONST # Fix the quantization range to [min_val, max_val]

def genModel():
    model = nn.Sequential(
        # Input quantization layer
        qnn.QuantHardTanh(act_quant=InputQuantizer),

        qnn.QuantConv1d(2, filters_conv, 3, padding=1, weight_bit_width=w_bits, bias=False),
        nn.BatchNorm1d(filters_conv),
        qnn.QuantReLU(bit_width=a_bits),
        nn.MaxPool1d(2),

        qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, weight_bit_width=w_bits, bias=False),
        nn.BatchNorm1d(filters_conv),
        qnn.QuantReLU(bit_width=a_bits),
        nn.MaxPool1d(2),

        qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, weight_bit_width=w_bits, bias=False),
        nn.BatchNorm1d(filters_conv),
        qnn.QuantReLU(bit_width=a_bits),
        nn.MaxPool1d(2),

        qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, weight_bit_width=w_bits,bias=False),
        nn.BatchNorm1d(filters_conv),
        qnn.QuantReLU(bit_width=a_bits),
        nn.MaxPool1d(2),

        qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, weight_bit_width=w_bits, bias=False),
        nn.BatchNorm1d(filters_conv),
        qnn.QuantReLU(bit_width=a_bits),
        nn.MaxPool1d(2),

        qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, weight_bit_width=w_bits, bias=False),
        nn.BatchNorm1d(filters_conv),
        qnn.QuantReLU(bit_width=a_bits),
        nn.MaxPool1d(2),

        qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, weight_bit_width=w_bits, bias=False),
        nn.BatchNorm1d(filters_conv),
        qnn.QuantReLU(bit_width=a_bits),
        nn.MaxPool1d(2),
        
        nn.Flatten(),

        qnn.QuantLinear(filters_conv*8, filters_dense, weight_bit_width=w_bits, bias=False),
        nn.BatchNorm1d(filters_dense),
        qnn.QuantReLU(bit_width=a_bits),

        qnn.QuantLinear(filters_dense, filters_dense, weight_bit_width=w_bits, bias=False),
        nn.BatchNorm1d(filters_dense),
        qnn.QuantReLU(bit_width=a_bits, return_quant_tensor=True),

        qnn.QuantLinear(filters_dense, 24, weight_bit_width=w_bits, bias=True, bias_quant=IntBias),
    )
    return model

# Load trained parameters of the model
savefile = '/workspace/sandbox/run4/bestnet10.pth'
checkpoint = torch.load(savefile, map_location=torch.device("cpu"))
model = genModel()

batch_size = 2**11 
data_loader_test = DataLoader(dataset, batch_size=batch_size, sampler=dataset.test_sampler, num_workers = 8)

model.load_state_dict(checkpoint)
if gpu is not None:
    model = model.cuda()

# Run inference on validation data
y_exp = np.empty((0))
y_snr = np.empty((0))
y_pred = np.empty((0,len(dataset.mod_classes)))
model.eval()
with torch.no_grad():
    for data in data_loader_test:
        inputs, target, snr = data
        if gpu is not None:
            inputs = inputs.cuda()
        output = model(inputs)
        y_pred = np.concatenate((y_pred,output.cpu()))
        y_exp = np.concatenate((y_exp,target))
        y_snr = np.concatenate((y_snr,snr))

conf = np.zeros([len(dataset.mod_classes),len(dataset.mod_classes)])
confnorm = np.zeros([len(dataset.mod_classes),len(dataset.mod_classes)])
for i in range(len(y_exp)):
    j = int(y_exp[i])
    k = int(np.argmax(y_pred[i,:]))
    conf[j,k] = conf[j,k] + 1
for i in range(0,len(dataset.mod_classes)):
    confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])

cor = np.sum(np.diag(conf))
ncor = np.sum(conf) - cor
print("Overall Accuracy across all SNRs: %f"%(cor / (cor+ncor)))


from brevitas.export.onnx.generic.manager import BrevitasONNXManager

export_onnx_path = "/workspace/sandbox/run4/model_export.onnx"
final_onnx_path = "/workspace/sandbox/run4/model_final.onnx"
cost_dict_path = "/workspace/sandbox/run4/model_cost.json"

BrevitasONNXManager.export(model.cpu(), input_t=torch.randn(1, 2, 1024), export_path=export_onnx_path)

from finn.util.inference_cost import inference_cost
import json

inference_cost(export_onnx_path, output_json=cost_dict_path, output_onnx=final_onnx_path,
               preprocess=True, discount_sparsity=True)

with open(cost_dict_path, 'r') as f:
    inference_cost_dict = json.load(f)

bops = int(inference_cost_dict["total_bops"])
w_bits = int(inference_cost_dict["total_mem_w_bits"])

bops_baseline = 807699904
w_bits_baseline = 1244936

score = 0.5*(bops/bops_baseline) + 0.5*(w_bits/w_bits_baseline)
print("Normalized inference cost score: %f" % score)
