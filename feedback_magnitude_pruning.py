"""
Feedback Magnitude Pruning (FMP)
Author: Jakob Krzyston (jakobk@gatech.edu)
For the ITU AI/ML in 5G Challenge: Lightning-Fast Modulation Classification with Hardware-Eﬀicient Neural Networks
"""

## Import Packages ##
import torch, math, copy, h5py, os.path, time
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import brevitas.nn as qnn
from brevitas.quant import IntBias
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.defaults import Int8ActPerTensorFloatMinMaxInit
import torch.nn.utils.prune as prune
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score


## Name experiment, make directory ##
expt_name = '4_bit_fmp'
print('Experiment Name: ' + expt_name)
os.mkdir(expt_name)
pruning_types = ['Unstructured']#['Structured','Unstructured']#

## Adjustable hyperparameters for base model ##
input_bits = 4 # Originally 8
a_bits = 4 # Originally 8
w_bits = 4 # Originally 8
filters_conv = 64 # Originally 64 
filters_dense = 128 # Originally 128

## Pruning parameters ##
# define the initial amount of pruning, p (will not go below 0.05, unless structured pruning will not go 0.01)
p = .2
p_copy = copy.deepcopy(p) # copy is used when changing pruning type
# define the factor by which p will be reduced by
n = 2

## Select which GPU to use (if available) ##
gpu = 0
if torch.cuda.is_available():
    torch.cuda.device(gpu)
    print("Using GPU %d" % gpu)
else:
    gpu = None
    print("Using CPU only")

## Load Data ##
dataset_path = path_to_data +"/GOLD_XYZ_OSC.0001_1024.hdf5"
os.path.isfile(dataset_path)

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

# Setting seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)



## Functions ##
def burnInPruningMask(model):
    for module in model.modules():
        if isinstance(module, (qnn.QuantConv1d, qnn.QuantLinear)) and prune.is_pruned(module):
            prune.remove(module, 'weight')

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

def train(model, train_loader, optimizer, criterion):
    losses = []
    # ensure model is in training mode
    model.train()    

    for (inputs, target, snr) in train_loader:   
        if gpu is not None:
            inputs = inputs.cuda()
            target = target.cuda()
                
        # forward pass
        output = model(inputs)
        loss = criterion(output, target)
        
        # backward pass + run optimizer to update weights
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        
        # keep track of loss value
        losses.append(loss.cpu().detach().numpy())
           
    return losses

def test(model, test_loader):    
    # ensure model is in eval mode
    model.eval() 
    y_true = []
    y_pred = []
   
    with torch.no_grad():
        for (inputs, target, snr) in test_loader:
            if gpu is not None:
                inputs = inputs.cuda()
                target = target.cuda()
            output = model(inputs)
            pred = output.argmax(dim=1, keepdim=True)
            y_true.extend(target.tolist()) 
            y_pred.extend(pred.reshape(-1).tolist())
        
    return accuracy_score(y_true, y_pred)

def display_loss_plot(losses, title="Training loss", xlabel="Iterations", ylabel="Loss"):
    x_axis = [i for i in range(len(losses))]
    plt.plot(x_axis,losses)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def pruneModel_unstructured(model, pruningAmountPerIteration):
    import torch.nn.utils.prune as prune
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, (qnn.QuantConv1d, qnn.QuantLinear)):
            parameters_to_prune.append((module,'weight'))
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruningAmountPerIteration
    )
def pruneModel_structured(model, pruningAmountPerIteration):
    import torch.nn.utils.prune as prune
    for module in model.modules():
        if isinstance(module, (qnn.QuantLinear, qnn.QuantConv1d)):
            prune.LnStructured.apply(
                module,
                name = 'weight',
                n = 1,
                amount=pruningAmountPerIteration,
                dim = 0)

def find_new_dimensions(model, layers_of_interest):
    result_structure = torch.zeros((len(layers_of_interest)))
    for idx,layer in enumerate(layers_of_interest):
        shapes = (model[layer].weight).shape
        if len(shapes) == 3: #conv layer
            for filters in range(shapes[0]):
                if shapes[1]*shapes[2] != sum(sum((model[layer].weight)[filters] == 0)):
                    result_structure[idx] += 1
        else: # dense layer
            for row in range(shapes[0]): 
                if shapes[0] != sum((model[layer].weight)[row] == 0):
                    result_structure[idx] +=1
    return result_structure

def gen_new_Model(result_structure):
    model = nn.Sequential(
        # Input quantization layer
        qnn.QuantHardTanh(act_quant=InputQuantizer),

        qnn.QuantConv1d(2, int(result_structure[0]), 3, padding=1, weight_bit_width=w_bits, bias=False),
        nn.BatchNorm1d(int(result_structure[0])),
        qnn.QuantReLU(bit_width=a_bits),
        nn.MaxPool1d(2),

        qnn.QuantConv1d(int(result_structure[0]), int(result_structure[1]), 3, padding=1, weight_bit_width=w_bits, bias=False),
        nn.BatchNorm1d(int(result_structure[1])),
        qnn.QuantReLU(bit_width=a_bits),
        nn.MaxPool1d(2),

        qnn.QuantConv1d(int(result_structure[1]), int(result_structure[2]), 3, padding=1, weight_bit_width=w_bits, bias=False),
        nn.BatchNorm1d(int(result_structure[2])),
        qnn.QuantReLU(bit_width=a_bits),
        nn.MaxPool1d(2),

        qnn.QuantConv1d(int(result_structure[2]), int(result_structure[3]), 3, padding=1, weight_bit_width=w_bits,bias=False),
        nn.BatchNorm1d(int(result_structure[3])),
        qnn.QuantReLU(bit_width=a_bits),
        nn.MaxPool1d(2),

        qnn.QuantConv1d(int(result_structure[3]), int(result_structure[4]), 3, padding=1, weight_bit_width=w_bits, bias=False),
        nn.BatchNorm1d(int(result_structure[4])),
        qnn.QuantReLU(bit_width=a_bits),
        nn.MaxPool1d(2),

        qnn.QuantConv1d(int(result_structure[4]), int(result_structure[5]), 3, padding=1, weight_bit_width=w_bits, bias=False),
        nn.BatchNorm1d(int(result_structure[5])),
        qnn.QuantReLU(bit_width=a_bits),
        nn.MaxPool1d(2),

        qnn.QuantConv1d(int(result_structure[5]), int(result_structure[6]), 3, padding=1, weight_bit_width=w_bits, bias=False),
        nn.BatchNorm1d(int(result_structure[6])),
        qnn.QuantReLU(bit_width=a_bits),
        nn.MaxPool1d(2),
        
        nn.Flatten(),

        qnn.QuantLinear(int(result_structure[6])*8, int(result_structure[7]), weight_bit_width=w_bits, bias=False),
        nn.BatchNorm1d(int(result_structure[7])),
        qnn.QuantReLU(bit_width=a_bits),

        qnn.QuantLinear(int(result_structure[7]), int(result_structure[8]), weight_bit_width=w_bits, bias=False),
        nn.BatchNorm1d(int(result_structure[8])),
        qnn.QuantReLU(bit_width=a_bits, return_quant_tensor=True),

        qnn.QuantLinear(int(result_structure[8]), 24, weight_bit_width=w_bits, bias=True, bias_quant=IntBias),
    )
    return model

def compression_ratio(prune_iters, p, n):
    ratio = np.ones((prune_iters.shape[0]))
    sparsities = np.ones((prune_iters.shape[0]))
    for i in range(prune_iters.shape[0]):
        p_n = copy.deepcopy(p)
        for j in range (prune_iters.shape[1]):
            ratio[i] *= 1/((1-p_n)**prune_iters[i,j])
            sparsities[i] *= 1-(p_n**prune_iters[i,j])
            p_n = p_n/n
    return ratio, sparsities

## End of functions ##




## generate model and specify the convolutional and dense layer locations ##
model = genModel()
if gpu is not None:
    model = model.cuda()
layers_of_interest = [1, 5, 9, 13, 17, 21, 25, 30, 33] # only needed if structurally pruning


# create array to store values for compression ratio computation
prune_iters = np.zeros((len(pruning_types), int(np.floor(math.log(p*100,n))))) # if p_threshold becomes 1 --> int(np.floor(math.log(p*100,n))+1)
pruning_iter = 0 # counter to be used for compression ratio computation  

## Training parameters and training/ test sets ##
batch_size = 2**10
num_epochs = 50 # currently the training counter does not reset after training
patience = 15
data_loader_train = DataLoader(dataset, batch_size=batch_size, sampler=dataset.train_sampler, num_workers = 8)
data_loader_test = DataLoader(dataset, batch_size=batch_size, sampler=dataset.test_sampler, num_workers = 8)
# loss criterion and optimizer
criterion = nn.CrossEntropyLoss()
if gpu is not None:
    criterion = criterion.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=1/10., patience=patience, verbose=True)
best_acc = 0 # Initialize a peak accuracy of 0
epoch = 0 # Initialize the training epoch counter

## Start a timer ##
start = time.time()


### Feedback Magnitude Pruning ###
for idx,prune_type in enumerate(pruning_types):
    print("Pruning Type: " + prune_type)
    prune_pct_idx = 0
    if prune_type == 'Structured': # strucutred pruning is more difficult, it needs to start smaller and requires extra granularity
        p_threshold = 0.01
        p = p/n
    else:
        p_threshold = 0.05
        p = p_copy
    print('Pruning threshold: '+str(p_threshold))
    print("p = %f"%(p))
    while p >= p_threshold: 
        epochs_without_improvement = 0
        checkpoint_name = "./"+expt_name+"/best_model_{}_p_{}.pth".format(prune_type,str(p))
        while epoch <= num_epochs:
            ep_start = time.time()
            loss_epoch = train(model, data_loader_train, optimizer, criterion)
            ep_end = time.time()
            test_acc = test(model, data_loader_test)
            print("p = %f, Epoch %d: Test accuracy = %f, %f mins"%(p, epoch, test_acc, np.round((ep_end-ep_start)/60,6)))          
            if test_acc > best_acc:
                print("Epoch %d: Found new best model, saving!"%epoch)
                saveModel = copy.deepcopy(model)
                burnInPruningMask(saveModel)
                checkpoint = {"model_state_dict": saveModel.state_dict()}
                torch.save(saveModel.state_dict(), checkpoint_name)
                best_acc = test_acc
                #always save better models, but do early stopping based on significantly better
                if test_acc>best_acc+0.001:
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement+=1
                    if epochs_without_improvement==patience:
                        print("Epoch %d: Too many epochs without improvement, pruning iteration %d done"%(epoch,pruning_iter))
                        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=1/10., patience=patience, verbose=False)
                        break
                lr_scheduler.step(test_acc)
                epoch += 1
            else:
                epochs_without_improvement+=1
                lr_scheduler.step(test_acc)
                epoch += 1
                if epochs_without_improvement==patience:
                    print("p = %f, Epoch %d: Patience threshold reached, pruning iteration %d done"%(p,epoch,pruning_iter))
                    break
            if best_acc >= 0.56:
                print("Accuracy threshold reached, pruning iteration %d"%(pruning_iter))
                prune_iters[idx, prune_pct_idx] = pruning_iter
                pruning_iter += 1
                saveModel = copy.deepcopy(model)
                burnInPruningMask(saveModel)
                checkpoint = {"model_state_dict": saveModel.state_dict()}
                torch.save(saveModel.state_dict(), "./"+expt_name+"/best_56_model.pth") # keep track of absolute best model
                if prune_type == 'Structured':
                    pruneModel_structured(model, p)
                elif prune_type == 'Unstructured':
                    pruneModel_unstructured(model, p)
                epochs_without_improvement = 0 # reset patience counter
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=1/10., patience=patience, verbose=False)
                best_acc = 0 # reset best accuracy
                epoch = 0 # reset training counter
        if best_acc < 0.56:
            p /= n # change the pruning rate
            best_acc = 0 # reset best accuracy
            epoch = 0 # reset training counter
            pruning_iter = 0 # reset counter for new pruning pct
            prune_pct_idx += 1 # new index to store number of pruning iterations into prune_iters
            print('Loading most pruned network capable of 56% accuracy')
            previous_checkpoint = torch.load("./"+expt_name+"/best_56_model.pth", map_location=torch.device("cuda"))
            model = genModel()
            model.load_state_dict(previous_checkpoint)
            if gpu is not None:
                model = model.cuda()
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=1/10., patience=patience, verbose=False)
            print("p = %f"%(p))
    if prune_type == 'Structured':
        #TODO: squeeze weights to fit new model? Currently reinitializes weights
        print('Creating reduced model, restoring pruning value, continuing with Unstructured pruning')
        result_structure = find_new_dimensions(model, layers_of_interest)
        print('Result Structure: '+str(result_structure))
        model = gen_new_Model(result_structure)
        if gpu is not None:
            model = model.cuda()
        p = p_copy       
    else:
        ratio, sparsities = compression_ratio(prune_iters, p, n)
        for i in range(len(pruning_types)):
            print('Prune Iterations '+pruning_types[i]+':  ' + str(prune_iters[i]))
            print('Compression Ratio '+pruning_types[i]+': ' + str(ratio[i]))
            print('Sparsity '+pruning_types[i]+':          ' + str(sparsities[i]))
        break

## End pruning timer ##
end = time.time()
print("Total time FMP: %f minutes"%(np.round((end-start)/60,2)))

## Evaluate final model ##
# Load trained parameters of the best model
checkpoint = torch.load("./"+expt_name+"/best_56_model.pth", map_location=torch.device("cpu"))
if pruning_types[0] == 'Structured':
    model = gen_new_Model(result_structure)
else:
    model = genModel()

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

export_onnx_path = "./"+expt_name+"/model_export.onnx"
final_onnx_path  = "./"+expt_name+"/model_final.onnx"
cost_dict_path   = "./"+expt_name+"/model_cost.json"

BrevitasONNXManager.export(model.cpu(), input_t=torch.randn(1, 2, 1024), export_path=export_onnx_path)

from finn.util.inference_cost import inference_cost
import json

inference_cost(export_onnx_path, output_json=cost_dict_path, output_onnx=final_onnx_path,
               preprocess=True, discount_sparsity=True)

with open(cost_dict_path, 'r') as f:
    inference_cost_dict = json.load(f)

bops = int(inference_cost_dict["total_bops"])
w_bits = int(inference_cost_dict["total_mem_w_bits"])

print('Bit_Ops: '+str(bops))
print('W_Bits: '+str(w_bits))

bops_baseline = 807699904
w_bits_baseline = 1244936

print('Please be advised the printed values do not match the output from the competition Docker, which are lower.')

score = 0.5*(bops/bops_baseline) + 0.5*(w_bits/w_bits_baseline)
print("Normalized inference cost score: %f" % score)