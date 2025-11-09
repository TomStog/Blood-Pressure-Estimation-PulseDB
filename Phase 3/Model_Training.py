import torch
import torch.utils.data as data
import random
import numpy as np
#import neurokit2 as nk
import progressbar
from mat73 import loadmat
from Model_Def.MyTrainer import Model_Trainer
#from Model_Def.ecg_func import ecg_func_v3
#from Model_Def.ppg_func import ppg_func_v2
from Model_Def import UNetViT1DRegressor, detsikas, DeepFFN

print(torch.version.cuda)

class Dataset(data.Dataset):
    def __init__(self, Input, Label):
        self.Input = Input
        self.Label = Label

    def __len__(self):
        return len(self.Input)

    def __getitem__(self, idx):
        return self.Input[idx, :], self.Label[[idx]]

#def Build_Dataset(Path, Label):
#    Data = loadmat(Path)
#    # Get the first two channels, which are the ECG and the PPG signals
#    return Dataset(Data['Subset']['Signals'][:, 0:2, :], Data['Subset'][Label])

def Build_Dataset(Path, Label):
    Data = loadmat(Path)
    # Get the first two channels, which are the ECG and the PPG signals

    Height = Data['Subset']['Height']
    Weight = Data['Subset']['Weight']

    # Create mask where both Height and Weight are NaN
    mask = (np.isnan(Height) & np.isnan(Weight)) #use only &(and) |(or)
    
    return Dataset(Data['Subset']['Signals'][mask, 0:2, :1248], Data['Subset'][Label][mask])

def Build_Dataset_v2(Path, Label):
    Data = loadmat(Path)

    # ECG
    ecg_signals = Data['Subset']['Signals'][:, 0, :].squeeze()
    # PPG
    ppg_signals = Data['Subset']['Signals'][:, 1, :].squeeze()

    # Process each signal individually
    ecg_features = []
    ppg_features = []
    valid_indices = []  # Track which indices have valid features

    for i in progressbar.progressbar(range(ppg_signals.shape[0])):
        ppg = ppg_signals[i, :]  # Get one signal: shape (1250,)
        ecg = ecg_signals[i, :]  # Get corresponding ECG signal

        features_ppg = ppg_func_v2(ppg)
        features_ecg = ecg_func_v3(ecg)
        
        # Only include if BOTH extractions succeeded
        if (features_ppg is not None) and (features_ecg is not None):
            ppg_features.append(features_ppg)
            ecg_features.append(features_ecg)
            valid_indices.append(i)
    
    # Convert to numpy array
    ecg_features = np.array(ecg_features)
    ppg_features = np.array(ppg_features)

    # Access Age and Gender only for valid indices
    Age = Data['Subset']['Age'][valid_indices] / 100
    Gender = np.array(Data['Subset']['Gender'])[valid_indices].squeeze()
    
    # Convert Gender to numerical 0-1 labels
    Gender = (Gender == 'M').astype(float).reshape(-1, 1)
    
    # Ensure Age is 2D for concatenation
    if Age.ndim == 1:
        Age = Age.reshape(-1, 1)

    concatenated_signals = np.concatenate([
        ecg_features, 
        ppg_features,
        Age,
        Gender
    ], axis=1)

    # Get labels only for valid indices
    labels = Data['Subset'][Label][valid_indices]

    return Dataset(concatenated_signals, labels)

torch.cuda.empty_cache()
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device)
print(torch.cuda.get_device_name(0))

# Replace 'YOUR_PATH' with the folder of your generated Training, CalBased and CalFree testing subsets.
data_folder = './Subset_Files/'
Train_File = data_folder+'Train_Subset.mat'
Test_CalBased_File = data_folder+'CalBased_Test_Subset.mat'
Test_CalFree_File = data_folder+'CalFree_Test_Subset.mat'


# Training model for estimating SBP. Replace 'SBP' with 'DBP' to train model for DBP.
Train_Data = Build_Dataset(Train_File, 'SBP')
Test_CalBased_Data = Build_Dataset(Test_CalBased_File, 'SBP')
Test_CalFree_Data = Build_Dataset(Test_CalFree_File, 'SBP')
# %% Start model training

if __name__ == '__main__':
    # Initialize model 
    #Seed(6)
    #model = ResNet.Resnet18_1D()
    model = detsikas.DilatedVisualAttentionResidualUNet(input_channels=2, starting_filters=16, activation='relu')
    #model = DeepFFN.DeepFFN(input_size=52, hidden_sizes=[256, 128, 64, 32])
    #Seed(6)
    
    # Prepare settings to be recorded
    Settings = {'BP_optimizer': 'torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0)',
                'trainer': 'Model_Trainer(model,torch.nn.HuberLoss(), BP_optimizer, device, Settings, batch_size=32, num_epochs=75, save_states=True, save_final=True)'
                }
    # Setup training device
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(device)
    print(torch.cuda.get_device_name(0))
    model.to(device)
    # Instantiate optimizer and model trainer
    BP_optimizer = eval(Settings['BP_optimizer'])
    model_trainer = eval(Settings['trainer'])
    # Set the training set and the two setting set under comparison
    model_trainer.Set_Dataset(Train_Data, {
                              'Test_CalBased': Test_CalBased_Data, 'Test_CalFree': Test_CalFree_Data})
    model_trainer.Train_Model()

    # Find the curves of error metrics in the TensorBoard folder
