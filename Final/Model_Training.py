import torch
import torch.utils.data as data
from torch.utils.data import ConcatDataset
import os
#import numpy as np
import progressbar
from mat73 import loadmat
from Model_Def.MyTrainer import Model_Trainer
from Model_Def import UNetViT1DRegressor, detsikas, detsikas2, detsikas3

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

    SBP = Data['Subset']['SBP'].squeeze()
    DBP = Data['Subset']['DBP'].squeeze()

    # Access Age of the subject corresponding to each of the 10-s segment
    Age = Data['Subset']['Age'].squeeze()

    age_mask = Age >= 17
    sbp_mask = (SBP >= 57) & (SBP <= 180)
    dbp_mask = (DBP >= 25) & (DBP <= 100)
    #subset_mask = (np.isnan(Height) & np.isnan(Weight))#use only &(and) |(or)

    mask = age_mask & sbp_mask & dbp_mask
    print(f"Number of rows matching all conditions: {mask.sum()}")
    
    return Dataset(Data['Subset']['Signals'][mask, 0:2, :1248], Data['Subset'][Label][mask])

torch.cuda.empty_cache()
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device)
print(torch.cuda.get_device_name(0))

data_folder = './Subset_Files/'
Train_File = data_folder + 'Train_Subset.mat'
Test_CalBased_File = data_folder + 'CalBased_Test_Subset.mat'
#Test_CalFree_File = data_folder + 'CalFree_Test_Subset.mat'
AAMI_Cal_File = data_folder + 'AAMI_Cal_Subset.mat'
AAMI_Test_File = data_folder + 'AAMI_Test_Subset.mat'

# Training model for estimating SBP. Replace 'SBP' with 'DBP' to train model for DBP.
Train_Data = Build_Dataset(Train_File, 'DBP')
#AAMI_Cal = Build_Dataset(AAMI_Cal_File, 'DBP')

#Train_Data = ConcatDataset([Train, AAMI_Cal])
Test_CalBased_Data = Build_Dataset(Test_CalBased_File, 'DBP')
#Test_CalFree_Data = Build_Dataset(Test_CalFree_File, 'DBP')
#AAMI_Test_Data = Build_Dataset(AAMI_Test_File, 'DBP')

if __name__ == '__main__':
    
    # Initialize model
    model = detsikas3.DualSignalDilatedAttnUNet1D(base_ch=32, levels=4, dropout=0.10)
        
    # Prepare settings to be recorded
    Settings = {'BP_optimizer': 'torch.optim.Adam(model.parameters(), lr=5*1e-4, weight_decay=0)',
                'trainer': 'Model_Trainer(model, torch.nn.HuberLoss(), BP_optimizer, device, Settings, batch_size=32, num_epochs=75, save_states=False, save_final=True)'
                }#torch.nn.MSELoss() torch.nn.HuberLoss(delta=5.0) torch.nn.L1Loss()
    
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
                              'Test_CalBased': Test_CalBased_Data})#, 'Test_AAMI': AAMI_Test_Data})
    model_trainer.Train_Model()
