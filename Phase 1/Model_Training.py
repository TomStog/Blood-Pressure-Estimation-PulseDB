import torch
import torch.utils.data as data
import random
import numpy as np
from mat73 import loadmat
from Model_Def.MyTrainer import Model_Trainer
from Model_Def import ResNet, UResIncNet, ResIncNet, UNetViT1DRegressor

print(torch.version.cuda)

def Seed(seed): 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class Dataset(data.Dataset):
    def __init__(self, Input, Label):
        self.Input = Input
        self.Label = Label

    def __len__(self):
        return len(self.Input)

    def __getitem__(self, idx):
        return self.Input[idx, :], self.Label[[idx]]

#class Dataset(data.Dataset):
#    def __init__(self, Input, Label, Age, Gender):
#        self.Input = Input
#        self.Label = Label
#        self.Age = Age
#        self.Gender = Gender

#    def __len__(self):
#        return len(self.Input)

#    def __getitem__(self, idx):
#        return self.Input[idx, :], self.Label[[idx]], self.Age[[idx]], self.Gender[[idx]]

#def Build_Dataset(Path, Label):
#    Data = loadmat(Path)
#    # Get the first two channels, which are the ECG and the PPG signals
#    return Dataset(Data['Subset']['Signals'][:, 0:2, :], Data['Subset'][Label])

def Build_Dataset(Path, Label):
    Data = loadmat(Path)
    # Get the first two channels (ECG and PPG signals)
    original_signals = Data['Subset']['Signals'][:, 0:2, :]
    
    # Calculate derivatives for each signal
    # First derivatives (gradient)
    first_derivatives = np.gradient(original_signals, axis=2)
    
    # Second derivatives (gradient of first derivatives)
    second_derivatives = np.gradient(first_derivatives, axis=2)
    
    # Concatenate: [original_ch1, original_ch2, 1st_deriv_ch1, 1st_deriv_ch2, 2nd_deriv_ch1, 2nd_deriv_ch2]
    extended_signals = np.concatenate([
        original_signals,      # channels 0-1: original signals
        first_derivatives,     # channels 2-3: first derivatives
        second_derivatives     # channels 4-5: second derivatives
    ], axis=1)

    Age=Data['Subset']['Age']
    # Access Gender of the subject corresponding to each of the 10-s segment
    Gender=np.array(Data['Subset']['Gender']).squeeze()
    # Convert Gender to numerical 0-1 labels
    Gender=(Gender=='M').astype(float)
    
    return Dataset(original_signals, Data['Subset'][Label])
    #return Dataset(extended_signals, Data['Subset'][Label], Age, Gender)

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
    model = UNetViT1DRegressor.UNetViT1DRegressor()
    #model = UResIncNet.UResIncNet(nclass=1, in_chans=6, first_out_chans=8, max_channels=256, depth=6, 
    #                     kernel_size=3, layers=1, sampling_factor=2, sampling_method="conv_stride", 
    #                     dropout=0.1, skip_connection=True, extra_final_conv=False, custom_weight_init=False)
    #model = ResIncNet.ResIncNet(nclass=1, in_size=1250, in_chans=6, max_channels=128, depth=4, kernel_size=3, layers=1,
    #             sampling_factor=2, sampling_method="not_conv_stride", skip_connection=True, custom_weight_init=False)
    #Seed(6)
    
    # Prepare settings to be recorded
    Settings = {'BP_optimizer': 'torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0)',
                'trainer': 'Model_Trainer(model,torch.nn.HuberLoss(), BP_optimizer, device, Settings, batch_size=32, num_epochs=70, save_states=True, save_final=True)'
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