import pandas as pd
import read_functions
import HIVAE

dataset_name = 'Cardiology'
dataset_path = '/data/projects/vectorisation/HI-VAE/data/Cardiology/'
train_file = '{}/train_data.csv'.format(dataset_path)
test_file  = '{}/test_data.csv'.format(dataset_path)

df_train = pd.read_csv(train_file)
df_test  = pd.read_csv(test_file)

results_path = '/data/projects/vectorisation/HI-VAE/data/Cardiology/results/'
network_path = '/data/projects/vectorisation/HI-VAE/data/Cardiology/networks/'

types_dict = {
    'gender':'countcat',2,2
    'age':'count',1
    'HGHT':'pos',1
    'WGHT':'pos',1
    'BMI':'pos',1
    'HR':'pos',1
    'BPDIA':'pos',1
    'BPSYS':'pos',1
    'O2SATS':'pos',1
}

network_dict = {
    'batch_size' = 32,
    'model'      = 'model_HIVAE_inputDropout',
    'z_dim'      = 5,
    'y_dim'      = 5,
    's_dim'      = 3,    
}


hivae = HIVAE.HIVAE(types_dict,network_dict,network_path)

hivae.train(df_train,epochs=200,true_miss_mask=None,results_path)
hivae.train(df_test,true_miss_mask=None,results_path)

