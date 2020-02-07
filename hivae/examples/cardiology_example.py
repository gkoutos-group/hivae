import pandas as pd
import read_functions
import HIVAE

dataset_name = 'Cardiology'
dataset_path = '/data/projects/vectorisation/HI-VAE/data/Cardiology/'
train_file = '{}/data_train.csv'.format(dataset_path)
test_file  = '{}/data_train.csv'.format(dataset_path)

df_train = pd.read_csv(train_file)
df_test  = pd.read_csv(test_file)

results_path = '/data/projects/vectorisation/HI-VAE/data/Cardiology/results/'
network_path = '/data/projects/vectorisation/HI-VAE/data/Cardiology/networks/'

types_dict = {
    'gender':['countcat',2,2],
    'age':['count',1,None],
    'HGHT':['pos',1,None],
    'WGHT':['pos',1,None],
    'BMI':['pos',1,None],
    'HR':['pos',1,None],
    'BPDIA':['pos',1,None],
    'BPSYS':['pos',1,None],
    'O2SATS':['pos',1,None]
}

network_dict = {
    'batch_size' : 32,
    'model_name': 'model_HIVAE_inputDropout',
    'dim_z': 5,
    'dim_y': 5,
    'dim_s': 3,
}


hivae = HIVAE.HIVAE(types_dict,network_dict,network_path)

hivae.train(df_train,epochs=200,results_path=results_path)
#hivae.train(df_test,results_path=results_path)
