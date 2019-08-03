import pandas as pd
import read_functions
import HIVAE_AK

#data_directory = '/data/projects/vectorisation/HI-VAE/data'
main_directory = '/Users/karwath/develop/GANs/hivae/hivae'
dataset_name = 'Adult'
dataset_path = '{}/data/{}'.format(main_directory,dataset_name)
results_path = '{}/results/{}'.format(main_directory,dataset_name)
network_path = '{}/network/{}'.format(main_directory,dataset_name)

print(dataset_path)
print(results_path)
print(network_path)

#train_file = '{}/data_train.csv'.format(dataset_path)
train_file = '{}/data.csv'.format(dataset_path)
test_file  = '{}/data.csv'.format(dataset_path)
print(train_file)

train_data = pd.read_csv(train_file)
test_data  = pd.read_csv(test_file)

types_list = [
    ('V1','count',1,None),
    ('V2','çat',7,7),
    ('V3','pos',1,None),
    ('V4','ordinal',16,16),
    ('V5','cat',7,7),
    ('V6','cat',14,14),
    ('V7','cat',6,6),
    ('V8','cat',5,5),
    ('V9','cat',2,2),
    ('V10','pos',1,None),
    ('V11','pos',1,None),
    ('V12','count',1,None)
    ]

network_dict = {
    'batch_size' : 32,
    'model_name': 'model_HIVAE_inputDropout',
    'dim_z': 5,
    'dim_y': 5,
    'dim_s': 3,
}


hivae = HIVAE_AK.HIVAE(types_list,network_dict,network_path)

hivae.training_ak(train_data,epochs=200,results_path=results_path)
#hivae.train(df_test,results_path=results_path)

