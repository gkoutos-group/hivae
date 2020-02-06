import pandas as pd
import read_functions
import HIVAE_AK

#data_directory = '/data/projects/vectorisation/HI-VAE/data'
main_directory = '/Users/karwath/develop/GANs/hivae/hivae'
dataset_name = 'defaultCredit'
dataset_path = '{}/data/{}'.format(main_directory,dataset_name)
results_path = '{}/results/{}'.format(main_directory,dataset_name)
network_path = '{}/network/{}'.format(main_directory,dataset_name)

print(dataset_path)
print(results_path)
print(network_path)

#train_file = '{}/data_train.csv'.format(dataset_path)
train_file = '{}/data.csv'.format(dataset_path)
test_file  = '{}/data.csv'.format(dataset_path)

train_data = pd.read_csv(train_file)
test_data  = pd.read_csv(test_file)

types_list = [
    ('V','pos',1,None),
    ('V','cat',3,3),
    ('V','cat',7,7),
    ('V','cat',4,4),
    ('V','count',1,None),
    ('V','ordinal',11,11),
    ('V','ordinal',11,11),
    ('V','ordinal',11,11),
    ('V','ordinal',11,11),
    ('V','ordinal',11,11),
    ('V','ordinal',11,11),
    ('V','real',1,None),
    ('V','real',1,None),
    ('V','real',1,None),
    ('V','real',1,None),
    ('V','real',1,None),
    ('V','real',1,None),
    ('V','pos',1,None),
    ('V','pos',1,None),
    ('V','pos',1,None),
    ('V','pos',1,None),
    ('V','pos',1,None),
    ('V','pos',1,None),
    ('V','cat',2,2),
    ]

network_dict = {
    'batch_size' : 32,
    'model_name': 'model_HIVAE_inputDropout',
    'dim_z': 5,
    'dim_y': 5,
    'dim_s': 3,
}


hivae = HIVAE_AK.HIVAE(types_list,network_dict,network_path,results_path)

hivae.training_ak(train_data,epochs=200,results_path=results_path)
#hivae.train(df_test,results_path=results_path)

