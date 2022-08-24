import warnings

warnings.simplefilter(action='ignore', category=UserWarning)

import os.path
import pandas as pd

import HIVAE_AK
import read_functions




# define the dataset under investigation
dataset_name = 'Diabetes'

# setting up types definitions for this dataset
hivae_types_hash = {
    'AGE':('count',1,None),
    'SEX':('cat',2,2),
    'BMI':('pos',1,None),
    'BP':('pos',1,None),
    'S1':('pos',1,None),
    'S2':('pos',1,None),
    'S3':('pos',1,None),
    'S4':('pos',1,None),
    'S5':('pos',1,None),
    'S6':('pos',1,None),
    'Y':('count',1,None)
    }


# could be a one-liner. But, one might want to use this more often    
def construct_hivae_types_list(hivae_types_hash, variables_list=[]):
    return [tuple([hkey]+list(hivae_types_hash[hkey])) for hkey in variables_list]

# variables to use
hivae_variables_all = ['AGE','SEX','BMI','BP','S1','S2','S3','S4','S5','S6','Y']
hivae_variables     = ['AGE','SEX','BMI','BP','S1','S2','S3','S4','S5','S6']

hivae_type_list = construct_hivae_types_list(hivae_types_hash, variables_list=hivae_variables)

print(hivae_type_list)

# define network structure
n_components = 5      # dimensions of the embedding
batchsize    = 32

hivae_network_dict = {
        'batch_size' : batchsize,
        'model_name':  'model_HIVAE_inputDropout',
        'dim_z':       n_components,
        'dim_y':       10,
        'dim_s':       5, # internal max clusters
        }

# some directory settings for sacing results and networks for later use

# where to save
save_directory = './save_results'
data_directory = './data'
# automatically generate directories, given the dataset name
data_path    = '{}/{}/'.format(data_directory,dataset_name)
network_path = '{}/{}/network/'.format(save_directory,dataset_name)
results_path = '{}/{}/results/'.format(save_directory,dataset_name)

# read in data

data_file = '{}/diabetes.tab.txt'.format(data_path)
data_df = pd.read_csv(data_file,header=0,names=hivae_variables_all,delimiter='\t')
data_df.insert(0,'ID',['{:s}'.format(str(x+1).zfill(3)) for x in range(len(data_df))])
print('Len data = ',len(data_df))

test_data_df  = data_df.sample(frac=0.3,replace=False,random_state=33)
print('Len data (test)  = ',len(test_data_df))
train_data_df = data_df.loc[set(data_df.index)-set(test_data_df.index)]
print('Len data (train) = ',len(train_data_df))


# create the network
hivae_model = HIVAE_AK.HIVAE(hivae_type_list,hivae_network_dict,network_path,results_path)
print(dir(hivae_model))

hivae_model.training_ak(train_data_df[hivae_variables],
                      epochs=50,
                      learning_rate=1e-3,
                      results_path=network_path,
                      restore=False,
                      train_or_test=True,
                      restore_session=False,
                      true_missing_mask=None,
                      missing_mask=None,
                      verbosity=3)

print(network_path)

# reconstruct using the trained network
(train_res_data, train_res_data_reconstructed, train_res_data_decoded, train_res_data_embedded_z, train_res_data_embedded_s) = hivae_model.training_ak(train_data_df[hivae_variables],epochs=1,results_path=results_path,train_or_test=False,restore_session=True,true_missing_mask=None)
# (test_data, test_data_reconstructed, test_data_decoded, test_data_embedded_z, test_data_embedded_s) = hivae_model.training_ak(train_data_df[hivae_variables],epochs=1,results_path=results_path,train_or_test=False,restore_session=True,true_missing_mask=None)
#print(train_res_data[:10])
#print(train_res_data_reconstructed[:10])
