import pandas as pd
import os,os.path
import read_functions
import HIVAE_AK
import numpy as np
import pprint
pprinter = pprint.PrettyPrinter(depth=3)

#data_directory = '/data/projects/vectorisation/HI-VAE/data'
main_directory = '/Users/karwath/develop/GANs/hivae/hivae'
dataset_name = 'Adult'
dataset_path = '{}/data/{}'.format(main_directory,dataset_name)
results_path = '{}/results/{}'.format(main_directory,dataset_name)
network_path = '{}/network/{}'.format(main_directory,dataset_name)

print(dataset_path)
print(results_path)
print(network_path)

data_file_org = '{}/data_org.csv'.format(dataset_path)
data_file     = '{}/data.csv'.format(dataset_path)
data_file_id  = '{}/data_id.csv'.format(dataset_path)
train_file    = '{}/data_train.csv'.format(dataset_path)
test_file     = '{}/data_test.csv'.format(dataset_path)

train_data = None
test_data  = None

# if train/test does not exist - create from new
if os.path.exists(data_file):
    data_df = pd.read_csv(data_file_org,header=-1)
    data_df.insert(0,'ID',['{:s}'.format(str(x).zfill(3)) for x in range(len(data_df))])
    print('Len data = ',len(data_df))
    #print(len(train_data_df))
    # save data including IDs 
    data_df.to_csv(data_file_id,header=False,index=False)
    # drop IDs again
    data_df = data_df.drop('ID', axis=1)
    data_df.to_csv(data_file,header=False,index=False)

    test_data = data_df.sample(frac=0.3,replace=False,random_state=33)
    train_data = data_df.loc[set(data_df.index)-set(test_data.index)]

    train_data.to_csv(train_file,header=False,index=False)
    test_data.to_csv(test_file,header=False,index=False)
#if not os.path.exists(train_file):
else:
    train_data = pd.read_csv(train_file)
    test_data  = pd.read_csv(test_file)


# construct missing data mask
missing_true_train = pd.DataFrame()
for x in list(train_data.columns.values):
    missing_true_train[x] = train_data[x].isna().map({True:0,False:1})

missing_true_test = pd.DataFrame()
for x in list(test_data.columns.values):
    missing_true_test  =  test_data[x].isna().map({True:0,False:1})
    
    
print(len(train_data))
print(list(train_data.index[:20]))
print(len(test_data))
print(list(test_data.index[:20]))

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
    'dim_z': 7,
    'dim_y': 10,
    'dim_s': 3,
}


hivae = HIVAE_AK.HIVAE(types_list,network_dict,network_path,results_path)

hivae.training_ak(train_data,epochs=200,results_path=results_path)
#hivae.train(df_test,results_path=results_path)


hivae = HIVAE_AK.HIVAE(types_list,network_dict,network_path,results_path)

hivae.training_ak(train_data,epochs=30,results_path=results_path,true_missing_mask=missing_true_train)
(test_data, test_data_reconstructed, test_data_decoded, test_data_embedded_z, test_data_embedded_s) = hivae.training_ak(test_data,epochs=1,results_path=results_path,train_or_test=False,restore_session=True,true_missing_mask=missing_true_test)
print(test_data_embedded_z)
