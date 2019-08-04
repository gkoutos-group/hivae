import pandas as pd
import read_functions
import HIVAE_AK

#data_directory = '/data/projects/vectorisation/HI-VAE/data'
main_directory = '/Users/karwath/develop/GANs/hivae/hivae'
dataset_name = 'Diabetes'
dataset_path = '{}/data/{}'.format(main_directory,dataset_name)
results_path = '{}/results/{}'.format(main_directory,dataset_name)
network_path = '{}/network/{}'.format(main_directory,dataset_name)

print(dataset_path)
print(results_path)
print(network_path)

train_file = '{}/data_train.csv'.format(dataset_path)
test_file  = '{}/data_train.csv'.format(dataset_path)
print(train_file)

train_data_df = pd.read_csv(train_file)
#print(len(train_data_df))

#test_data  = pd.read_csv(test_file)
test_data = train_data_df.sample(frac=0.3,replace=False,random_state=33)
train_data = train_data_df.loc[set(train_data_df.index)-set(test_data.index)]
print(len(train_data))
print(list(train_data.index[:20]))
print(len(test_data))
print(list(test_data.index[:20]))

types_list = [
    ('AGE','count',1,None),
    ('SEX','cat',2,2),
    ('BMI','pos',1,None),
    ('BP','pos',1,None),
    ('S1','pos',1,None),
    ('S2','pos',1,None),
    ('S3','pos',1,None),
    ('S4','pos',1,None),
    ('S5','pos',1,None),
    ('S6','pos',1,None),
    # ('Y','pos',1,None)
    ]

network_dict = {
    'batch_size' : 32,
    'model_name': 'model_HIVAE_inputDropout',
    'dim_z': 5,
    'dim_y': 5,
    'dim_s': 3,
}


hivae = HIVAE_AK.HIVAE(types_list,network_dict,network_path,results_path)

hivae.training_ak(train_data,epochs=5,results_path=results_path)
hivae.training_ak(test_data,epochs=1,results_path=results_path,train_or_test=False,restore_session=True)

