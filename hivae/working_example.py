import pandas as pd
import read_functions
import HIVAE
#Example 1  if only the files are given
dataset = '/Adult/'
types_file ='Adult/data_types.csv'
miss_file = "/Adult/Missing/20_1.csv',"
true_miss_file ="/Adult/MissingTrue.csv"
train_data, types_dict, miss_mask, true_miss_mask, n_samples = read_functions.read_data(dataset,types_file,miss_file,true_miss_file)#take the files and produce the input for the model, didnt produce datafrane yet.
hi = HIVAE.HIVAE()
hiave = hi.training(train_data,types_dict,miss_mask,true_miss_mask,n_samples,10,200)
p = hi.testing(200)
#Example 2  if you run this and use dataframes as input
diabetes_df = pd.read_csv('data/Diabetes/data.csv')
types_dict = {
    'AGE':'count',
    'SEX':'cat',
    'BMI':'pos',
    'BP':'pos',
    'S1':'pos',
    'S2':'pos',
    'S3':'pos',
    'S4':'pos',
    'S5':'pos',
    'S6':'pos',
    'Y':'pos'
}
train_data, types_dict, miss_mask, true_miss_mask, n_samples = read_functions.read_data_df_as_input(dataset,types_file)#take the files and produce the input for the model, didnt produce datafrane yet.
hi = HIVAE.HIVAE()
hiave = hi.training(train_data,types_dict,miss_mask,true_miss_mask,n_samples,10,200)
p = hi.testing(200)