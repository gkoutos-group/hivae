import pandas
import read_functions
import HIVAE
dataset = '/Adult/'
types_file ='Adult/data_types.csv'
miss_file = "/Adult/Missing/20_1.csv',"
true_miss_file ="/Adult/MissingTrue.csv"
train_data, types_dict, miss_mask, true_miss_mask, n_samples = read_functions.read_data(dataset,types_file,miss_file,true_miss_file)#take the files and produce the input for the model, didnt produce datafrane yet.
hi = HIVAE.HIVAE()
hiave = hi.training(train_data,types_dict,miss_mask,true_miss_mask,n_samples,10,200)
p = hi.testing(200)