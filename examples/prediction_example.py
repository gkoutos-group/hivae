from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import cross_val_score
import pandas as pd
import HIVAE_AK

#setup regarding where to save everything
models_data = [] #list which will contain all the models which will be later used later
models_embed = [] #list which will contain all the models which will be later used later

main_directory = '/Users/fathyshalaby/Documents/GitHub/hivae/hivae' #change it to your directory
dataset_name = 'Diabetes'
dataset_path = '{}/data/{}'.format(main_directory,dataset_name)
results_path = '{}/results/{}'.format(main_directory,dataset_name)
network_path = '{}/network/{}'.format(main_directory,dataset_name)
print(dataset_path)
print(results_path)
print(network_path)
datacsvpath = dataset_path+'/pima-indians-diabetes.data.csv'
#Import the full dataset( data.csv in the Diabetes Folder in the data folder)
full_dataset_df = pd.read_csv(datacsvpath,header = None)
X = full_dataset_df[full_dataset_df.columns[[0,1,2,3,4,5,6,7]]]
Y = full_dataset_df[8].values
train_data,test_data,y_train,y_test = train_test_split(X,Y,test_size=.4,random_state=42)
#setup for the Heterogenous incomplete Varational Autoencoder
missing_true_train = pd.DataFrame()
for x in list(train_data.columns.values):
    missing_true_train[x] = train_data[x].isna().map({True:0,False:1})
missing_true_test = pd.DataFrame()
for x in list(test_data.columns.values):
    missing_true_test  =  test_data[x].isna().map({True:0,False:1})
types_list_d = {}
types_list_d['Diabetes'] = [
    ('V1', 'pos', 1, None),
    ('V2', 'pos', 1, None),
    ('V3', 'pos', 1, None),
    ('V4', 'pos', 1, None),
    ('V5', 'pos', 1, None),
    ('V6', 'pos', 1, None),
    ('V7', 'pos', 1, None),
    ('V8', 'count', 1, None),
    ]
types_list = types_list_d[dataset_name]

network_dict = {
    'batch_size' : len(train_data),
    'model_name': 'model_HIVAE_inputDropout',
    'dim_z': 7,
    'dim_y': 10,
    'dim_s': 3,
}
hivae = HIVAE_AK.HIVAE(types_list,network_dict,network_path,results_path)
train_encoded =hivae.training_ak(train_data,epochs=10000,results_path=results_path,true_missing_mask=missing_true_train)
(test_data, test_data_reconstructed, test_data_decoded, test_data_embedded_z, test_data_embedded_s) = hivae.training_ak(test_data,epochs=1,results_path=results_path,train_or_test=False,restore_session=True,true_missing_mask=missing_true_test)
test_embeded = test_data_embedded_z
#initilizing the differnt models
RF = RandomForestClassifier()
AB = AdaBoostClassifier()
mlp = MLPClassifier()
GB = GradientBoostingClassifier()
models_data.append(RF)
models_data.append(AB)
models_data.append(mlp)
models_data.append(GB)
#training the different models with original data (not embeded)
RF.fit(train_data,y_train)
AB.fit(train_data,y_train)
mlp.fit(train_data,y_train)
GB.fit(train_data,y_train)
#testing the differnt models with original data(not embeded)
model_auc = {}
model_cf = {}
v = RF.predict(test_data)
metrics.confusion_matrix(y_test, v)
auc = metrics.roc_auc_score(y_test, v)
cf = metrics.confusion_matrix(y_test, v)
model_auc['RF_data']= auc
model_cf['RF_data']= cf
v = AB.predict(test_data)
auc = metrics.roc_auc_score(y_test, v)
cf = metrics.confusion_matrix(y_test, v)
model_cf['AB_data']= cf
model_auc['AB_data']= auc
v = mlp.predict(test_data)
auc = metrics.roc_auc_score(y_test, v)
cf = metrics.confusion_matrix(y_test, v)
model_auc['mlp_data']= auc
model_cf['mlp_data']= cf
v = GB.predict(test_data)
auc = metrics.roc_auc_score(y_test, v)
cf = metrics.confusion_matrix(y_test, v)
model_auc['GB_data']= auc
model_cf['GB_data']= cf
#retrain models but now with embeeding from hivae
RF = RandomForestClassifier()
AB = AdaBoostClassifier()
mlp = MLPClassifier()
GB = GradientBoostingClassifier()
RF.fit(train_encoded,y_train)
AB.fit(train_encoded,y_train)
mlp.fit(train_encoded,y_train)
GB.fit(train_encoded,y_train)
models_embed.append(RF)
models_embed.append(AB)
models_embed.append(mlp)
models_embed.append(GB)
#testing the differnt models with embeded
v = RF.predict(test_embeded)
auc = metrics.roc_auc_score(y_test, v)
cf = metrics.confusion_matrix(y_test, v)
model_auc['RF_emebed']= auc
model_cf['RF_embed']= cf
v = AB.predict(test_embeded)
auc = metrics.roc_auc_score(y_test, v)
cf = metrics.confusion_matrix(y_test, v)
model_auc['AB_embed']= auc
model_cf['AB_embed']= cf
v = mlp.predict(test_embeded)
auc = metrics.roc_auc_score(y_test, v)
cf = metrics.confusion_matrix(y_test, v)
model_auc['mlp_embed']= auc
model_cf['ml_embed']= cf
v = GB.predict(test_embeded)
auc = metrics.roc_auc_score(y_test, v)
cf = metrics.confusion_matrix(y_test, v)
model_auc['GB_embed']= auc
model_cf['GB_embed']= cf

print(model_auc)
print(model_cf)

'''#cross validation
for model in models_data:
    scores = cross_val_score(model, X, Y, cv=5)
    print(scores.mean())
missing_true_train = pd.DataFrame()
for x in list(X.columns.values):
    missing_true_train[x] = X[x].isna().map({True:0,False:1})
missing_true_test = pd.DataFrame()
for x in list(X.columns.values):
    missing_true_test  =  X[x].isna().map({True:0,False:1})
network_dict = {
    'batch_size' : len(X),
    'model_name': 'model_HIVAE_inputDropout',
    'dim_z': 7,
    'dim_y': 10,
    'dim_s': 3,}
hivae = HIVAE_AK.HIVAE(types_list,network_dict,network_path,results_path)
train_encoded =hivae.training_ak(X,epochs=500,results_path=results_path,true_missing_mask=missing_true_train)
(test_data, test_data_reconstructed, test_data_decoded, test_data_embedded_z, test_data_embedded_s) = hivae.training_ak(X,epochs=1,results_path=results_path,train_or_test=False,restore_session=True,true_missing_mask=missing_true_test)
X_embed = test_data_embedded_z
for model in models_embed:
    scores = cross_val_score(model, X_embed, Y, cv=5)
    print(scores.mean())
'''