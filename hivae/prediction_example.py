from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import HIVAE_AK

def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return metrics.roc_auc_score(y_test, y_pred, average=average)
#setup regarding where to save everything
main_directory = '/Users/fathyshalaby/Documents/GitHub/hivae/hivae' #change it to your directory
dataset_name = 'Diabetes'
dataset_path = '{}/data/{}'.format(main_directory,dataset_name)
results_path = '{}/results/{}'.format(main_directory,dataset_name)
network_path = '{}/network/{}'.format(main_directory,dataset_name)
print(dataset_path)
print(results_path)
print(network_path)
datacsvpath = dataset_path+'/data_with_y.csv'
#Import the full dataset( data.csv in the Diabetes Folder in the data folder)
full_dataset_df = pd.read_csv(datacsvpath,header = None)
X = full_dataset_df[full_dataset_df.columns[[0,1,2,3,4,5,6,7,8,9]]]
Y = full_dataset_df[10]
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
    ('AGE', 'pos', 1, None),
    ('SEX', 'cat', 2, 2),
    ('BMI', 'pos', 1, None),
    ('BP', 'pos', 1, None),
    ('S1', 'pos', 1, None),
    ('S2', 'pos', 1, None),
    ('S3', 'pos', 1, None),
    ('S4', 'pos', 1, None),
    ('S5', 'pos', 1, None),
    ('S6', 'pos', 1, None),
    # ('Y','pos',1,None)
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
train_encoded =hivae.training_ak(train_data,epochs=500,results_path=results_path,true_missing_mask=missing_true_train)
(test_data, test_data_reconstructed, test_data_decoded, test_data_embedded_z, test_data_embedded_s) = hivae.training_ak(test_data,epochs=1,results_path=results_path,train_or_test=False,restore_session=True,true_missing_mask=missing_true_test)
test_embeded = test_data_embedded_z

#initilizing the differnt models
RF = RandomForestClassifier()
AB = AdaBoostClassifier()
svc = SVC()
mlp = MLPClassifier()
GB = GradientBoostingClassifier()
#training the different models with original data (not embeded)
RF.fit(train_data,y_train)
AB.fit(train_data,y_train)
svc.fit(train_data,y_train)
mlp.fit(train_data,y_train)
GB.fit(train_data,y_train)
#testing the differnt models with original data(not embeded)
model_auc = {}
v = RF.predict(test_data)
auc = multiclass_roc_auc_score(y_test, v)
model_auc['RF_data']= auc
v = AB.predict(test_data)
auc = multiclass_roc_auc_score(y_test, v)
model_auc['AB_data']= auc
v = svc.predict(test_data)
auc = multiclass_roc_auc_score(y_test, v)
model_auc['svc_data']= auc
v = mlp.predict(test_data)
auc = multiclass_roc_auc_score(y_test, v)
model_auc['mlp_data']= auc
v = GB.predict(test_data)
auc = multiclass_roc_auc_score(y_test, v)
model_auc['GB_data']= auc
#retrain models but now with embeeding from hivae
RF = RandomForestClassifier()
AB = AdaBoostClassifier()
svc = SVC()
mlp = MLPClassifier()
GB = GradientBoostingClassifier()
RF.fit(train_encoded,y_train)
AB.fit(train_encoded,y_train)
svc.fit(train_encoded,y_train)
mlp.fit(train_encoded,y_train)
GB.fit(train_encoded,y_train)
#testing the differnt models with embeded
v = RF.predict(test_embeded)
auc = multiclass_roc_auc_score(y_test, v)
model_auc['RF_embeed']= auc
v = AB.predict(test_embeded)
auc = multiclass_roc_auc_score(y_test, v)
model_auc['AB_embeed']= auc
v = svc.predict(test_embeded)
auc = multiclass_roc_auc_score(y_test, v)
model_auc['svc_embeed']= auc
v = mlp.predict(test_embeded)
auc = multiclass_roc_auc_score(y_test, v)
model_auc['mlp_embeed']= auc
v = GB.predict(test_embeded)
auc = multiclass_roc_auc_score(y_test, v)
model_auc['GB_embeed']= auc

print(model_auc)
