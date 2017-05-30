import zipfile
import pandas as pd
from keras.optimizers import SGD
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy as sp
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, PReLU, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import random


def logloss(act, pred):
  epsilon = 1e-15
  pred = sp.maximum(epsilon, pred)
  pred = sp.minimum(1-epsilon, pred)
  ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
  ll = ll * -1.0/len(act)
  return ll
top_model_weights_path = 'saved/First_try.h5'
# load data
data_root = "./pre"
dfTrain = pd.read_csv("%s/train.csv"%data_root)
dfTest = pd.read_csv("%s/test.csv"%data_root)
dfAd = pd.read_csv("%s/ad.csv"%data_root)
dfPos = pd.read_csv("%s/position.csv"%data_root)
dfAppc = pd.read_csv("%s/app_categories.csv"%data_root)
dfUser = pd.read_csv("%s/user.csv"%data_root)
dfUsera = pd.read_csv("%s/user_app_actions.csv"%data_root)
# dfUseri = pd.read_csv("%s/user_installedapps.csv"%data_root)

#Merge data
dfAd = pd.merge(dfAd, dfAppc, on="appID")
#Train
dfTrain = pd.merge(dfTrain, dfAd, on="creativeID")
dfTrain = pd.merge(dfTrain, dfPos, on="positionID")
dfTrain = pd.merge(dfTrain, dfUser, on="userID")
#Test
dfTest = pd.merge(dfTest, dfAd, on="creativeID")
dfTest = pd.merge(dfTest, dfPos, on="positionID")
dfTest = pd.merge(dfTest, dfUser, on="userID")

y_train_pro = dfTrain["label"].values
encoder = LabelEncoder()
# feature engineering/encoding
X_train = dfTrain.values[:,3:]
X_test = dfTest.values[:,3:]
enc = OneHotEncoder()
feats = ["appID", "appCategory", "age","gender","education","marriageStatus","haveBaby", "sitesetID",
         "positionType", "connectionType", "telecomsOperator"]
for i,feat in enumerate(feats):
    x_train = enc.fit_transform(dfTrain[feat].values.reshape(-1, 1))
    x_test = enc.transform(dfTest[feat].values.reshape(-1, 1))
    X_train, X_test = sparse.hstack((X_train, x_train)), sparse.hstack((X_test, x_test))

X_train = X_train.todense()
X_test = X_test.todense()
X_train = numpy.array(X_train)
X_test = numpy.array(X_test)
X_train_one = []
X_train_zero = []
for i in range(0,len(X_train)):
    if y_train_pro[i] == 1:
        X_train_one.append(X_train[i])
    else:
        X_train_zero.append(X_train[i])
random.shuffle(X_train_zero)
y_train_one = [1]*len(X_train_one)
for i in range(0,5*len(X_train_one)):
    X_train_one.append(X_train_zero[i])
    y_train_one.append(0)
X_train_one = numpy.array(X_train_one)
y_train_one = numpy.array(y_train_one)
encoder.fit(y_train_one)
encoded_Y = encoder.transform(y_train_one)
# convert integers to dummy variables (i.e. one hot encoded)
y_train_one = np_utils.to_categorical(encoded_Y)
#PCA
stdsc = StandardScaler(with_mean=False)
X_train_std = stdsc.fit_transform(X_train_one)
X_test_std = stdsc.fit_transform(X_test)
pca = PCA(n_components=100)
X_train_one = pca.fit_transform(X_train_std)
X_test = pca.fit_transform(X_test_std)
X_train, X_val, y_train, y_val = train_test_split(X_train_one, y_train_one, test_size=0.3, random_state=0)

def activation():
    # return Activation("relu")
    return PReLU()

model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], init='glorot_normal'))
# model.add(BatchNormalization(axis=1))
model.add(activation())
model.add(Dense(64, init='glorot_normal'))
# model.add(BatchNormalization(axis=1))
model.add(activation())
model.add(Dense(2,init='glorot_normal'))
# model.add(BatchNormalization(axis=1))
model.add(Activation('sigmoid'))
# Compile model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(X_train_one, y_train_one, nb_epoch=10, verbose=1,validation_data=(X_val,y_val))
model.save_weights(top_model_weights_path)
proba_test = model.predict_proba(X_test)[:,1]

y_pred_train = model.predict_proba(X_train)[:,1]
y_pred_val = model.predict_proba(X_val)[:,1]
y_pred_train_acc = model.predict_classes(X_train)
y_pred_val_acc = model.predict_classes(X_val)
print('Train_Accuracy: %.2f' % accuracy_score(y_train, y_pred_train_acc))
print('Val_Accuracy: %.2f' % accuracy_score(y_val, y_pred_val_acc))
print logloss(y_train,y_pred_train)
print logloss(y_val,y_pred_val)
print logloss(y_val,y_pred_val_acc)

# submission
df = pd.DataFrame({"instanceID": dfTest["instanceID"].values, "proba": proba_test})
df.sort_values("instanceID", inplace=True)
df.to_csv("submission.csv", index=False)
with zipfile.ZipFile("submission.zip", "w") as fout:
    fout.write("submission.csv", compress_type=zipfile.ZIP_DEFLATED)

a = 10