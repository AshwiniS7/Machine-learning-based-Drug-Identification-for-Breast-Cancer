#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from chembl_webresource_client.new_client import new_client
import numpy as np
import tensorflow as tf


# In[2]:


# get data on efficacy of drugs against a target protein from chembl
def get_drug_data(target_protein, chembl_idx):
    # find chembl id of target protein
    chembl_result = pd.DataFrame.from_dict(new_client.target.search(target_protein))
    chembl_id = chembl_result["target_chembl_id"][chembl_idx]
    print("chembl_result: ", chembl_result)
    print("chembl_id: ", chembl_id)
    drug_data = new_client.activity.filter(target_chembl_id=chembl_id).filter(standard_type="IC50")
    return pd.DataFrame.from_dict(drug_data)


# In[3]:


# HER2: breast cancer target protein
target_protein = "HER2"
drug_data_all_df = get_drug_data(target_protein, 1)


# In[4]:


# each row represents a different drug that was tested against the target_protein
drug_data_all_df


# In[5]:


# standard_type column is the measurement type (represents drug's efficacy against target protein). 
# In this case, we use IC50, which is the half-maximal inhibitory concentration
drug_data_all_df['standard_type'].unique()


# In[6]:


# standard_value column is the actual measurement itself 
# tells the potency of the drug - the lower the value, the better the drug
# in order to elicit 50% inhibition of the target protein, you would need lower concentration of the drug
drug_data_all_df['standard_value']


# In[7]:


drug_data_all_df['canonical_smiles']


# In[8]:


print(pd.isna(drug_data_all_df.loc[1716, "canonical_smiles"]))


# In[9]:


# some smiles strings are missing so drop those rows
drug_data_df = drug_data_all_df[["molecule_chembl_id", "canonical_smiles", "standard_value"]].copy()
print("Original dataset size: ", len(drug_data_df))
drug_data_df.dropna(inplace=True)
drug_data_df.reset_index(inplace=True)
print("Final dataset size: ", len(drug_data_df))


# In[10]:


from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
# from rdkit.Chem import DataStructs


# In[11]:


# generate morgan fingerprints for each chemical (1D descriptor)
def gen_1d_desc(df):
    morgan_fp_length = 2048
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=morgan_fp_length)
    cols = ['mf_' + str(n) for n in range(morgan_fp_length)]
    morgan_df = pd.DataFrame(columns=cols)
    for idx in range(len(df)):
        print("idx: ", idx)
        print("chembl_id: ", df.loc[idx, 'molecule_chembl_id'])
        print("smiles: ", df.loc[idx, 'canonical_smiles'])
        molecule = Chem.MolFromSmiles(df.loc[idx, 'canonical_smiles'])
        morgan_fp = morgan_gen.GetFingerprint(molecule)
        ## print(morgan_fp)
        feature_vec = np.array(morgan_fp)
        feature_vec = np.asarray(feature_vec).astype('float32')
        morgan_df.loc[len(morgan_df)] = feature_vec
        print(sum(feature_vec))
    return morgan_df


# In[12]:


# generate molecular weight and atomic counts for each chemical (0D descriptor)
def gen_0d_desc(df):
    cols = ['molwt']
    min_atomic_num = 1 # hydrogen
    max_atomic_num = 118 # Oganesson
    cols = cols + ["atmct_" + str(atom) for atom in range(min_atomic_num, max_atomic_num+1)]
    zerod_df = pd.DataFrame(columns=cols)
    
    for idx in range(len(df)):
        print("idx: ", idx)
        print("chembl_id: ", df.loc[idx, 'molecule_chembl_id'])
        print("smiles: ", df.loc[idx, 'canonical_smiles'])
        
        molecule = Chem.MolFromSmiles(df.loc[idx, 'canonical_smiles'])
        molecule_weight = Descriptors.MolWt(molecule)
        # molecular weight
        print("molwt: ", molecule_weight)
        zerod_df.loc[idx, 'molwt'] = float(molecule_weight)
        
        molecule_H = Chem.AddHs(molecule)
        # atom counts
        # initialize all counts to 0
        for atomic_num in range(min_atomic_num, max_atomic_num+1):
            zerod_df.loc[idx, 'atmct_' + str(atomic_num)] = 0.0
            
        for atom in molecule_H.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            # print(atomic_num)
            zerod_df.loc[idx, 'atmct_' + str(atomic_num)] += 1.0
    return zerod_df


# In[41]:


# generate features (0d or 1d descriptors)
def gen_features(zerod = True, oned = True):
    zerod_df = None
    oned_df = None
    chemblid_df = drug_data_df['molecule_chembl_id']
    
    if zerod:
        zerod_df = gen_0d_desc(drug_data_df)
    
    if oned:
        oned_df = gen_1d_desc(drug_data_df)
    
    if (zerod and oned):
        features_df = pd.concat([chemblid_df, zerod_df, oned_df], axis=1)
    elif (zerod):
        features_df = zerod_df
    else:
        features_df = oned_df
        
    print("Features: ", features_df.columns)
    return features_df.astype('float32')


# In[14]:


def gen_pIC50_target(df):
    print("IC50 range: ", df['standard_value'].min(), " - ", df['standard_value'].max())
    # convert IC50 to pIC50
    pIC50 = []
    for val in df['standard_value']:
        # convert nanomolar to molar
        molar = float(val)*(10**-9)
        pIC50.append(-np.log10(molar))
        
    df['pIC50'] = pIC50
    print("pIC50 range: ", df['pIC50'].min(), " - ", df['pIC50'].max())
    return df['pIC50']


# In[15]:


from sklearn.model_selection import train_test_split


# In[42]:


# Uses 0D descriptors (atomic counts, molecular weight) and 1D descriptor (Morgan fingerprint) as features
X = gen_features()


# In[43]:


X


# In[18]:


Y = gen_pIC50_target(drug_data_df)


# In[19]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)


# In[20]:


print("X length: ", len(X))
print("Y length: ", len(Y))
print("X_train length: ", len(X_train))
print("X_test length: ", len(X_test))
print("Y_train length: ", len(Y_train))
print("Y_test length: ", len(Y_test))


# In[21]:


print(Y)


# In[34]:


# Neural network
import keras
from keras.layers import Dense
from keras.models import Sequential
nn_model = Sequential()
nn_model.add(Dense(16, input_dim=2167, activation='relu'))
nn_model.add(Dense(8, activation='relu'))
nn_model.add(Dense(1, activation='linear'))
nn_model.compile(loss='mean_squared_error', optimizer='adam')
nn_model.fit(X_train, Y_train, epochs=100, batch_size=32)


# In[35]:


from sklearn.metrics import mean_squared_error
Y_pred_train = nn_model.predict(X_train)
Y_pred_test = nn_model.predict(X_test)
train_error = mean_squared_error(Y_train, Y_pred_train)
test_error = mean_squared_error(Y_test, Y_pred_test)
print("Train MSE:", train_error)
print("Test MSE:", test_error)


# In[36]:


import matplotlib.pyplot as plt


# In[37]:


plt.scatter(Y_test, Y_pred_test)
plt.xlabel("Y_test")
plt.ylabel("Y_pred")
# plt.xlim(0, 10000)
# plt.ylim(-10000, 80000)


# In[38]:


print(Y_pred_test)


# In[27]:


print(Y_test)


# In[57]:


results = pd.DataFrame(columns=cols)
for idx in range(0, len(Y_test)):
    # print(Y_test.index[idx])
    print("chembl_id: ", drug_data_df.loc[Y_test.index[idx], 'molecule_chembl_id'])
    print("Y_test: ", Y_test[Y_test.index[idx]])
    print("Y_test_pred: ", Y_pred_test[idx][0])
    
    cols = ["chembl_id", "Y_test", "Y_test_pred"]
    results.loc[idx, "chembl_id"] = drug_data_df.loc[Y_test.index[idx], 'molecule_chembl_id']
    results.loc[idx, "Y_test"] = Y_test[Y_test.index[idx]]
    results.loc[idx, "Y_test_pred"] = Y_pred_test[idx][0]


# In[59]:


results.sort_values(by=['Y_test_pred'], ascending=False)


# In[ ]:




