import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif, chi2


label_encoder = LabelEncoder()

train_csv = pd.read_csv('train_ver2.csv', low_memory=False)
test_csv = pd.read_csv('test_ver2.csv', low_memory=False)

print(train_csv)

train_csv['indrel_1mes'] = train_csv['indrel_1mes'].replace(np.nan, 1)
train_csv['indrel_1mes'] = train_csv['indrel_1mes'].replace('P', 0)
train_csv['indrel_1mes'] = train_csv['indrel_1mes'].astype('float').astype(int)
# print(train_csv['indrel_1mes'].isna().sum())
train_csv['indrel_1mes'].unique()
train_csv.dropna(how='all', subset=['ind_nomina_ult1'], inplace=True)
train_csv.dropna(how='all', subset=['ind_nom_pens_ult1'], inplace=True)

train_csv['ind_empleado'] = train_csv['ind_empleado'].fillna('N')  # N appears the most, so fill nan to N
train_csv['ind_empleado'].unique()

test_csv['ind_empleado'] = test_csv['ind_empleado'].fillna('N')  # N appears the most, so fill nan to N
test_csv['ind_empleado'].unique()

train_csv['ult_fec_cli_1t'] = train_csv['ult_fec_cli_1t'].replace(np.nan, 0)
train_csv['ult_fec_cli_1t'] = train_csv['ult_fec_cli_1t'].where(train_csv['ult_fec_cli_1t'] == 0, 1)  # Primary customer: 1, nan: 0
train_csv['ult_fec_cli_1t'].unique()

test_csv['ult_fec_cli_1t'] = test_csv['ult_fec_cli_1t'].replace(np.nan, 0)
test_csv['ult_fec_cli_1t'] = test_csv['ult_fec_cli_1t'].where(test_csv['ult_fec_cli_1t'] == 0, 1)  # Primary customer: 1, nan: 0
test_csv['ult_fec_cli_1t'].unique()

# print(train_csv.loc[:, 'conyuemp'])
train_csv.drop(['conyuemp'], axis = 1, inplace = True)  # too many nan, so drop it
test_csv.drop(['conyuemp'], axis = 1, inplace = True)

# print(train_csv['renta'].mode())
train_csv['renta'] = train_csv['renta'].fillna(train_csv['renta'].mode()[0])
test_csv['renta'] = test_csv['renta'].replace('         NA', np.nan)
test_csv['renta'] = test_csv['renta'].fillna(test_csv['renta'].mode()[0])
test_csv['renta'] = test_csv['renta'].astype(float)
test_csv['renta'].unique()
# print(test_csv['renta'].mode()[0])

# print(train_csv['cod_prov'].head(4))
# because there are only few number of nan, I drop the whole row
train_csv.dropna(how='all', subset=['cod_prov'], inplace=True)
train_csv.dropna(how='all', subset=['segmento'], inplace=True)
train_csv.dropna(how='all', subset=['canal_entrada'], inplace=True)
train_csv.dropna(how='all', subset=['sexo'], inplace=True)
#New
# train_csv.dropna(how='all', subset=['indrel_1mes'], inplace=True)
# train_csv.dropna(how='all', subset=['ind_nomina_ult1'], inplace=True)
# train_csv.dropna(how='all', subset=['ind_nom_pens_ult1'], inplace=True)

'''test_csv.dropna(how='all', subset=['cod_prov'], inplace=True)
test_csv.dropna(how='all', subset=['segmento'], inplace=True)
test_csv.dropna(how='all', subset=['canal_entrada'], inplace=True)
test_csv.dropna(how='all', subset=['sexo'], inplace=True)'''
#New
# test_csv.dropna(how='all', subset=['indrel_1mes'], inplace=True)
# test_csv.dropna(how='all', subset=['ind_nomina_ult1'], inplace=True)
# test_csv.dropna(how='all', subset=['ind_nom_pens_ult1'], inplace=True)

train_csv['ind_empleado'] = label_encoder.fit_transform(train_csv['ind_empleado'])
train_csv['ind_empleado'].unique()

train_csv['pais_residencia'] = label_encoder.fit_transform(train_csv['pais_residencia'])
train_csv['pais_residencia'].unique()

train_csv['sexo'] = label_encoder.fit_transform(train_csv['sexo'])
train_csv['sexo'].unique()

train_csv['tiprel_1mes'] = label_encoder.fit_transform(train_csv['tiprel_1mes'])
train_csv['tiprel_1mes'].unique()

train_csv['indresi'] = label_encoder.fit_transform(train_csv['indresi'])
train_csv['indresi'].unique()

train_csv['indext'] = label_encoder.fit_transform(train_csv['indext'])
train_csv['indext'].unique()

train_csv['canal_entrada'] = label_encoder.fit_transform(train_csv['canal_entrada'])
train_csv['canal_entrada'].unique()

train_csv['indfall'] = label_encoder.fit_transform(train_csv['indfall'])
train_csv['indfall'].unique()

train_csv['nomprov'] = label_encoder.fit_transform(train_csv['nomprov'])
train_csv['nomprov'].unique()

train_csv['segmento'] = label_encoder.fit_transform(train_csv['segmento'])
train_csv['segmento'].unique()


train_csv['fecha_alta'] = pd.to_datetime(train_csv['fecha_alta'], format='%Y-%m-%d')
train_csv['fecha_alta'] = train_csv['fecha_alta'].view(np.int64) // 10 ** 9

test_csv['ind_empleado'] = label_encoder.fit_transform(test_csv['ind_empleado'])
test_csv['ind_empleado'].unique()

test_csv['pais_residencia'] = label_encoder.fit_transform(test_csv['pais_residencia'])
test_csv['pais_residencia'].unique()

test_csv['sexo'] = label_encoder.fit_transform(test_csv['sexo'])
test_csv['sexo'].unique()

test_csv['tiprel_1mes'] = label_encoder.fit_transform(test_csv['tiprel_1mes'])
test_csv['tiprel_1mes'].unique()

test_csv['indresi'] = label_encoder.fit_transform(test_csv['indresi'])
test_csv['indresi'].unique()

test_csv['indext'] = label_encoder.fit_transform(test_csv['indext'])
test_csv['indext'].unique()

test_csv['canal_entrada'] = label_encoder.fit_transform(test_csv['canal_entrada'])
test_csv['canal_entrada'].unique()

test_csv['indfall'] = label_encoder.fit_transform(test_csv['indfall'])
test_csv['indfall'].unique()

test_csv['nomprov'] = label_encoder.fit_transform(test_csv['nomprov'])
test_csv['nomprov'].unique()

test_csv['segmento'] = label_encoder.fit_transform(test_csv['segmento'])
test_csv['segmento'].unique()

test_csv['fecha_alta'] = pd.to_datetime(test_csv['fecha_alta'], format='%Y-%m-%d')
test_csv['fecha_alta'] = test_csv['fecha_alta'].view(np.int64) // 10 ** 9

train_y = train_csv.loc[:, 'ind_ahor_fin_ult1':'ind_recibo_ult1'].to_numpy()
train_x = train_csv.loc[:, 'ind_empleado':'segmento'].to_numpy()

test_x = test_csv.loc[:, 'ind_empleado':'segmento'].to_numpy()

test_x_with_code = test_csv.loc[:, 'ind_empleado':'segmento']
test_x_with_code = pd.concat([test_x_with_code, test_csv['ncodpers']], axis=1)
test_x_with_code = test_x_with_code.to_numpy()


'''print(train_csv.isnull().any())
print(train_y)
print(train_x)
print(train_y.shape)
print(train_x.shape)
print(test_x)
print(test_x.shape)

# Try several feature selection method

sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
new_x = sel.fit_transform(train_x)
print(sel.get_feature_names_out())

# Try find out each feature's contribution, and plot it below

selected_features = [] 
for t in range(train_y.shape[1]):
    selector = SelectKBest(f_classif, k='all')
    selector.fit(new_x, train_y[:,t])
    selected_features.append(list(selector.scores_))
selected_features = np.mean(selected_features, axis=0)
print(selected_features)'''

chosen_col_idx = [2, 3, 4, 10, 13, 18, 20]
chosen_col_idx2 = [2, 3, 4, 10, 13, 18, 20, 21]

print(train_x[:,chosen_col_idx])
np.save("train_x", train_x[:,chosen_col_idx])
np.save("train_y", train_y)
np.save("test_x", test_x[:,chosen_col_idx])
np.save("test_x_with_code", test_x_with_code[:,chosen_col_idx2])
