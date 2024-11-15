import os
import csv
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report

def get_recomendations(recommendations, test_data_loc, filename):
  '''
  This function takes recomendation, test data location and file name 
  creates a csv file in the submission file format

  recommendations : list of recomendations
  test_data_loc : location of test data
  filename : Name of the submission file
  '''
  test_data_file = csv.DictReader(open(test_data_loc)) #Reading the test data
  test_ids = np.array([int(row['ncodpers']) for row in test_data_file]) #Get the user ids of customers in test data
  fields = ['ncodpers', 'added_products'] #Column names
  rows = np.vstack((test_ids, np.array(recommendations))).T #Creating an array of test ids and recomendations

  with open(filename, 'w') as csvfile: #Creating a csv file with given name
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields) #Writing column names tothe file
    csvwriter.writerows(rows) #Writing the rows to the file

  print(filename, 'is created')

test_x = np.load("test_x.npy", allow_pickle=True)
test_x_with_code = np.load("test_x_with_code.npy", allow_pickle=True)
train_x = np.load("train_x.npy", allow_pickle=True)
train_Y = np.load("train_y.npy", allow_pickle=True)

train_y = np.zeros(len(train_Y))

df = pd.DataFrame(test_x_with_code)
code = df[7]

table = pd.read_csv('product.csv')
table = table.drop('product_name', axis=1)
table = table['product_id'].to_dict()

for idx, data in enumerate(train_Y):
    train_y[idx] = np.argmax(data)

valid_size = 1000000
print(f"original train_x shape: {train_x.shape}")
print(f"original train_y shape: {train_y.shape}\n")

valid_x = train_x[:valid_size]  # 100,000 samples for validation
valid_y = train_y[:valid_size]  # 100,000 samples for validation
valid_Y = train_Y[:valid_size]

train_x = train_x[valid_size:]
train_y = train_y[valid_size:]
train_Y = train_Y[valid_size:]

print(f"final train_x shape: {train_x.shape}")
print(f"final train_y shape: {train_y.shape}")
print(f"final valid_x shape: {valid_x.shape}")
print(f"final valid_y shape: {valid_y.shape}")
print(f"final test_x shape: {test_x.shape}")

# Tree Base Algorithm

rf = RandomForestClassifier()

'''xgb_model = XGBClassifier(num_class=24,
                        random_state=0,
                        objective='multi:softmax',
                        learning_rate=0.1,
                        max_depth=3    
                        )

xgb_model.fit(train_x, train_y, verbose=1)
y_pred = xgb_model.predict(valid_x)
print(classification_report(valid_y, y_pred))'''

'''gbm_model = LGBMClassifier(learning_rate = 0.05, 
                            n_estimators = 100, 
                            random_state=0,
                            objective='multiclass',
                            num_class=24,
                            max_depth=3,
                            )

gbm_model.fit(train_x, train_y, verbose=1)
y_pred = gbm_model.predict(valid_x)
print(classification_report(valid_y, y_pred))'''

cat_model = CatBoostClassifier(iterations=200,
                           learning_rate=0.01,
                           depth=8,
                           loss_function='MultiLogloss',
                           random_seed=0)

cat_model.fit(train_x, train_Y, eval_set=(valid_x, valid_Y), verbose=1, plot=True)

y_pred = cat_model.predict(valid_x)
print(classification_report(valid_Y, y_pred))

y_pred = cat_model.predict(test_x)
pred_res = np.zeros(len(y_pred))

for idx, data in enumerate(y_pred):
    pred_res[idx] = np.argmax(data)

print(len(pred_res))
prediction = pd.DataFrame(pred_res)
print(prediction)
print(len(code))
print(code)
prediction = prediction[0].map(table)

result = pd.concat([code, prediction], axis=1)

print(result)
result.to_csv('predict.csv', index=False)


