import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv("C:\PepperPepper\pepperProject.csv", encoding='unicode_escape', engine='python')
df_data = df.copy()

new_sheet = pd.DataFrame(df_data,
                         columns=['OMOP_COMP_CODE', 'CTRL_JOB', 'STAGE_CODE', 'MARKET_TYPE', 'POTENTIAL_REV_AMT',
                                  'TOTAL_HOURS'])
new_sheet = new_sheet[~new_sheet['MARKET_TYPE'].isin(['Select Market', 'Self Performed Work', 'Self Performed Direct'])]
new_sheet = new_sheet[new_sheet['POTENTIAL_REV_AMT'] > 0]
location_100 = new_sheet[new_sheet.OMOP_COMP_CODE == 100]
location_100 = location_100.drop('OMOP_COMP_CODE', 1)
JobHour_by_StageMarket = location_100.groupby(['CTRL_JOB', 'STAGE_CODE', 'MARKET_TYPE'])[
    'POTENTIAL_REV_AMT', 'TOTAL_HOURS'].sum().reset_index()
revAmt_Hour0 = JobHour_by_StageMarket.iloc[:, -2:].abs()

z_scores = stats.zscore(revAmt_Hour0)
abs_z_scores = np.abs(z_scores)
revAmt_Hour1 = revAmt_Hour0[(abs_z_scores < 3).all(axis=1)]

rest = JobHour_by_StageMarket.iloc[:, :-2]
JobHour_by_StageMarket = rest.join(revAmt_Hour1, how='outer')
JobHour_by_StageMarket = JobHour_by_StageMarket.dropna()

standardscaler = preprocessing.StandardScaler()
numer_feature = standardscaler.fit_transform(JobHour_by_StageMarket["POTENTIAL_REV_AMT"].values.reshape(-1, 1))
numer_feature = pd.DataFrame(numer_feature, columns=["POTENTIAL_REV_AMT"])

ohe = preprocessing.OneHotEncoder(categories='auto')
feature_arr = ohe.fit_transform(JobHour_by_StageMarket[['STAGE_CODE', 'MARKET_TYPE']]).toarray()

feature_labels = ohe.get_feature_names()
feature_labels = np.array(feature_labels, dtype=object).ravel()
features = pd.DataFrame(feature_arr, columns=feature_labels)
predictors = np.concatenate([features, numer_feature], axis=1)
target = JobHour_by_StageMarket['TOTAL_HOURS']

X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.20, random_state=37)
reg = RandomForestRegressor(n_estimators=200, min_samples_split=5, min_samples_leaf=4, max_features='auto',
                            max_depth=80, bootstrap='True')
reg.fit(X_train, y_train)

joblib.dump(reg, 'model.pkl')
print("Model dumped!")

joblib.dump(ohe, 'model_ohe.pkl')
print("Models ohe dumped!")

joblib.dump(standardscaler, 'model_standardscaler.pkl')
print("Models standardscaler dumped!")

# JavaScript fetch API <- HTML