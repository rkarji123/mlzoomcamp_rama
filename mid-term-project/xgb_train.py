import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from preprocess import extract_base_features, add_target_features

# Load and clean dataset
df = pd.read_csv('cars.csv')

for col in ['Id', 'year']:
    if col in df.columns:
        del df[col]

df = df.drop_duplicates().reset_index(drop=True)
df = extract_base_features(df)

if 'Zest' in df['model_name'].values:
    df.loc[(df['model_name'] == 'Zest') & (df['price'] > 495000.0), 'price'] = 455000.0

# Identify categorical and numerical columns
categorical = df.select_dtypes(include=['object', 'string']).columns.tolist()
numerical = df.select_dtypes(include='number').columns.tolist()

numerical = [num for num in numerical if num != 'price']
categorical = [cat for cat in categorical if cat not in ('full_model_name', 'model_name')]
base_features = categorical + numerical

# Fit DictVectorizer
dv = DictVectorizer(sparse=False)
x_base = dv.fit_transform(df[base_features].to_dict(orient='records'))
y_series = df['price'].values

base_feature_names = list(dv.get_feature_names_out())
full_feature_names = base_feature_names + ['brand_median_price', 'model_median_price', 'depreciated_model_val']

xgb_params = {
    'eta': 0.3,
    'max_depth': 6,
    'min_child_weight': 1,
    'objective': 'reg:absoluteerror',
    'eval_metric': 'mae',
    'nthread': 8,
    'seed': 42,
    'verbosity': 1
}

# 1. Cross-Validation Evaluation
xgb_scores = []
kfld = KFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in kfld.split(df):
    df_train_fold = df.iloc[train_idx]
    df_val_fold = df.iloc[val_idx]
    
    g_med = df_train_fold['price'].median()
    b_med = df_train_fold.groupby('brand')['price'].median()
    m_med = df_train_fold.groupby('model_name_clean')['price'].median()
    
    x_tr = add_target_features(x_base[train_idx], df_train_fold, g_med, b_med, m_med)
    x_va = add_target_features(x_base[val_idx], df_val_fold, g_med, b_med, m_med)
    
    dtrain = xgb.DMatrix(x_tr, label=y_series[train_idx], feature_names=full_feature_names)
    dval = xgb.DMatrix(x_va, label=y_series[val_idx], feature_names=full_feature_names)
    
    model = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=200,
        verbose_eval=False
    )
    
    y_pred = model.predict(dval)
    xgb_scores.append(mean_absolute_error(y_series[val_idx], y_pred))

print(f"Mean CV MAE = {round(np.mean(xgb_scores), 2)}")

# 2. Train Final Model on Full Dataset & Save Artifacts
global_median = float(df['price'].median())
brand_medians = df.groupby('brand')['price'].median().to_dict()
model_medians = df.groupby('model_name_clean')['price'].median().to_dict()

x_full = add_target_features(x_base, df, global_median, brand_medians, model_medians)
dfull = xgb.DMatrix(x_full, label=y_series, feature_names=full_feature_names)

final_model = xgb.train(
    params=xgb_params,
    dtrain=dfull,
    num_boost_round=200,
    verbose_eval=False
)

artifacts = {
    'dv': dv,
    'model': final_model,
    'base_features': base_features,
    'global_median': global_median,
    'brand_medians': brand_medians,
    'model_medians': model_medians
}

with open('xgb_regressor.bin', 'wb') as f_out:
    pickle.dump(artifacts, f_out)

print("Final model and preprocessing artifacts saved to xgb_regressor.bin")
