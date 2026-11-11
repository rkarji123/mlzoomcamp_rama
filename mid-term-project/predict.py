import pickle
import pandas as pd
from flask import Flask, request, jsonify
from preprocess import extract_base_features, add_target_features

model_file = 'xgb_regressor.bin'

with open(model_file, 'rb') as f_in:
    artifacts = pickle.load(f_in)

dv = artifacts['dv']
model = artifacts['model']
base_features = artifacts['base_features']
global_median = artifacts['global_median']
brand_medians = artifacts['brand_medians']
model_medians = artifacts['model_medians']

app = Flask('car_price_prediction')

@app.route('/car_price_predict', methods=['POST'])
def car_price_predict():
    car_data = request.get_json()
    
    # Convert single JSON record to DataFrame and apply shared preprocessing
    df_input = pd.DataFrame([car_data])
    df_clean = extract_base_features(df_input)
    
    # Transform base features and append target-encoded features
    x_base = dv.transform(df_clean[base_features].to_dict(orient='records'))
    x_final = add_target_features(x_base, df_clean, global_median, brand_medians, model_medians)
    
    # Predict and extract scalar float
    y_pred = model.inplace_predict(x_final)
    
    result = {
        'predicted_car_price': float(y_pred[0])
    }
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
