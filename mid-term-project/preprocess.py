import re
import pandas as pd
import numpy as np

def clean_model_name(row):
    text = re.sub(rf'^{re.escape(str(row["brand"]))}\s+', '', str(row['full_model_name']), flags=re.IGNORECASE)
    words = text.split()
    multi_word = {'7', '5', '3', '6', 'range', 'swift', 'grand', 'wagon', 'alto', 'innova', 'corolla', 'land', 'continental'}
    if len(words) >= 2 and words[0].lower() in multi_word:
        return f"{words[0]} {words[1]}"
    return words[0] if words else str(row.get('model_name', 'UNKNOWN'))

def extract_base_features(df):
    df = df.copy()
    patt = r'\b(at|mt|automatic|manual|multitronic|tiptronic|amt|cvt|dct|dsg|vtvt|vct)\b'
    
    df['transmission'] = (
        df['full_model_name']
        .str.extract(patt, flags=re.IGNORECASE, expand=False)
        .str.upper()
        .fillna('UNKNOWN')
    )
    
    engine_litres = df['full_model_name'].str.extract(r'(\d\.\d)', expand=False)
    df['engine_litres'] = (engine_litres + 'L').fillna('UNKNOWN')
    df['model_name_clean'] = df.apply(clean_model_name, axis=1)
    df['fuel_type'] = df['fuel_type'].astype(str).str.replace(r'\s*\+\s*\d+', '', regex=True)
    
    return df

def add_target_features(X_matrix, df_meta, global_median, brand_medians, model_medians):
    # Keep brand_med as a Pandas Series (do NOT add .values here)
    brand_med = df_meta['brand'].map(brand_medians).fillna(global_median)

    # Now .fillna(brand_med) works because brand_med is a Series
    model_med = df_meta['model_name_clean'].map(model_medians).fillna(brand_med)

    # Convert to NumPy arrays at the end
    brand_med_vals = brand_med.values
    model_med_vals = model_med.values
    depreciated_val = model_med_vals / (df_meta['car_age'].values + 1)

    extra_cols = np.column_stack([brand_med_vals, model_med_vals, depreciated_val])
    return np.hstack([X_matrix, extra_cols])
