import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

model_path = "model.pkl"
model = None

# Cache metrics and parameters
model_metrics = {
    'train': {'r2': None, 'mse': None, 'mae': None},
    'test': {'r2': None, 'mse': None, 'mae': None}
}
model_params = {
    'coefficients': None,
    'intercept': None
}

# Hardcoded cached fallback coefficients & metrics
CACHED_COEF = [
    0.44867490966571805, 
    0.0097242575179049, 
    -0.12332334282795997, 
    0.7831449067929749, 
    -2.0296205801018097e-06, 
    -0.0035263184871341014, 
    -0.41979248658836005, 
    -0.43370806496398745
]
CACHED_INTERCEPT = -37.02327770606416
CACHED_METRICS = {
    'train': {'r2': 0.6125511913966952, 'mse': 0.5179331255246699, 'mae': 0.5286283596581923},
    'test': {'r2': 0.575787706032451, 'mse': 0.5558915986952442, 'mae': 0.5332001304956555}
}

def load_data_and_compute_metrics():
    global model, model_metrics, model_params
    
    # 1. Load model pickle
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        except Exception as e:
            print(f"Error loading model pickle: {e}")
            model = None

    # Load coefficients/intercept from model if possible, otherwise use cached fallback
    if model is not None and hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
        model_params['coefficients'] = model.coef_.tolist()
        model_params['intercept'] = float(model.intercept_)
    else:
        model_params['coefficients'] = CACHED_COEF
        model_params['intercept'] = CACHED_INTERCEPT

    dataset_loaded = False
    # 2. Try fetching dataset to calculate metrics
    try:
        from sklearn.datasets import fetch_california_housing
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        
        housing = fetch_california_housing(as_frame=True)
        df = housing.frame
        X = df.drop("MedHouseVal", axis=1)
        y = df["MedHouseVal"]
        dataset_loaded = True
    except Exception as e1:
        print(f"Failed to fetch sklearn dataset: {e1}. Trying local fallback.")
        try:
            from _california_housing import fetch_california_housing as local_fetch
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            
            housing = local_fetch(as_frame=True)
            df = housing.frame
            X = df.drop("MedHouseVal", axis=1)
            y = df["MedHouseVal"]
            dataset_loaded = True
        except Exception as e2:
            print(f"Local fallback failed: {e2}. Falling back to cached metrics.")

    if dataset_loaded and model is not None:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            # Predict
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Metrics
            model_metrics['train']['r2'] = float(r2_score(y_train, y_train_pred))
            model_metrics['train']['mse'] = float(mean_squared_error(y_train, y_train_pred))
            model_metrics['train']['mae'] = float(mean_absolute_error(y_train, y_train_pred))
            
            model_metrics['test']['r2'] = float(r2_score(y_test, y_test_pred))
            model_metrics['test']['mse'] = float(mean_squared_error(y_test, y_test_pred))
            model_metrics['test']['mae'] = float(mean_absolute_error(y_test, y_test_pred))
            model_metrics['source'] = 'calculated_on_startup'
        except Exception as e_calc:
            print(f"Error calculating metrics: {e_calc}. Using cached fallback values.")
            model_metrics.update(CACHED_METRICS)
            model_metrics['source'] = 'cached_fallback_on_calculation_error'
    else:
        # Fallback to cached metrics
        model_metrics.update(CACHED_METRICS)
        model_metrics['source'] = 'cached_fallback_on_load_error'

# Run startup loading
load_data_and_compute_metrics()

def calculate_confidence(features):
    med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, lat, lon = features
    penalty = 0.0
    
    # MedInc: typical 0.5 to 15.0
    if med_inc < 0.5:
        penalty += (0.5 - med_inc) * 0.1
    elif med_inc > 15.0:
        penalty += (med_inc - 15.0) * 0.05
        
    # HouseAge: typical 1 to 52
    if house_age < 1:
        penalty += (1 - house_age) * 0.05
    elif house_age > 52:
        penalty += (house_age - 52) * 0.02
        
    # AveRooms: typical 3 to 8
    if ave_rooms < 3:
        penalty += (3 - ave_rooms) * 0.1
    elif ave_rooms > 8:
        penalty += (ave_rooms - 8) * 0.08
        
    # AveBedrms: typical 0.8 to 1.5
    if ave_bedrms < 0.8:
        penalty += (0.8 - ave_bedrms) * 0.2
    elif ave_bedrms > 1.5:
        penalty += (ave_bedrms - 1.5) * 0.2
        
    # Population: typical 50 to 10000
    if population < 50:
        penalty += (50 - population) * 0.002
    elif population > 10000:
        penalty += (population - 10000) * 0.00002
        
    # AveOccup: typical 1.5 to 6
    if ave_occup < 1.5:
        penalty += (1.5 - ave_occup) * 0.1
    elif ave_occup > 6:
        penalty += (ave_occup - 6) * 0.1
        
    # Latitude & Longitude bounds
    if not (32.5 <= lat <= 42.0):
        penalty += 0.25
    if not (-124.5 <= lon <= -114.0):
        penalty += 0.25
        
    confidence = max(0.5, 1.0 - penalty)
    return float(round(confidence, 2))

def calculate_rental_yield(predicted_price, med_inc, ave_occup):
    base_yield = 0.08
    price_penalty = min(0.045, (predicted_price / 1_000_000.0) * 0.04)
    occup_bonus = min(0.015, (ave_occup / 6.0) * 0.015)
    inc_modifier = -0.01 if med_inc > 8.0 else 0.0
    
    yield_val = base_yield - price_penalty + occup_bonus + inc_modifier
    yield_val = max(0.03, min(0.12, yield_val))
    return float(round(yield_val * 100, 2))

def calculate_appreciation(med_inc, house_age, lat, lon):
    base_appr = 0.045
    inc_bonus = min(0.03, (med_inc / 15.0) * 0.03)
    age_bonus = 0.015 * (1.0 - min(1.0, house_age / 50.0))
    
    is_prime_la = (33.5 <= lat <= 34.5) and (-119.0 <= lon <= -117.5)
    is_prime_sf = (37.2 <= lat <= 38.0) and (-123.0 <= lon <= -121.8)
    location_bonus = 0.015 if (is_prime_la or is_prime_sf) else 0.0
    
    appr = base_appr + inc_bonus + age_bonus + location_bonus
    appr = max(0.01, min(0.10, appr))
    return float(round(appr * 100, 2))

def calculate_investment_rating(total_return):
    if total_return >= 13.0:
        return "AAA"
    elif total_return >= 11.0:
        return "AA"
    elif total_return >= 9.0:
        return "A"
    elif total_return >= 7.0:
        return "B"
    elif total_return >= 5.0:
        return "C"
    else:
        return "D"

def calculate_risk_score(med_inc, house_age, ave_occup, population):
    risk = 5.0
    risk -= min(2.5, (med_inc / 15.0) * 2.5)
    risk += min(2.0, (house_age / 52.0) * 2.0)
    
    if ave_occup > 4.0:
        risk += min(2.0, ((ave_occup - 4.0) / 4.0) * 2.0)
        
    if population > 5000:
        risk += min(1.5, ((population - 5000) / 20000) * 1.5)
        
    if med_inc < 2.0:
        risk += 1.5
        
    risk = max(1, min(10, round(risk)))
    return int(risk)

def calculate_growth_score(med_inc, house_age, appreciation):
    score = 50.0
    score += min(25.0, (med_inc / 10.0) * 25.0)
    score += 10.0 * (1.0 - min(1.0, house_age / 52.0))
    score += min(15.0, ((appreciation - 2.0) / 8.0) * 15.0)
    
    score = max(1, min(100, round(score)))
    return int(score)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/metrics', methods=['GET'])
def get_metrics():
    return jsonify({
        'metrics': model_metrics,
        'parameters': model_params
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model pickle file not found'}), 500
        
    try:
        data = request.get_json()
        features = [
            float(data['MedInc']),
            float(data['HouseAge']),
            float(data['AveRooms']),
            float(data['AveBedrms']),
            float(data['Population']),
            float(data['AveOccup']),
            float(data['Latitude']),
            float(data['Longitude'])
        ]
        
        # Predict using pandas DataFrame to avoid warnings about missing feature names
        import pandas as pd
        feature_names = [
            "MedInc", "HouseAge", "AveRooms", "AveBedrms", 
            "Population", "AveOccup", "Latitude", "Longitude"
        ]
        input_df = pd.DataFrame([features], columns=feature_names)
        prediction = model.predict(input_df)[0]
        
        # Ensure value is positive (in dollars)
        actual_dollars = max(0.0, prediction * 100000.0)
        
        # Derive metadata
        confidence = calculate_confidence(features)
        rental_yield = calculate_rental_yield(actual_dollars, features[0], features[5])
        appreciation = calculate_appreciation(features[0], features[1], features[6], features[7])
        rating = calculate_investment_rating(rental_yield + appreciation)
        risk = calculate_risk_score(features[0], features[1], features[5], features[4])
        growth_score = calculate_growth_score(features[0], features[1], appreciation)
        
        return jsonify({
            'predicted_price': actual_dollars,
            'metadata': {
                'confidence': confidence,
                'rental_yield': rental_yield,
                'appreciation': appreciation,
                'rating': rating,
                'risk': risk,
                'growth_score': growth_score
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Run on local port 8000
    app.run(host='127.0.0.1', port=8000, debug=True)
