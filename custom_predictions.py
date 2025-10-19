"""
City Insights 360 - Custom Prediction Tool
==========================================

Create your own predictions with specific parameters
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import json

class CustomPredictionTool:
    """Make custom predictions with your trained models"""
    
    def __init__(self):
        self.models_dir = Path("predictive_models/models")
        self.models = self._load_models()
    
    def _load_models(self):
        """Load all trained models"""
        models = {}
        
        if not self.models_dir.exists():
            print("❌ Models directory not found. Train models first!")
            return models
        
        # Load each model
        model_files = {
            'air_quality': 'air_quality_model.joblib',
            'traffic': 'traffic_model.joblib', 
            'digital_readiness': 'digital_readiness_model.joblib'
        }
        
        for model_name, filename in model_files.items():
            model_path = self.models_dir / filename
            scaler_path = self.models_dir / f"{model_name}_scaler.joblib"
            
            if model_path.exists():
                try:
                    models[model_name] = {
                        'model': joblib.load(model_path),
                        'scaler': joblib.load(scaler_path) if scaler_path.exists() else None
                    }
                    print(f"✅ Loaded {model_name} model")
                except Exception as e:
                    print(f"❌ Failed to load {model_name}: {e}")
        
        return models
    
    def predict_air_quality(self, hour=12, day_of_week=1, month=6, 
                           pm25=85, pm10=120, no2=45, so2=15, o3=60, co=2.5, city_code=1):
        """
        Predict air quality (AQI) for specific conditions
        
        Parameters:
        - hour: Hour of day (0-23)
        - day_of_week: Day of week (0=Monday, 6=Sunday) 
        - month: Month (1-12)
        - pm25: PM2.5 level (µg/m³)
        - pm10: PM10 level (µg/m³)
        - no2: NO2 level (µg/m³)
        - so2: SO2 level (µg/m³)
        - o3: O3 level (µg/m³)
        - co: CO level (mg/m³)
        - city_code: City encoding (0-10)
        """
        if 'air_quality' not in self.models:
            print("❌ Air quality model not available")
            return None
        
        # Prepare features
        features = [
            hour, day_of_week, month, 
            1 if day_of_week >= 5 else 0,  # is_weekend
            pm25, pm10, no2, so2, o3, co, city_code
        ]
        
        # Make prediction
        model_info = self.models['air_quality']
        feature_array = np.array([features])
        
        if model_info['scaler']:
            feature_array = model_info['scaler'].transform(feature_array)
        
        prediction = model_info['model'].predict(feature_array)[0]
        
        # Interpret result
        if prediction <= 50:
            level = "Good 😊"
        elif prediction <= 100:
            level = "Moderate 😐" 
        elif prediction <= 150:
            level = "Unhealthy for Sensitive 🟡"
        elif prediction <= 200:
            level = "Unhealthy 😷"
        elif prediction <= 300:
            level = "Very Unhealthy 🚨"
        else:
            level = "Hazardous ☠️"
        
        return {
            'predicted_aqi': round(prediction, 1),
            'air_quality_level': level,
            'input_conditions': {
                'time': f"{hour:02d}:00",
                'day': ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][day_of_week],
                'month': month,
                'pollutants': {
                    'PM2.5': pm25, 'PM10': pm10, 'NO2': no2,
                    'SO2': so2, 'O3': o3, 'CO': co
                }
            }
        }
    
    def predict_traffic_congestion(self, hour=8, day_of_week=1, weather_condition=1):
        """
        Predict traffic congestion for specific time
        
        Parameters:
        - hour: Hour of day (0-23)
        - day_of_week: Day of week (0=Monday, 6=Sunday)
        - weather_condition: Weather factor (0-2, 0=good, 2=poor)
        """
        if 'traffic' not in self.models:
            print("❌ Traffic model not available") 
            return None
        
        # Create feature vector (simplified for demo)
        features = [hour, day_of_week, weather_condition, 
                   1 if day_of_week >= 5 else 0]  # is_weekend
        
        model_info = self.models['traffic']
        
        try:
            # Pad with average values if needed (traffic model expects more features)
            while len(features) < 10:  # Assuming 10 features
                features.append(0.5)  # Default middle value
            
            feature_array = np.array([features[:10]])  # Take first 10 features
            prediction = model_info['model'].predict(feature_array)[0]
            
            # Interpret congestion level
            if prediction < 0.3:
                level = "Light Traffic 🟢"
            elif prediction < 0.6:
                level = "Moderate Traffic 🟡"
            else:
                level = "Heavy Traffic 🔴"
            
            return {
                'predicted_congestion': round(prediction, 3),
                'traffic_level': level,
                'input_conditions': {
                    'time': f"{hour:02d}:00",
                    'day': ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][day_of_week],
                    'weather': ['Good','Fair','Poor'][weather_condition]
                }
            }
        except Exception as e:
            print(f"⚠️ Traffic prediction error: {e}")
            return None
    
    def predict_digital_growth(self, city_name="Mumbai", current_year=2024, 
                              current_score=65, growth_rate=0.05):
        """
        Predict digital infrastructure growth
        
        Parameters:
        - city_name: Name of the city
        - current_year: Current year
        - current_score: Current digital readiness score (0-100)
        - growth_rate: Expected annual growth rate (0.05 = 5%)
        """
        if 'digital_readiness' not in self.models:
            print("❌ Digital readiness model not available")
            return None
        
        projections = []
        
        for year_offset in range(1, 4):  # Next 3 years
            future_year = current_year + year_offset
            projected_score = min(100, current_score * ((1 + growth_rate) ** year_offset))
            
            # Create feature vector for prediction
            features = [
                future_year, projected_score, projected_score * 0.8,  # Basic features
                projected_score * 0.9, projected_score * 0.7, 50, 75  # Additional features
            ]
            
            try:
                model_info = self.models['digital_readiness']
                feature_array = np.array([features[:model_info['model'].n_features_in_]])
                prediction = model_info['model'].predict(feature_array)[0]
                
                projections.append({
                    'year': future_year,
                    'predicted_score': round(prediction, 1),
                    'growth_assumption': f"{growth_rate*100:.1f}% annually"
                })
            except Exception as e:
                print(f"⚠️ Digital growth prediction error: {e}")
                break
        
        return {
            'city': city_name,
            'projections': projections,
            'current_baseline': {
                'year': current_year,
                'score': current_score
            }
        }

def main():
    """Interactive prediction examples"""
    print("🔮 City Insights 360 - Custom Prediction Tool")
    print("=" * 50)
    
    predictor = CustomPredictionTool()
    
    if not predictor.models:
        print("No models available. Please train models first!")
        return
    
    # Example 1: Air Quality Prediction
    print("\n🌬️ EXAMPLE: Air Quality Prediction")
    print("-" * 35)
    
    aqi_result = predictor.predict_air_quality(
        hour=14,  # 2 PM
        day_of_week=2,  # Wednesday 
        month=7,  # July
        pm25=95,  # High PM2.5
        pm10=140,  # High PM10
        no2=50,
        so2=20,
        o3=75,
        co=3.0,
        city_code=2
    )
    
    if aqi_result:
        print(f"Predicted AQI: {aqi_result['predicted_aqi']}")
        print(f"Level: {aqi_result['air_quality_level']}")
        print(f"Conditions: {aqi_result['input_conditions']['time']} on {aqi_result['input_conditions']['day']}")
    
    # Example 2: Traffic Prediction  
    print("\n🚦 EXAMPLE: Traffic Congestion Prediction")
    print("-" * 40)
    
    traffic_result = predictor.predict_traffic_congestion(
        hour=8,  # 8 AM rush hour
        day_of_week=1,  # Tuesday
        weather_condition=0  # Good weather
    )
    
    if traffic_result:
        print(f"Congestion Index: {traffic_result['predicted_congestion']}")
        print(f"Level: {traffic_result['traffic_level']}")
        print(f"Conditions: {traffic_result['input_conditions']['time']} - {traffic_result['input_conditions']['weather']} weather")
    
    # Example 3: Digital Growth Prediction
    print("\n📱 EXAMPLE: Digital Infrastructure Growth")
    print("-" * 40)
    
    digital_result = predictor.predict_digital_growth(
        city_name="Bangalore",
        current_year=2024,
        current_score=70,
        growth_rate=0.08  # 8% annual growth
    )
    
    if digital_result:
        print(f"City: {digital_result['city']}")
        print(f"Current Score: {digital_result['current_baseline']['score']}/100")
        print("Future Projections:")
        for proj in digital_result['projections']:
            print(f"  {proj['year']}: {proj['predicted_score']}/100")
    
    print("\n🎯 Create Your Own Predictions!")
    print("Modify the parameters in this script to test different scenarios.")
    print("Use the predictor.predict_* methods with your own values.")

if __name__ == "__main__":
    main()