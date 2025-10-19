"""
City Insights 360 - Predictive Analytics Models
==============================================

This module provides machine learning models for forecasting urban challenges
including air quality prediction, traffic congestion forecasting, and trend analysis.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import joblib
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class UrbanPredictiveModels:
    """Predictive models for urban analytics"""
    
    def __init__(self, integrated_data_dir: str):
        self.data_dir = Path(integrated_data_dir)
        self.models = {}
        self.scalers = {}
        self.model_performance = {}
        self.datasets = self._load_datasets()
        
    def _load_datasets(self) -> dict:
        """Load integrated datasets"""
        datasets = {}
        
        files_to_load = [
            'air_quality_integrated.csv',
            'traffic_mobility_integrated.csv',
            'digital_infrastructure_integrated.csv',
            'demographics_integrated.csv'
        ]
        
        for filename in files_to_load:
            file_path = self.data_dir / filename
            if file_path.exists():
                try:
                    key = filename.replace('_integrated.csv', '')
                    datasets[key] = pd.read_csv(file_path)
                    print(f"✅ Loaded {key}: {len(datasets[key])} records")
                except Exception as e:
                    print(f"❌ Error loading {filename}: {str(e)}")
        
        return datasets
    
    def build_air_quality_prediction_model(self):
        """Build air quality prediction model"""
        print("🌬️ Building air quality prediction model...")
        
        if 'air_quality' not in self.datasets:
            print("❌ No air quality data available for modeling")
            return
        
        df = self.datasets['air_quality'].copy()
        
        # Prepare features
        features = []
        target = 'AQI'
        
        if target not in df.columns:
            print(f"❌ Target column '{target}' not found")
            return
        
        # Add temporal features
        if 'datetime' in df.columns:
            try:
                df['datetime_parsed'] = pd.to_datetime(df['datetime'], errors='coerce')
                df['hour'] = df['datetime_parsed'].dt.hour
                df['day_of_week'] = df['datetime_parsed'].dt.dayofweek
                df['month'] = df['datetime_parsed'].dt.month
                df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
                features.extend(['hour', 'day_of_week', 'month', 'is_weekend'])
            except:
                print("⚠️ Could not parse datetime for temporal features")
        
        # Add pollutant features
        pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO']
        for pollutant in pollutants:
            if pollutant in df.columns:
                features.append(pollutant)
        
        # Add city encoding
        if 'standardized_city' in df.columns:
            le_city = LabelEncoder()
            df['city_encoded'] = le_city.fit_transform(df['standardized_city'].fillna('unknown'))
            features.append('city_encoded')
        
        # Prepare data
        if len(features) == 0:
            print("❌ No suitable features found for modeling")
            return
        
        # Clean data
        model_data = df[features + [target]].dropna()
        
        if len(model_data) < 100:
            print("❌ Insufficient data for modeling (need at least 100 records)")
            return
        
        X = model_data[features]
        y = model_data[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple models
        models_to_try = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression()
        }
        
        best_model = None
        best_score = float('-inf')
        
        for model_name, model in models_to_try.items():
            try:
                # Train model
                if model_name == 'LinearRegression':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Evaluate
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                print(f"  {model_name}: R²={r2:.3f}, MAE={mae:.2f}, RMSE={rmse:.2f}")
                
                if r2 > best_score:
                    best_score = r2
                    best_model = (model_name, model, scaler if model_name == 'LinearRegression' else None)
                
                # Store performance
                self.model_performance[f'air_quality_{model_name}'] = {
                    'r2_score': float(r2),
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'features_used': features
                }
                
            except Exception as e:
                print(f"  ❌ Error training {model_name}: {str(e)}")
        
        if best_model:
            model_name, model, scaler = best_model
            self.models['air_quality'] = {
                'model': model,
                'scaler': scaler,
                'features': features,
                'model_type': model_name,
                'performance': self.model_performance[f'air_quality_{model_name}']
            }
            print(f"✅ Best model: {model_name} (R²={best_score:.3f})")
    
    def build_traffic_prediction_model(self):
        """Build traffic congestion prediction model"""
        print("🚦 Building traffic prediction model...")
        
        if 'traffic_mobility' not in self.datasets:
            print("❌ No traffic data available for modeling")
            return
        
        df = self.datasets['traffic_mobility'].copy()
        
        features = []
        target = 'congestion_index'
        
        if target not in df.columns:
            target = 'Road_Occupancy_%'  # Fallback target
        
        if target not in df.columns:
            print(f"❌ No suitable target column found for traffic modeling")
            return
        
        # Add temporal features
        if 'hour' in df.columns:
            features.append('hour')
        if 'day_of_week' in df.columns:
            features.append('day_of_week')
        
        # Add traffic features
        traffic_features = ['Vehicle_Count', 'Traffic_Speed_kmh']
        for feature in traffic_features:
            if feature in df.columns and feature != target:
                features.append(feature)
        
        # Weather impact
        if 'weather_impact_factor' in df.columns:
            features.append('weather_impact_factor')
        elif 'Weather_Condition' in df.columns:
            le_weather = LabelEncoder()
            df['weather_encoded'] = le_weather.fit_transform(df['Weather_Condition'].fillna('Clear'))
            features.append('weather_encoded')
        
        if len(features) == 0:
            print("❌ No suitable features found for traffic modeling")
            return
        
        # Prepare data
        model_data = df[features + [target]].dropna()
        
        if len(model_data) < 100:
            print("❌ Insufficient data for modeling")
            return
        
        X = model_data[features]
        y = model_data[target]
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Use Random Forest as it handles mixed data types well
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        self.models['traffic'] = {
            'model': model,
            'features': features,
            'model_type': 'RandomForest',
            'target': target,
            'performance': {
                'r2_score': float(r2),
                'mae': float(mae)
            }
        }
        
        print(f"✅ Traffic model trained: R²={r2:.3f}, MAE={mae:.2f}")
    
    def build_digital_readiness_model(self):
        """Build digital infrastructure development model"""
        print("💻 Building digital readiness prediction model...")
        
        if 'digital_infrastructure' not in self.datasets:
            print("❌ No digital infrastructure data available")
            return
        
        df = self.datasets['digital_infrastructure'].copy()
        
        # Check if we have multiple years for trend analysis
        if 'Year' not in df.columns or df['Year'].nunique() < 2:
            print("❌ Need multiple years of data for digital readiness modeling")
            return
        
        features = []
        target = 'digital_readiness_score'
        
        if target not in df.columns:
            print(f"❌ Target column '{target}' not found")
            return
        
        # Infrastructure features
        infra_features = [
            'Household Internet Access (%)',
            'Fixed Broadband Subscriptions (%)',
            'Wireless Broadband Coverage 4G (%)',
            'Smart Electricity Meters (%)',
            'e-Government (%)'
        ]
        
        for feature in infra_features:
            if feature in df.columns:
                features.append(feature)
        
        # Add year as a feature for trend analysis
        if 'Year' in df.columns:
            features.append('Year')
        
        # Add city encoding
        if 'standardized_city' in df.columns:
            le_city = LabelEncoder()
            df['city_encoded'] = le_city.fit_transform(df['standardized_city'].fillna('unknown'))
            features.append('city_encoded')
        
        if len(features) == 0:
            print("❌ No suitable features found")
            return
        
        # Prepare data
        model_data = df[features + [target]].dropna()
        
        if len(model_data) < 30:
            print("❌ Insufficient data for modeling")
            return
        
        X = model_data[features]
        y = model_data[target]
        
        # Train model
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        # Cross-validation for small dataset
        cv_scores = cross_val_score(model, X, y, cv=3, scoring='r2')
        
        self.models['digital_readiness'] = {
            'model': model,
            'features': features,
            'model_type': 'RandomForest',
            'performance': {
                'cv_r2_mean': float(cv_scores.mean()),
                'cv_r2_std': float(cv_scores.std())
            }
        }
        
        print(f"✅ Digital readiness model: CV R²={cv_scores.mean():.3f}±{cv_scores.std():.3f}")
    
    def generate_predictions(self, forecast_horizon_days: int = 30):
        """Generate predictions for various metrics"""
        print(f"🔮 Generating {forecast_horizon_days}-day forecasts...")
        
        predictions = {}
        
        # Air quality predictions
        if 'air_quality' in self.models:
            predictions['air_quality'] = self._predict_air_quality(forecast_horizon_days)
        
        # Traffic predictions
        if 'traffic' in self.models:
            predictions['traffic'] = self._predict_traffic_patterns()
        
        # Digital readiness projections
        if 'digital_readiness' in self.models:
            predictions['digital_readiness'] = self._predict_digital_growth()
        
        return predictions
    
    def _predict_air_quality(self, days: int):
        """Generate air quality predictions"""
        model_info = self.models['air_quality']
        model = model_info['model']
        
        # Create future scenarios
        scenarios = []
        
        # Current conditions scenario
        if 'air_quality' in self.datasets:
            recent_data = self.datasets['air_quality'].tail(100)
            
            # Calculate average conditions for features
            avg_conditions = {}
            for feature in model_info['features']:
                if feature in recent_data.columns:
                    avg_conditions[feature] = recent_data[feature].mean()
                else:
                    avg_conditions[feature] = 0
            
            scenarios.append({
                'scenario': 'current_conditions',
                'description': 'Maintaining current pollution levels',
                'predicted_aqi': self._make_prediction(model, avg_conditions, model_info)
            })
            
            # Improved conditions (20% reduction in pollutants)
            improved_conditions = avg_conditions.copy()
            pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO']
            for p in pollutants:
                if p in improved_conditions:
                    improved_conditions[p] *= 0.8
            
            scenarios.append({
                'scenario': 'improved_conditions',
                'description': '20% reduction in major pollutants',
                'predicted_aqi': self._make_prediction(model, improved_conditions, model_info)
            })
        
        return {
            'forecast_horizon_days': days,
            'scenarios': scenarios,
            'model_performance': model_info['performance']
        }
    
    def _predict_traffic_patterns(self):
        """Predict traffic congestion patterns"""
        model_info = self.models['traffic']
        
        # Predict for different times of day
        hourly_predictions = []
        
        if 'traffic_mobility' in self.datasets:
            df = self.datasets['traffic_mobility']
            
            for hour in range(24):
                # Create feature vector for this hour
                features = {}
                features['hour'] = hour
                features['day_of_week'] = 1  # Assume weekday
                
                # Use average values for other features
                for feature in model_info['features']:
                    if feature not in features and feature in df.columns:
                        features[feature] = df[feature].mean()
                    elif feature not in features:
                        features[feature] = 0
                
                prediction = self._make_prediction(model_info['model'], features, model_info)
                
                hourly_predictions.append({
                    'hour': hour,
                    f"predicted_{model_info['target']}": prediction
                })
        
        return {
            'hourly_patterns': hourly_predictions,
            'model_performance': model_info['performance']
        }
    
    def _predict_digital_growth(self):
        """Predict digital infrastructure growth"""
        model_info = self.models['digital_readiness']
        
        projections = []
        
        if 'digital_infrastructure' in self.datasets:
            df = self.datasets['digital_infrastructure']
            latest_year = df['Year'].max()
            
            # Project for next 3 years
            for future_year in range(latest_year + 1, latest_year + 4):
                year_projections = []
                
                # Get unique cities
                cities = df['standardized_city'].unique()[:10]  # Top 10 cities
                
                for city in cities:
                    city_data = df[df['standardized_city'] == city].tail(1)
                    
                    if not city_data.empty:
                        features = {}
                        features['Year'] = future_year
                        
                        # Use latest values for other features, with growth assumption
                        for feature in model_info['features']:
                            if feature == 'Year':
                                continue
                            elif feature in city_data.columns:
                                # Assume 5% annual growth for infrastructure metrics
                                current_value = city_data[feature].iloc[0]
                                years_ahead = future_year - latest_year
                                features[feature] = min(100, current_value * (1.05 ** years_ahead))
                            else:
                                features[feature] = 50  # Default value
                        
                        prediction = self._make_prediction(model_info['model'], features, model_info)
                        
                        year_projections.append({
                            'city': city,
                            'predicted_score': prediction
                        })
                
                projections.append({
                    'year': future_year,
                    'city_projections': year_projections
                })
        
        return {
            'yearly_projections': projections,
            'model_performance': model_info['performance']
        }
    
    def _make_prediction(self, model, features_dict, model_info):
        """Make a single prediction"""
        try:
            # Create feature vector in correct order
            feature_vector = []
            for feature in model_info['features']:
                feature_vector.append(features_dict.get(feature, 0))
            
            feature_array = np.array([feature_vector])
            
            # Scale if needed
            if model_info.get('scaler'):
                feature_array = model_info['scaler'].transform(feature_array)
            
            prediction = model.predict(feature_array)[0]
            return float(prediction)
        
        except Exception as e:
            print(f"⚠️ Prediction error: {str(e)}")
            return None
    
    def export_models(self, output_dir: str):
        """Export trained models and predictions"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save models
        models_dir = output_path / "models"
        models_dir.mkdir(exist_ok=True)
        
        for model_name, model_info in self.models.items():
            model_file = models_dir / f"{model_name}_model.joblib"
            joblib.dump(model_info['model'], model_file)
            
            if model_info.get('scaler'):
                scaler_file = models_dir / f"{model_name}_scaler.joblib"
                joblib.dump(model_info['scaler'], scaler_file)
        
        # Generate and save predictions
        predictions = self.generate_predictions()
        
        with open(output_path / "predictions.json", 'w') as f:
            json.dump(predictions, f, indent=2, default=str)
        
        # Save model summary
        summary = {
            'models_trained': list(self.models.keys()),
            'model_performance': self.model_performance,
            'training_date': datetime.now().isoformat(),
            'total_models': len(self.models)
        }
        
        with open(output_path / "model_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"📊 Models and predictions exported to {output_path}")

if __name__ == "__main__":
    # Initialize predictive models
    data_dir = r"C:\Users\91892\OneDrive\Desktop\City Insights 360\integrated_data"
    models = UrbanPredictiveModels(data_dir)
    
    print("🤖 Building predictive models...")
    
    # Build models
    models.build_air_quality_prediction_model()
    models.build_traffic_prediction_model()
    models.build_digital_readiness_model()
    
    # Export results
    output_dir = Path(r"C:\Users\91892\OneDrive\Desktop\City Insights 360") / "predictive_models"
    models.export_models(str(output_dir))
    
    print("\n🎯 Predictive modeling completed successfully!")
    print(f"📈 Built {len(models.models)} predictive models")
    print(f"📁 Results saved to: {output_dir}")