"""
City Insights 360 - Predictive Models Usage Guide
=================================================

This script demonstrates how to use these trained predictive models
for real-world urban planning and decision-making.
"""

import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class PredictiveInsightsManager:
    """Easy-to-use interface for your trained urban analytics models"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.models_dir = self.base_dir / "predictive_models"
        self.predictions_file = self.models_dir / "predictions.json"
        self.models_loaded = {}
        self.predictions_cache = None
        
        self._load_predictions()
        self._load_trained_models()
    
    def _load_predictions(self):
        """Load pre-generated predictions"""
        if self.predictions_file.exists():
            with open(self.predictions_file, 'r') as f:
                self.predictions_cache = json.load(f)
            print("✅ Loaded existing predictions")
        else:
            print("❌ No predictions file found. Run predictive_models.py first.")
    
    def _load_trained_models(self):
        """Load trained model files"""
        models_path = self.models_dir / "models"
        if not models_path.exists():
            print("❌ Models directory not found")
            return
        
        # Load available models
        model_files = list(models_path.glob("*_model.joblib"))
        for model_file in model_files:
            model_name = model_file.stem.replace('_model', '')
            try:
                self.models_loaded[model_name] = {
                    'model': joblib.load(model_file),
                    'loaded_at': datetime.now()
                }
                
                # Load scaler if exists
                scaler_file = models_path / f"{model_name}_scaler.joblib"
                if scaler_file.exists():
                    self.models_loaded[model_name]['scaler'] = joblib.load(scaler_file)
                
                print(f"✅ Loaded {model_name} model")
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {str(e)}")
    
    def show_air_quality_insights(self):
        """Display air quality predictions and insights"""
        print("\n🌬️  AIR QUALITY PREDICTIONS")
        print("=" * 50)
        
        if 'air_quality' not in self.predictions_cache:
            print("❌ No air quality predictions available")
            return
        
        aq_data = self.predictions_cache['air_quality']
        
        print("📊 30-Day Air Quality Forecast:")
        for scenario in aq_data['scenarios']:
            print(f"  • {scenario['description']}")
            print(f"    → Predicted AQI: {scenario['predicted_aqi']:.1f}")
            
            # Interpret AQI levels
            aqi_value = scenario['predicted_aqi']
            if aqi_value <= 50:
                level = "Good 😊"
            elif aqi_value <= 100:
                level = "Moderate 😐"
            elif aqi_value <= 150:
                level = "Unhealthy for Sensitive Groups ⚠️"
            elif aqi_value <= 200:
                level = "Unhealthy 😷"
            elif aqi_value <= 300:
                level = "Very Unhealthy 🚨"
            else:
                level = "Hazardous ☠️"
            
            print(f"    → Air Quality Level: {level}")
            print()
        
        print("🎯 Key Insights:")
        improvement_scenario = next(s for s in aq_data['scenarios'] if s['scenario'] == 'improved_conditions')
        current_scenario = next(s for s in aq_data['scenarios'] if s['scenario'] == 'current_conditions')
        
        improvement_potential = current_scenario['predicted_aqi'] - improvement_scenario['predicted_aqi']
        
        if improvement_potential > 0:
            print(f"  • Pollution reduction policies could improve AQI by {improvement_potential:.1f} points")
        else:
            print("  • Current model suggests minimal improvement from pollution reduction")
        
        print(f"  • Model uses {len(aq_data['model_performance']['features_used'])} features including time, pollutants, and location")
        print(f"  • Model accuracy (R²): {aq_data['model_performance']['r2_score']:.3f}")
    
    def show_traffic_insights(self):
        """Display traffic congestion predictions"""
        print("\n🚦 TRAFFIC CONGESTION PATTERNS")
        print("=" * 50)
        
        if 'traffic' not in self.predictions_cache:
            print("❌ No traffic predictions available")
            return
        
        traffic_data = self.predictions_cache['traffic']
        hourly_patterns = traffic_data['hourly_patterns']
        
        # Find peak and low traffic hours
        congestion_by_hour = [(h['hour'], h['predicted_congestion_index']) for h in hourly_patterns]
        congestion_by_hour.sort(key=lambda x: x[1], reverse=True)
        
        print("📈 24-Hour Traffic Forecast:")
        print("  🔴 Peak Congestion Hours:")
        for i, (hour, congestion) in enumerate(congestion_by_hour[:3]):
            time_str = f"{hour:02d}:00"
            print(f"    {i+1}. {time_str} - Congestion Index: {congestion:.3f}")
        
        print("\n  🟢 Low Traffic Hours:")
        for i, (hour, congestion) in enumerate(congestion_by_hour[-3:]):
            time_str = f"{hour:02d}:00"
            print(f"    {i+1}. {time_str} - Congestion Index: {congestion:.3f}")
        
        # Rush hour analysis
        morning_rush = [h for h in hourly_patterns if 7 <= h['hour'] <= 9]
        evening_rush = [h for h in hourly_patterns if 17 <= h['hour'] <= 19]
        
        morning_avg = np.mean([h['predicted_congestion_index'] for h in morning_rush])
        evening_avg = np.mean([h['predicted_congestion_index'] for h in evening_rush])
        
        print("\n🎯 Rush Hour Analysis:")
        print(f"  • Morning Rush (7-9 AM): {morning_avg:.3f} average congestion")
        print(f"  • Evening Rush (5-7 PM): {evening_avg:.3f} average congestion")
        
        if evening_avg > morning_avg:
            print("  • Evening commute is typically more congested")
        else:
            print("  • Morning commute shows higher congestion")
        
        print(f"  • Model accuracy (R²): {traffic_data['model_performance']['r2_score']:.3f}")
    
    def show_digital_growth_insights(self):
        """Display digital infrastructure growth predictions"""
        print("\n📱 DIGITAL INFRASTRUCTURE GROWTH")
        print("=" * 50)
        
        if 'digital_readiness' not in self.predictions_cache:
            print("❌ No digital readiness predictions available")
            return
        
        digital_data = self.predictions_cache['digital_readiness']
        projections = digital_data['yearly_projections']
        
        print("🚀 3-Year Digital Infrastructure Projections:")
        
        # Show top performing cities
        latest_year = projections[-1]
        top_cities = sorted(latest_year['city_projections'], 
                          key=lambda x: x['predicted_score'], reverse=True)[:5]
        
        print(f"\n🏆 Top Digital Cities by {latest_year['year']}:")
        for i, city in enumerate(top_cities, 1):
            print(f"  {i}. {city['city']}: {city['predicted_score']:.1f}/100")
        
        # Growth analysis
        print("\n📈 Growth Trajectory Analysis:")
        
        # Compare first and last year for a sample city
        sample_city = "Delhi"  # Use a major city
        first_year_data = next((proj for proj in projections[0]['city_projections'] 
                              if proj['city'] == sample_city), None)
        last_year_data = next((proj for proj in projections[-1]['city_projections'] 
                             if proj['city'] == sample_city), None)
        
        if first_year_data and last_year_data:
            growth = last_year_data['predicted_score'] - first_year_data['predicted_score']
            years = projections[-1]['year'] - projections[0]['year']
            annual_growth = growth / years
            
            print(f"  • {sample_city} projected growth: {growth:.1f} points over {years} years")
            print(f"  • Average annual growth rate: {annual_growth:.1f} points/year")
        
        print(f"\n  • Model performance (CV R²): {digital_data['model_performance']['cv_r2_mean']:.3f}")
        print(f"  • Prediction confidence: ±{digital_data['model_performance']['cv_r2_std']:.3f}")
    
    def generate_planning_recommendations(self):
        """Generate actionable urban planning recommendations"""
        print("\n🎯 URBAN PLANNING RECOMMENDATIONS")
        print("=" * 50)
        
        recommendations = []
        
        # Air Quality Recommendations
        if 'air_quality' in self.predictions_cache:
            aq_data = self.predictions_cache['air_quality']
            current_aqi = next(s['predicted_aqi'] for s in aq_data['scenarios'] 
                             if s['scenario'] == 'current_conditions')
            
            if current_aqi > 150:
                recommendations.append({
                    'category': 'Air Quality',
                    'priority': 'HIGH',
                    'action': 'Implement emergency pollution control measures',
                    'details': f'Current AQI forecast: {current_aqi:.0f} (Unhealthy level)'
                })
            elif current_aqi > 100:
                recommendations.append({
                    'category': 'Air Quality',
                    'priority': 'MEDIUM',
                    'action': 'Strengthen air quality monitoring and green initiatives',
                    'details': f'Current AQI forecast: {current_aqi:.0f} (Moderate level)'
                })
        
        # Traffic Recommendations
        if 'traffic' in self.predictions_cache:
            traffic_data = self.predictions_cache['traffic']
            hourly = traffic_data['hourly_patterns']
            
            peak_hours = sorted(hourly, key=lambda x: x['predicted_congestion_index'], reverse=True)[:3]
            peak_congestion = peak_hours[0]['predicted_congestion_index']
            
            if peak_congestion > 0.6:
                recommendations.append({
                    'category': 'Traffic Management',
                    'priority': 'HIGH',
                    'action': 'Implement dynamic traffic management during peak hours',
                    'details': f'Peak congestion at {peak_hours[0]["hour"]:02d}:00 - Index: {peak_congestion:.2f}'
                })
            else:
                recommendations.append({
                    'category': 'Traffic Management',
                    'priority': 'LOW',
                    'action': 'Monitor traffic patterns and optimize signal timing',
                    'details': f'Current traffic levels manageable (Peak: {peak_congestion:.2f})'
                })
        
        # Digital Infrastructure Recommendations
        if 'digital_readiness' in self.predictions_cache:
            digital_data = self.predictions_cache['digital_readiness']
            latest_projections = digital_data['yearly_projections'][-1]['city_projections']
            avg_score = np.mean([city['predicted_score'] for city in latest_projections])
            
            if avg_score < 60:
                recommendations.append({
                    'category': 'Digital Infrastructure',
                    'priority': 'MEDIUM',
                    'action': 'Accelerate digital infrastructure investment',
                    'details': f'Average readiness score: {avg_score:.1f}/100 - Below smart city threshold'
                })
            else:
                recommendations.append({
                    'category': 'Digital Infrastructure',
                    'priority': 'LOW',
                    'action': 'Maintain current digital infrastructure development pace',
                    'details': f'Average readiness score: {avg_score:.1f}/100 - On track for smart city goals'
                })
        
        # Display recommendations
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                priority_emoji = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}
                print(f"{i}. {priority_emoji[rec['priority']]} {rec['category']} ({rec['priority']} Priority)")
                print(f"   Action: {rec['action']}")
                print(f"   Context: {rec['details']}\n")
        else:
            print("❌ Unable to generate recommendations - insufficient prediction data")
    
    def create_prediction_summary(self):
        """Create a comprehensive prediction summary"""
        print("\n📋 PREDICTION SUMMARY REPORT")
        print("=" * 50)
        
        if not self.predictions_cache:
            print("❌ No predictions available")
            return
        
        summary = {
            'generated_at': datetime.now().isoformat(),
            'models_available': list(self.predictions_cache.keys()),
            'total_models': len(self.models_loaded)
        }
        
        print(f"📅 Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🤖 Active Models: {', '.join(summary['models_available'])}")
        print(f"📊 Total Trained Models: {summary['total_models']}")
        
        # Save summary to file
        summary_file = self.base_dir / "prediction_summary_report.json"
        summary.update(self.predictions_cache)
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"💾 Detailed report saved to: {summary_file}")

def main():
    """Main execution function"""
    print("🤖 City Insights 360 - Predictive Analytics Dashboard")
    print("=" * 60)
    
    # Initialize the manager
    manager = PredictiveInsightsManager()
    
    # Display all insights
    manager.show_air_quality_insights()
    manager.show_traffic_insights()
    manager.show_digital_growth_insights()
    manager.generate_planning_recommendations()
    manager.create_prediction_summary()
    
    print("\n🎉 Predictive insights analysis completed!")
    print("\n🔄 To refresh predictions, run: python src\\predictive_models.py")
    print("📊 To view in Power BI, load files from: powerbi_dashboard\\")

if __name__ == "__main__":
    main()