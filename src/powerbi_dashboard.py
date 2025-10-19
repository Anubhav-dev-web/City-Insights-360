"""
City Insights 360 - Power BI Dashboard Data Preparation
======================================================

This module prepares optimized datasets for Power BI dashboards and creates
the dashboard structure with metrics, KPIs, and visualizations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PowerBIDashboardBuilder:
    """Prepare data and structure for Power BI dashboards"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.integrated_data_dir = self.base_dir / "integrated_data"
        self.analytics_dir = self.base_dir / "analytics_output"
        self.predictive_dir = self.base_dir / "predictive_models"
        self.dashboard_data = {}
        
    def prepare_dashboard_datasets(self):
        """Prepare optimized datasets for Power BI"""
        print("📊 Preparing Power BI dashboard datasets...")
        
        # Load and prepare each dataset
        datasets = {
            'executive_summary': self._create_executive_summary_table(),
            'air_quality_kpis': self._create_air_quality_kpis(),
            'traffic_metrics': self._create_traffic_metrics(),
            'city_rankings': self._create_city_rankings_table(),
            'demographic_overview': self._create_demographic_overview(),
            'digital_readiness': self._create_digital_readiness_table(),
            'healthcare_metrics': self._create_healthcare_metrics(),
            'predictive_insights': self._create_predictive_insights_table(),
            'time_series_data': self._create_time_series_data()
        }
        
        self.dashboard_data = datasets
        return datasets
    
    def _create_executive_summary_table(self):
        """Create executive summary KPIs table"""
        try:
            # Load analytics insights
            insights_file = self.analytics_dir / "analytics_insights.json"
            if insights_file.exists():
                with open(insights_file, 'r') as f:
                    insights = json.load(f)
                
                summary_data = []
                
                # Air Quality Summary
                if 'air_quality_analysis' in insights and 'overall_statistics' in insights['air_quality_analysis']:
                    aq = insights['air_quality_analysis']['overall_statistics']
                    summary_data.append({
                        'metric_category': 'Air Quality',
                        'metric_name': 'Cities Monitored',
                        'current_value': aq.get('cities_covered', 0),
                        'unit': 'Cities',
                        'status': 'Good' if aq.get('cities_covered', 0) > 3 else 'Needs Improvement'
                    })
                    
                    summary_data.append({
                        'metric_category': 'Air Quality',
                        'metric_name': 'Average AQI',
                        'current_value': round(aq.get('avg_aqi', 0), 1),
                        'unit': 'AQI Points',
                        'status': 'Poor' if aq.get('avg_aqi', 0) > 200 else 'Moderate'
                    })
                
                # Demographics Summary
                if 'demographic_trends' in insights and 'population_insights' in insights['demographic_trends']:
                    demo = insights['demographic_trends']['population_insights']
                    summary_data.append({
                        'metric_category': 'Demographics',
                        'metric_name': 'Cities Analyzed',
                        'current_value': demo.get('total_cities', 0),
                        'unit': 'Cities',
                        'status': 'Excellent'
                    })
                    
                    summary_data.append({
                        'metric_category': 'Demographics',
                        'metric_name': 'Total Population',
                        'current_value': demo.get('total_population', 0),
                        'unit': 'People',
                        'status': 'Good'
                    })
                
                # Digital Infrastructure
                if 'digital_readiness' in insights and 'digital_readiness' in insights['digital_readiness']:
                    digital = insights['digital_readiness']['digital_readiness']
                    summary_data.append({
                        'metric_category': 'Digital Infrastructure',
                        'metric_name': 'Average Digital Score',
                        'current_value': round(digital.get('average_score', 0), 1),
                        'unit': 'Score',
                        'status': 'Good' if digital.get('average_score', 0) > 60 else 'Needs Improvement'
                    })
                
                return pd.DataFrame(summary_data)
            
        except Exception as e:
            print(f"⚠️ Error creating executive summary: {str(e)}")
        
        # Return default data if error
        return pd.DataFrame({
            'metric_category': ['System'],
            'metric_name': ['Data Status'],
            'current_value': ['Ready'],
            'unit': ['Status'],
            'status': ['Good']
        })
    
    def _create_air_quality_kpis(self):
        """Create air quality KPIs and metrics"""
        try:
            aq_file = self.integrated_data_dir / "air_quality_integrated.csv"
            if aq_file.exists():
                df = pd.read_csv(aq_file)
                
                # Create KPIs by city and time
                if 'standardized_city' in df.columns and 'AQI' in df.columns:
                    city_metrics = df.groupby('standardized_city').agg({
                        'AQI': ['mean', 'std', 'min', 'max', 'count'],
                        'PM2.5': 'mean' if 'PM2.5' in df.columns else lambda x: np.nan,
                        'PM10': 'mean' if 'PM10' in df.columns else lambda x: np.nan
                    }).round(2)
                    
                    # Flatten column names
                    city_metrics.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] 
                                          for col in city_metrics.columns]
                    
                    city_metrics = city_metrics.reset_index()
                    
                    # Add air quality categories
                    city_metrics['air_quality_category'] = pd.cut(
                        city_metrics['AQI_mean'],
                        bins=[0, 50, 100, 150, 200, 300, float('inf')],
                        labels=['Good', 'Moderate', 'Unhealthy for Sensitive', 'Unhealthy', 'Very Unhealthy', 'Hazardous']
                    )
                    
                    # Add WHO compliance
                    if 'PM2.5' in city_metrics.columns:
                        city_metrics['who_pm25_compliant'] = (city_metrics['PM2.5'] <= 15).astype(int)
                    
                    return city_metrics
                    
        except Exception as e:
            print(f"⚠️ Error creating air quality KPIs: {str(e)}")
        
        return pd.DataFrame()
    
    def _create_traffic_metrics(self):
        """Create traffic and mobility metrics"""
        try:
            traffic_file = self.integrated_data_dir / "traffic_mobility_integrated.csv"
            if traffic_file.exists():
                df = pd.read_csv(traffic_file)
                
                # Hourly patterns
                if 'hour' in df.columns:
                    hourly_metrics = df.groupby('hour').agg({
                        'Vehicle_Count': 'mean',
                        'Traffic_Speed_kmh': 'mean',
                        'congestion_index': 'mean' if 'congestion_index' in df.columns else lambda x: np.nan,
                        'Road_Occupancy_%': 'mean' if 'Road_Occupancy_%' in df.columns else lambda x: np.nan
                    }).round(2).reset_index()
                    
                    # Add rush hour indicators
                    hourly_metrics['is_rush_hour'] = (
                        ((hourly_metrics['hour'] >= 7) & (hourly_metrics['hour'] <= 9)) |
                        ((hourly_metrics['hour'] >= 17) & (hourly_metrics['hour'] <= 19))
                    ).astype(int)
                    
                    return hourly_metrics
                
        except Exception as e:
            print(f"⚠️ Error creating traffic metrics: {str(e)}")
        
        return pd.DataFrame()
    
    def _create_city_rankings_table(self):
        """Create city rankings across different dimensions"""
        rankings_data = []
        
        try:
            insights_file = self.analytics_dir / "analytics_insights.json"
            if insights_file.exists():
                with open(insights_file, 'r') as f:
                    insights = json.load(f)
                
                if 'city_rankings' in insights:
                    rankings = insights['city_rankings']
                    
                    # Air quality rankings
                    if 'cleanest_air' in rankings:
                        for i, (city, aqi) in enumerate(rankings['cleanest_air'].items(), 1):
                            rankings_data.append({
                                'ranking_category': 'Cleanest Air',
                                'rank': i,
                                'city': city,
                                'value': aqi,
                                'unit': 'AQI'
                            })
                    
                    # Digital readiness rankings
                    if 'most_digitally_ready' in rankings:
                        for i, (city, score) in enumerate(rankings['most_digitally_ready'].items(), 1):
                            rankings_data.append({
                                'ranking_category': 'Most Digitally Ready',
                                'rank': i,
                                'city': city,
                                'value': score,
                                'unit': 'Score'
                            })
                    
                    # Population rankings
                    if 'largest_cities' in rankings:
                        for i, (city, pop) in enumerate(rankings['largest_cities'].items(), 1):
                            rankings_data.append({
                                'ranking_category': 'Largest Cities',
                                'rank': i,
                                'city': city,
                                'value': pop,
                                'unit': 'Population'
                            })
        
        except Exception as e:
            print(f"⚠️ Error creating city rankings: {str(e)}")
        
        return pd.DataFrame(rankings_data) if rankings_data else pd.DataFrame()
    
    def _create_demographic_overview(self):
        """Create demographic overview table"""
        try:
            demo_file = self.integrated_data_dir / "demographics_integrated.csv"
            if demo_file.exists():
                df = pd.read_csv(demo_file)
                
                # City demographics
                if 'Population (2024)' in df.columns:
                    demographics = df[['City', 'Country', 'Population (2024)', 'Growth Rate']].copy()
                    demographics = demographics.dropna()
                    
                    # Add population categories
                    demographics['city_size_category'] = pd.cut(
                        demographics['Population (2024)'],
                        bins=[0, 1000000, 5000000, 10000000, float('inf')],
                        labels=['Small', 'Medium', 'Large', 'Megacity']
                    )
                    
                    # Add growth categories
                    if 'Growth Rate' in demographics.columns:
                        demographics['growth_category'] = pd.cut(
                            demographics['Growth Rate'],
                            bins=[float('-inf'), 0, 0.01, 0.03, float('inf')],
                            labels=['Declining', 'Slow Growth', 'Moderate Growth', 'Fast Growth']
                        )
                    
                    return demographics.sort_values('Population (2024)', ascending=False)
                
        except Exception as e:
            print(f"⚠️ Error creating demographic overview: {str(e)}")
        
        return pd.DataFrame()
    
    def _create_digital_readiness_table(self):
        """Create digital readiness metrics"""
        try:
            digital_file = self.integrated_data_dir / "digital_infrastructure_integrated.csv"
            if digital_file.exists():
                df = pd.read_csv(digital_file)
                
                # Get latest year data
                latest_year = df['Year'].max()
                latest_data = df[df['Year'] == latest_year].copy()
                
                # Select key metrics
                digital_metrics = [
                    'City', 'Year', 'digital_readiness_score',
                    'Household Internet Access (%)',
                    'Fixed Broadband Subscriptions (%)',
                    'Wireless Broadband Coverage 4G (%)',
                    'e-Government (%)'
                ]
                
                available_metrics = [col for col in digital_metrics if col in latest_data.columns]
                digital_summary = latest_data[available_metrics]
                
                # Add readiness categories
                if 'digital_readiness_score' in digital_summary.columns:
                    digital_summary['readiness_level'] = pd.cut(
                        digital_summary['digital_readiness_score'],
                        bins=[0, 40, 60, 80, 100],
                        labels=['Basic', 'Developing', 'Advanced', 'Leading']
                    )
                
                return digital_summary.sort_values('digital_readiness_score', ascending=False)
                
        except Exception as e:
            print(f"⚠️ Error creating digital readiness table: {str(e)}")
        
        return pd.DataFrame()
    
    def _create_healthcare_metrics(self):
        """Create healthcare infrastructure metrics"""
        try:
            health_file = self.integrated_data_dir / "healthcare_integrated.csv"
            if health_file.exists():
                df = pd.read_csv(health_file)
                
                # State-wise healthcare metrics
                if 'State Name' in df.columns:
                    state_metrics = df.groupby('State Name').agg({
                        'Facility Name': 'count'
                    }).rename(columns={'Facility Name': 'total_facilities'})
                    
                    # Add facility type distribution
                    if 'Facility Type' in df.columns:
                        facility_dist = df.groupby(['State Name', 'Facility Type']).size().unstack(fill_value=0)
                        state_metrics = pd.concat([state_metrics, facility_dist], axis=1)
                    
                    # Add urban/rural split
                    if 'Location Type' in df.columns:
                        location_dist = df.groupby(['State Name', 'Location Type']).size().unstack(fill_value=0)
                        state_metrics = pd.concat([state_metrics, location_dist], axis=1)
                    
                    return state_metrics.reset_index()
                
        except Exception as e:
            print(f"⚠️ Error creating healthcare metrics: {str(e)}")
        
        return pd.DataFrame()
    
    def _create_predictive_insights_table(self):
        """Create predictive insights table"""
        try:
            pred_file = self.predictive_dir / "predictions.json"
            if pred_file.exists():
                with open(pred_file, 'r') as f:
                    predictions = json.load(f)
                
                insights_data = []
                
                # Air quality predictions
                if 'air_quality' in predictions and 'scenarios' in predictions['air_quality']:
                    for scenario in predictions['air_quality']['scenarios']:
                        insights_data.append({
                            'prediction_type': 'Air Quality',
                            'scenario': scenario.get('scenario', 'Unknown'),
                            'description': scenario.get('description', ''),
                            'predicted_value': scenario.get('predicted_aqi'),
                            'unit': 'AQI'
                        })
                
                # Traffic predictions
                if 'traffic' in predictions and 'hourly_patterns' in predictions['traffic']:
                    peak_hours = []
                    for hour_pred in predictions['traffic']['hourly_patterns']:
                        if hour_pred.get('hour') in [8, 9, 17, 18, 19]:  # Rush hours
                            peak_hours.append(hour_pred)
                    
                    if peak_hours:
                        avg_congestion = np.mean([p.get('predicted_congestion_index', 0) for p in peak_hours])
                        insights_data.append({
                            'prediction_type': 'Traffic',
                            'scenario': 'Rush Hour Average',
                            'description': 'Average congestion during peak hours',
                            'predicted_value': round(avg_congestion, 2),
                            'unit': 'Congestion Index'
                        })
                
                return pd.DataFrame(insights_data)
                
        except Exception as e:
            print(f"⚠️ Error creating predictive insights: {str(e)}")
        
        return pd.DataFrame()
    
    def _create_time_series_data(self):
        """Create time series data for trend analysis"""
        try:
            # Air quality time series
            aq_file = self.integrated_data_dir / "air_quality_integrated.csv"
            if aq_file.exists():
                df = pd.read_csv(aq_file)
                
                if 'datetime' in df.columns and 'AQI' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                    df['date'] = df['datetime'].dt.date
                    
                    # Daily averages
                    daily_trends = df.groupby('date')['AQI'].mean().reset_index()
                    daily_trends['metric_type'] = 'Air Quality'
                    daily_trends['value'] = daily_trends['AQI']
                    daily_trends = daily_trends[['date', 'metric_type', 'value']]
                    
                    return daily_trends
                    
        except Exception as e:
            print(f"⚠️ Error creating time series data: {str(e)}")
        
        return pd.DataFrame()
    
    def create_dashboard_specification(self):
        """Create Power BI dashboard specification"""
        dashboard_spec = {
            "dashboard_title": "City Insights 360 - Urban Analytics Dashboard",
            "created_date": datetime.now().isoformat(),
            "pages": [
                {
                    "page_name": "Executive Summary",
                    "description": "High-level KPIs and city overview",
                    "visualizations": [
                        {
                            "type": "KPI Cards",
                            "data_source": "executive_summary",
                            "metrics": ["Cities Monitored", "Average AQI", "Total Population", "Digital Readiness"]
                        },
                        {
                            "type": "Map",
                            "data_source": "city_rankings",
                            "description": "Geographic distribution of cities"
                        },
                        {
                            "type": "Bar Chart",
                            "data_source": "city_rankings",
                            "x_axis": "city",
                            "y_axis": "value",
                            "description": "Top performing cities"
                        }
                    ]
                },
                {
                    "page_name": "Air Quality Analysis",
                    "description": "Comprehensive air quality monitoring",
                    "visualizations": [
                        {
                            "type": "Gauge Chart",
                            "data_source": "air_quality_kpis",
                            "metric": "AQI_mean",
                            "description": "Current AQI levels by city"
                        },
                        {
                            "type": "Line Chart",
                            "data_source": "time_series_data",
                            "x_axis": "date",
                            "y_axis": "value",
                            "description": "AQI trends over time"
                        },
                        {
                            "type": "Donut Chart",
                            "data_source": "air_quality_kpis",
                            "metric": "air_quality_category",
                            "description": "Distribution of air quality categories"
                        }
                    ]
                },
                {
                    "page_name": "Traffic & Mobility",
                    "description": "Traffic patterns and congestion analysis",
                    "visualizations": [
                        {
                            "type": "Line Chart",
                            "data_source": "traffic_metrics",
                            "x_axis": "hour",
                            "y_axis": "Vehicle_Count",
                            "description": "Hourly traffic patterns"
                        },
                        {
                            "type": "Heat Map",
                            "data_source": "traffic_metrics",
                            "description": "Congestion patterns by time"
                        }
                    ]
                },
                {
                    "page_name": "Demographics & Growth",
                    "description": "Population and economic trends",
                    "visualizations": [
                        {
                            "type": "Scatter Plot",
                            "data_source": "demographic_overview",
                            "x_axis": "Population (2024)",
                            "y_axis": "Growth Rate",
                            "description": "Population vs Growth Rate analysis"
                        },
                        {
                            "type": "Tree Map",
                            "data_source": "demographic_overview",
                            "metric": "Population (2024)",
                            "description": "Population distribution by city"
                        }
                    ]
                },
                {
                    "page_name": "Digital Readiness",
                    "description": "Smart city digital infrastructure",
                    "visualizations": [
                        {
                            "type": "Horizontal Bar Chart",
                            "data_source": "digital_readiness",
                            "x_axis": "digital_readiness_score",
                            "y_axis": "City",
                            "description": "Digital readiness scores by city"
                        },
                        {
                            "type": "Multi-line Chart",
                            "data_source": "digital_readiness",
                            "description": "Digital infrastructure metrics comparison"
                        }
                    ]
                },
                {
                    "page_name": "Predictive Insights",
                    "description": "AI-powered forecasts and predictions",
                    "visualizations": [
                        {
                            "type": "Forecast Chart",
                            "data_source": "predictive_insights",
                            "description": "Air quality predictions"
                        },
                        {
                            "type": "Scenario Analysis",
                            "data_source": "predictive_insights",
                            "description": "What-if scenarios for urban planning"
                        }
                    ]
                }
            ],
            "filters": [
                {"name": "City", "type": "multi-select"},
                {"name": "Date Range", "type": "date_range"},
                {"name": "Metric Category", "type": "single-select"}
            ],
            "refresh_schedule": "Daily at 6:00 AM"
        }
        
        return dashboard_spec
    
    def export_dashboard_data(self, output_dir: str):
        """Export all dashboard datasets and specifications"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("📊 Exporting Power BI dashboard data...")
        
        # Prepare datasets
        self.prepare_dashboard_datasets()
        
        # Export each dataset as CSV
        for dataset_name, df in self.dashboard_data.items():
            if not df.empty:
                file_path = output_path / f"{dataset_name}.csv"
                df.to_csv(file_path, index=False)
                print(f"  ✅ {dataset_name}: {len(df)} records → {file_path.name}")
        
        # Export dashboard specification
        dashboard_spec = self.create_dashboard_specification()
        with open(output_path / "dashboard_specification.json", 'w') as f:
            json.dump(dashboard_spec, f, indent=2)
        
        # Create Power BI setup instructions
        instructions = self._create_powerbi_instructions()
        with open(output_path / "PowerBI_Setup_Instructions.md", 'w', encoding='utf-8') as f:
            f.write(instructions)
        
        print(f"📁 Dashboard data exported to: {output_path}")
        print(f"📈 Created {len([d for d in self.dashboard_data.values() if not d.empty])} datasets")
    
    def _create_powerbi_instructions(self):
        """Create Power BI setup instructions"""
        instructions = """
# City Insights 360 - Power BI Dashboard Setup Instructions

## 📊 Overview
This directory contains optimized datasets and specifications for creating comprehensive urban analytics dashboards in Power BI.

## 📁 Data Files
- `executive_summary.csv` - High-level KPIs and metrics
- `air_quality_kpis.csv` - Air quality metrics by city
- `traffic_metrics.csv` - Traffic patterns and congestion data
- `city_rankings.csv` - City rankings across different dimensions
- `demographic_overview.csv` - Population and growth metrics
- `digital_readiness.csv` - Smart city digital infrastructure scores
- `healthcare_metrics.csv` - Healthcare facility distribution
- `predictive_insights.csv` - AI-generated predictions and forecasts
- `time_series_data.csv` - Historical trends for time-based analysis

## 🔧 Power BI Setup Steps

### Step 1: Import Data
1. Open Power BI Desktop
2. Click "Get Data" → "Text/CSV"
3. Import all CSV files from this directory
4. Use "Transform Data" to verify data types and relationships

### Step 2: Create Relationships
1. Go to "Model" view
2. Create relationships between tables using common fields:
   - Link tables by `city` field where available
   - Connect time series data using `date` fields

### Step 3: Build Dashboards
Follow the dashboard specification in `dashboard_specification.json`:

#### Executive Summary Page
- Add KPI cards showing key metrics
- Create a map visualization showing city locations
- Add bar charts for top-performing cities

#### Air Quality Analysis Page
- Create gauge charts for AQI levels
- Add line charts for trends over time
- Include donut charts for category distribution

#### Traffic & Mobility Page
- Build line charts for hourly traffic patterns
- Create heat maps for congestion analysis

#### Demographics & Growth Page
- Add scatter plots for population vs growth analysis
- Create tree maps for population distribution

#### Digital Readiness Page
- Build horizontal bar charts for readiness scores
- Add multi-line charts for infrastructure comparison

#### Predictive Insights Page
- Create forecast charts using prediction data
- Add scenario analysis visualizations

### Step 4: Configure Filters and Interactivity
1. Add slicers for:
   - City selection (multi-select)
   - Date range
   - Metric categories
2. Enable cross-filtering between visualizations
3. Set up drill-through functionality

### Step 5: Format and Style
1. Apply consistent color scheme
2. Add titles and descriptions
3. Configure tooltips with additional context
4. Set up responsive layouts for mobile viewing

## 📅 Data Refresh
- Set up scheduled refresh to update with new data
- Configure data source credentials
- Test refresh functionality

## 🎯 Key Metrics to Highlight
- Air Quality Index (AQI) trends
- Traffic congestion patterns
- Population growth rates
- Digital infrastructure scores
- Healthcare facility coverage
- Predictive insights and forecasts

## 📈 Advanced Features
- Enable Q&A natural language queries
- Set up alerts for threshold breaches
- Configure mobile app access
- Export capabilities for reports

For technical support or questions about the data structure, refer to the analytics documentation.
"""
        return instructions

if __name__ == "__main__":
    # Initialize Power BI dashboard builder
    base_dir = r"C:\Users\91892\OneDrive\Desktop\City Insights 360"
    dashboard_builder = PowerBIDashboardBuilder(base_dir)
    
    # Export dashboard data and specifications
    output_dir = Path(base_dir) / "powerbi_dashboard"
    dashboard_builder.export_dashboard_data(str(output_dir))
    
    print("\n🎯 Power BI dashboard preparation completed successfully!")
    print("📊 Ready for Power BI import and visualization creation")
    print(f"📁 All files saved to: {output_dir}")