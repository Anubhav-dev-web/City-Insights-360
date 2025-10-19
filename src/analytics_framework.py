"""
City Insights 360 - Core Analytics Framework
==========================================

This module provides comprehensive analytics capabilities for urban data,
including air quality analysis, traffic patterns, demographics, and key metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Any, Tuple, Optional
import json
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class UrbanAnalytics:
    """Core analytics framework for urban insights"""
    
    def __init__(self, integrated_data_dir: str):
        self.data_dir = Path(integrated_data_dir)
        self.datasets = {}
        self.insights = {}
        self._load_integrated_data()
        
    def _load_integrated_data(self):
        """Load all integrated datasets"""
        print("📂 Loading integrated datasets...")
        
        dataset_files = {
            'air_quality': 'air_quality_integrated.csv',
            'traffic_mobility': 'traffic_mobility_integrated.csv', 
            'demographics': 'demographics_integrated.csv',
            'healthcare': 'healthcare_integrated.csv',
            'infrastructure': 'infrastructure_integrated.csv',
            'digital_infrastructure': 'digital_infrastructure_integrated.csv',
            'unified_metrics': 'unified_metrics_integrated.csv'
        }
        
        for name, filename in dataset_files.items():
            file_path = self.data_dir / filename
            if file_path.exists():
                try:
                    self.datasets[name] = pd.read_csv(file_path)
                    print(f"  ✅ {name}: {len(self.datasets[name])} records")
                except Exception as e:
                    print(f"  ❌ Error loading {name}: {str(e)}")
            else:
                print(f"  ⚠️ {filename} not found")
    
    def generate_comprehensive_insights(self) -> Dict[str, Any]:
        """Generate comprehensive analytics insights"""
        print("🔍 Generating comprehensive urban insights...")
        
        insights = {
            'air_quality_analysis': self.analyze_air_quality(),
            'traffic_patterns': self.analyze_traffic_patterns(),
            'demographic_trends': self.analyze_demographics(),
            'healthcare_accessibility': self.analyze_healthcare(),
            'digital_readiness': self.analyze_digital_infrastructure(),
            'city_rankings': self.generate_city_rankings(),
            'executive_summary': {}
        }
        
        # Generate executive summary
        insights['executive_summary'] = self._create_executive_summary(insights)
        
        self.insights = insights
        return insights
    
    def analyze_air_quality(self) -> Dict[str, Any]:
        """Comprehensive air quality analysis"""
        if 'air_quality' not in self.datasets or self.datasets['air_quality'].empty:
            return {'error': 'No air quality data available'}
        
        df = self.datasets['air_quality'].copy()
        
        analysis = {
            'overall_statistics': {},
            'city_comparisons': {},
            'temporal_trends': {},
            'pollution_hotspots': {},
            'aqi_distribution': {}
        }
        
        # Overall statistics
        if 'AQI' in df.columns:
            analysis['overall_statistics'] = {
                'total_measurements': len(df),
                'cities_covered': df['standardized_city'].nunique() if 'standardized_city' in df.columns else 0,
                'avg_aqi': float(df['AQI'].mean()),
                'max_aqi': float(df['AQI'].max()),
                'min_aqi': float(df['AQI'].min()),
                'aqi_std': float(df['AQI'].std())
            }
        
        # City comparisons
        if 'standardized_city' in df.columns and 'AQI' in df.columns:
            city_stats = df.groupby('standardized_city')['AQI'].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).round(2)
            
            # Get top 10 most polluted cities
            top_polluted = city_stats.nlargest(10, 'mean')
            analysis['city_comparisons']['most_polluted'] = top_polluted.to_dict('index')
            
            # Get top 10 cleanest cities
            cleanest = city_stats.nsmallest(10, 'mean')
            analysis['city_comparisons']['cleanest'] = cleanest.to_dict('index')
        
        # Temporal trends
        if 'datetime' in df.columns and 'AQI' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['datetime'], format='mixed').dt.date
            except:
                df['date'] = pd.to_datetime(df['datetime'], errors='coerce').dt.date
            daily_trends = df.groupby('date')['AQI'].mean().tail(365)  # Last year
            
            analysis['temporal_trends'] = {
                'recent_trend_direction': 'improving' if daily_trends.diff().mean() < 0 else 'worsening',
                'seasonal_pattern': self._detect_seasonal_pattern(df),
                'monthly_averages': df.groupby(pd.to_datetime(df['datetime'], errors='coerce').dt.month)['AQI'].mean().to_dict() if len(df) > 100 else {}
            }
        
        # AQI category distribution
        if 'AQI_Bucket' in df.columns:
            aqi_dist = df['AQI_Bucket'].value_counts(normalize=True) * 100
            analysis['aqi_distribution'] = aqi_dist.round(1).to_dict()
        
        # Pollutant analysis
        pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'O3']
        pollutant_stats = {}
        for pollutant in pollutants:
            if pollutant in df.columns:
                pollutant_stats[pollutant] = {
                    'mean': float(df[pollutant].mean()),
                    'max': float(df[pollutant].max()),
                    'cities_exceeding_standards': self._count_cities_exceeding_standards(df, pollutant)
                }
        
        analysis['pollutant_analysis'] = pollutant_stats
        
        return analysis
    
    def analyze_traffic_patterns(self) -> Dict[str, Any]:
        """Analyze traffic and mobility patterns"""
        if 'traffic_mobility' not in self.datasets or self.datasets['traffic_mobility'].empty:
            return {'error': 'No traffic data available'}
        
        df = self.datasets['traffic_mobility'].copy()
        
        analysis = {
            'overall_metrics': {},
            'temporal_patterns': {},
            'congestion_analysis': {},
            'weather_impact': {}
        }
        
        # Overall metrics
        numeric_cols = ['Vehicle_Count', 'Traffic_Speed_kmh', 'Road_Occupancy_%']
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if available_cols:
            analysis['overall_metrics'] = {
                'total_observations': len(df),
                'avg_vehicle_count': float(df['Vehicle_Count'].mean()) if 'Vehicle_Count' in df.columns else None,
                'avg_speed': float(df['Traffic_Speed_kmh'].mean()) if 'Traffic_Speed_kmh' in df.columns else None,
                'avg_road_occupancy': float(df['Road_Occupancy_%'].mean()) if 'Road_Occupancy_%' in df.columns else None,
                'avg_congestion_index': float(df['congestion_index'].mean()) if 'congestion_index' in df.columns else None
            }
        
        # Temporal patterns
        if 'hour' in df.columns and 'Vehicle_Count' in df.columns:
            hourly_patterns = df.groupby('hour')['Vehicle_Count'].mean()
            analysis['temporal_patterns'] = {
                'peak_hours': hourly_patterns.nlargest(3).index.tolist(),
                'off_peak_hours': hourly_patterns.nsmallest(3).index.tolist(),
                'hourly_averages': hourly_patterns.round(1).to_dict()
            }
        
        # Day of week patterns
        if 'day_of_week' in df.columns and 'Traffic_Speed_kmh' in df.columns:
            dow_patterns = df.groupby('day_of_week')['Traffic_Speed_kmh'].mean()
            analysis['temporal_patterns']['day_of_week'] = {
                'fastest_days': dow_patterns.nlargest(3).index.tolist(),
                'slowest_days': dow_patterns.nsmallest(3).index.tolist(),
                'daily_speeds': dow_patterns.round(1).to_dict()
            }
        
        # Congestion analysis
        if 'traffic_intensity' in df.columns:
            intensity_dist = df['traffic_intensity'].value_counts(normalize=True) * 100
            analysis['congestion_analysis']['intensity_distribution'] = intensity_dist.round(1).to_dict()
        
        # Weather impact
        if 'Weather_Condition' in df.columns and 'Traffic_Speed_kmh' in df.columns:
            weather_impact = df.groupby('Weather_Condition')['Traffic_Speed_kmh'].mean()
            analysis['weather_impact'] = {
                'speed_by_weather': weather_impact.round(1).to_dict(),
                'worst_weather_for_traffic': weather_impact.idxmin(),
                'best_weather_for_traffic': weather_impact.idxmax()
            }
        
        return analysis
    
    def analyze_demographics(self) -> Dict[str, Any]:
        """Analyze demographic and economic trends"""
        if 'demographics' not in self.datasets or self.datasets['demographics'].empty:
            return {'error': 'No demographic data available'}
        
        df = self.datasets['demographics'].copy()
        
        analysis = {
            'population_insights': {},
            'growth_patterns': {},
            'top_cities': {}
        }
        
        # Population insights
        if 'Population (2024)' in df.columns:
            analysis['population_insights'] = {
                'total_cities': len(df),
                'total_population': int(df['Population (2024)'].sum()),
                'average_city_size': int(df['Population (2024)'].mean()),
                'largest_city': df.loc[df['Population (2024)'].idxmax(), 'City'] if 'City' in df.columns else None,
                'largest_population': int(df['Population (2024)'].max())
            }
        
        # Growth patterns
        if 'Growth Rate' in df.columns:
            growth_stats = df['Growth Rate'].describe()
            analysis['growth_patterns'] = {
                'avg_growth_rate': float(growth_stats['mean']),
                'fastest_growing': df.loc[df['Growth Rate'].idxmax(), 'City'] if 'City' in df.columns else None,
                'fastest_growth_rate': float(df['Growth Rate'].max()),
                'declining_cities_count': int((df['Growth Rate'] < 0).sum()),
                'high_growth_cities': int((df['Growth Rate'] > 0.02).sum())  # >2% growth
            }
        
        # Top cities by population
        if 'Population (2024)' in df.columns and 'City' in df.columns:
            top_20 = df.nlargest(20, 'Population (2024)')
            analysis['top_cities'] = {
                'top_20_by_population': top_20[['City', 'Country', 'Population (2024)', 'Growth Rate']].to_dict('records') if 'Country' in df.columns else top_20[['City', 'Population (2024)']].to_dict('records')
            }
        
        return analysis
    
    def analyze_healthcare(self) -> Dict[str, Any]:
        """Analyze healthcare infrastructure accessibility"""
        if 'healthcare' not in self.datasets or self.datasets['healthcare'].empty:
            return {'error': 'No healthcare data available'}
        
        df = self.datasets['healthcare'].copy()
        
        analysis = {
            'infrastructure_overview': {},
            'facility_distribution': {},
            'geographic_coverage': {}
        }
        
        # Infrastructure overview
        analysis['infrastructure_overview'] = {
            'total_facilities': len(df),
            'unique_states': df['State Name'].nunique() if 'State Name' in df.columns else 0,
            'unique_districts': df['District Name'].nunique() if 'District Name' in df.columns else 0
        }
        
        # Facility type distribution
        if 'facility_category' in df.columns:
            facility_dist = df['facility_category'].value_counts()
            analysis['facility_distribution'] = facility_dist.head(10).to_dict()
        elif 'Facility Type' in df.columns:
            facility_dist = df['Facility Type'].value_counts()
            analysis['facility_distribution'] = facility_dist.head(10).to_dict()
        
        # Urban vs Rural distribution
        if 'location_type_clean' in df.columns:
            location_dist = df['location_type_clean'].value_counts(normalize=True) * 100
            analysis['geographic_coverage']['urban_rural_split'] = location_dist.round(1).to_dict()
        elif 'Location Type' in df.columns:
            location_dist = df['Location Type'].value_counts(normalize=True) * 100
            analysis['geographic_coverage']['urban_rural_split'] = location_dist.round(1).to_dict()
        
        # State-wise distribution
        if 'State Name' in df.columns:
            state_counts = df['State Name'].value_counts().head(10)
            analysis['geographic_coverage']['top_states'] = state_counts.to_dict()
        
        return analysis
    
    def analyze_digital_infrastructure(self) -> Dict[str, Any]:
        """Analyze digital infrastructure and smart city readiness"""
        if 'digital_infrastructure' not in self.datasets or self.datasets['digital_infrastructure'].empty:
            return {'error': 'No digital infrastructure data available'}
        
        df = self.datasets['digital_infrastructure'].copy()
        
        analysis = {
            'digital_readiness': {},
            'connectivity_metrics': {},
            'smart_city_indicators': {},
            'year_over_year_progress': {}
        }
        
        # Digital readiness scores
        if 'digital_readiness_score' in df.columns:
            latest_year = df['Year'].max()
            latest_data = df[df['Year'] == latest_year]
            
            readiness_stats = latest_data['digital_readiness_score'].describe()
            analysis['digital_readiness'] = {
                'average_score': float(readiness_stats['mean']),
                'top_performer': latest_data.loc[latest_data['digital_readiness_score'].idxmax(), 'City'] if 'City' in latest_data.columns else None,
                'top_score': float(latest_data['digital_readiness_score'].max()),
                'cities_above_50': int((latest_data['digital_readiness_score'] > 50).sum()),
                'digital_divide': float(readiness_stats['max'] - readiness_stats['min'])
            }
        
        # Connectivity metrics
        connectivity_cols = [
            'Household Internet Access (%)',
            'Fixed Broadband Subscriptions (%)', 
            'Wireless Broadband Coverage 4G (%)'
        ]
        
        available_conn_cols = [col for col in connectivity_cols if col in df.columns]
        if available_conn_cols:
            latest_data = df[df['Year'] == df['Year'].max()]
            connectivity_stats = {}
            
            for col in available_conn_cols:
                connectivity_stats[col.replace(' (%)', '')] = {
                    'average': float(latest_data[col].mean()),
                    'top_city': latest_data.loc[latest_data[col].idxmax(), 'City'] if 'City' in latest_data.columns else None,
                    'max_value': float(latest_data[col].max())
                }
            
            analysis['connectivity_metrics'] = connectivity_stats
        
        # Smart city indicators
        smart_cols = [
            'Smart Electricity Meters (%)',
            'Smart Water Meters (%)',
            'Traffic Monitoring (%)',
            'e-Government (%)'
        ]
        
        available_smart_cols = [col for col in smart_cols if col in df.columns]
        if available_smart_cols:
            latest_data = df[df['Year'] == df['Year'].max()]
            smart_stats = {}
            
            for col in available_smart_cols:
                smart_stats[col.replace(' (%)', '')] = float(latest_data[col].mean())
            
            analysis['smart_city_indicators'] = smart_stats
        
        # Year-over-year progress
        if len(df['Year'].unique()) > 1:
            years = sorted(df['Year'].unique())
            if len(years) >= 2 and 'digital_readiness_score' in df.columns:
                recent_year = years[-1]
                previous_year = years[-2]
                
                recent_avg = df[df['Year'] == recent_year]['digital_readiness_score'].mean()
                previous_avg = df[df['Year'] == previous_year]['digital_readiness_score'].mean()
                
                analysis['year_over_year_progress'] = {
                    'recent_year_avg': float(recent_avg),
                    'previous_year_avg': float(previous_avg),
                    'improvement': float(recent_avg - previous_avg),
                    'improvement_percentage': float(((recent_avg - previous_avg) / previous_avg) * 100)
                }
        
        return analysis
    
    def generate_city_rankings(self) -> Dict[str, Any]:
        """Generate comprehensive city rankings across multiple dimensions"""
        rankings = {}
        
        # Air Quality Rankings
        if 'air_quality' in self.datasets and not self.datasets['air_quality'].empty:
            aq_df = self.datasets['air_quality']
            if 'standardized_city' in aq_df.columns and 'AQI' in aq_df.columns:
                city_aqi = aq_df.groupby('standardized_city')['AQI'].mean()
                rankings['cleanest_air'] = city_aqi.nsmallest(10).round(1).to_dict()
                rankings['most_polluted'] = city_aqi.nlargest(10).round(1).to_dict()
        
        # Digital Readiness Rankings
        if 'digital_infrastructure' in self.datasets and not self.datasets['digital_infrastructure'].empty:
            digital_df = self.datasets['digital_infrastructure']
            latest_year = digital_df['Year'].max()
            latest_digital = digital_df[digital_df['Year'] == latest_year]
            
            if 'digital_readiness_score' in latest_digital.columns:
                digital_scores = latest_digital.set_index('standardized_city')['digital_readiness_score']
                rankings['most_digitally_ready'] = digital_scores.nlargest(10).round(1).to_dict()
        
        # Population Rankings
        if 'demographics' in self.datasets and not self.datasets['demographics'].empty:
            demo_df = self.datasets['demographics']
            if 'Population (2024)' in demo_df.columns and 'standardized_city' in demo_df.columns:
                pop_rankings = demo_df.set_index('standardized_city')['Population (2024)']
                rankings['largest_cities'] = pop_rankings.nlargest(10).to_dict()
        
        return rankings
    
    def _detect_seasonal_pattern(self, df: pd.DataFrame) -> str:
        """Detect seasonal patterns in air quality data"""
        if 'datetime' not in df.columns or 'AQI' not in df.columns:
            return 'insufficient_data'
        
        try:
            try:
                df['month'] = pd.to_datetime(df['datetime'], format='mixed').dt.month
            except:
                df['month'] = pd.to_datetime(df['datetime'], errors='coerce').dt.month
            monthly_aqi = df.groupby('month')['AQI'].mean()
            
            winter_months = [12, 1, 2]  # Dec, Jan, Feb
            summer_months = [6, 7, 8]   # Jun, Jul, Aug
            
            winter_aqi = monthly_aqi[monthly_aqi.index.isin(winter_months)].mean()
            summer_aqi = monthly_aqi[monthly_aqi.index.isin(summer_months)].mean()
            
            if winter_aqi > summer_aqi * 1.2:
                return 'worse_in_winter'
            elif summer_aqi > winter_aqi * 1.2:
                return 'worse_in_summer'
            else:
                return 'no_clear_pattern'
        except:
            return 'analysis_error'
    
    def _count_cities_exceeding_standards(self, df: pd.DataFrame, pollutant: str) -> int:
        """Count cities exceeding WHO air quality standards"""
        who_standards = {
            'PM2.5': 15,  # μg/m³ annual mean
            'PM10': 45,   # μg/m³ annual mean
            'NO2': 40,    # μg/m³ annual mean
            'SO2': 20     # μg/m³ 24-hour mean
        }
        
        if pollutant not in who_standards or pollutant not in df.columns:
            return 0
        
        threshold = who_standards[pollutant]
        city_averages = df.groupby('standardized_city')[pollutant].mean()
        return int((city_averages > threshold).sum())
    
    def _create_executive_summary(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Create executive summary of key findings"""
        summary = {
            'key_metrics': {},
            'top_concerns': [],
            'positive_trends': [],
            'recommendations': []
        }
        
        # Extract key metrics
        if 'air_quality_analysis' in insights and 'overall_statistics' in insights['air_quality_analysis']:
            aq_stats = insights['air_quality_analysis']['overall_statistics']
            summary['key_metrics']['air_quality'] = {
                'cities_monitored': aq_stats.get('cities_covered', 0),
                'average_aqi': round(aq_stats.get('avg_aqi', 0), 1)
            }
        
        if 'demographic_trends' in insights and 'population_insights' in insights['demographic_trends']:
            demo_stats = insights['demographic_trends']['population_insights']
            summary['key_metrics']['demographics'] = {
                'total_cities_analyzed': demo_stats.get('total_cities', 0),
                'total_population': demo_stats.get('total_population', 0)
            }
        
        # Identify concerns and positive trends
        if 'air_quality_analysis' in insights:
            aq = insights['air_quality_analysis']
            if 'temporal_trends' in aq and aq['temporal_trends'].get('recent_trend_direction') == 'worsening':
                summary['top_concerns'].append("Air quality showing worsening trend")
            else:
                summary['positive_trends'].append("Air quality trends stable or improving")
        
        if 'digital_readiness' in insights:
            dr = insights['digital_readiness']
            if 'year_over_year_progress' in dr and dr['year_over_year_progress'].get('improvement', 0) > 0:
                summary['positive_trends'].append("Digital infrastructure showing improvement")
        
        # Generate recommendations
        summary['recommendations'] = [
            "Implement real-time air quality monitoring in high-pollution areas",
            "Expand digital infrastructure in lower-performing cities", 
            "Develop integrated traffic management systems",
            "Enhance healthcare facility distribution in underserved areas"
        ]
        
        return summary
    
    def export_insights(self, output_dir: str):
        """Export analytics insights to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export JSON insights
        with open(output_path / "analytics_insights.json", 'w') as f:
            json.dump(self.insights, f, indent=2, default=str)
        
        # Create summary report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CITY INSIGHTS 360 - ANALYTICS SUMMARY REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        if 'executive_summary' in self.insights:
            exec_sum = self.insights['executive_summary']
            report_lines.append("🔍 KEY FINDINGS:")
            
            if 'key_metrics' in exec_sum:
                for domain, metrics in exec_sum['key_metrics'].items():
                    report_lines.append(f"  {domain.upper()}:")
                    for metric, value in metrics.items():
                        report_lines.append(f"    • {metric}: {value}")
            report_lines.append("")
            
            if 'top_concerns' in exec_sum and exec_sum['top_concerns']:
                report_lines.append("⚠️ TOP CONCERNS:")
                for concern in exec_sum['top_concerns']:
                    report_lines.append(f"  • {concern}")
                report_lines.append("")
            
            if 'positive_trends' in exec_sum and exec_sum['positive_trends']:
                report_lines.append("✅ POSITIVE TRENDS:")
                for trend in exec_sum['positive_trends']:
                    report_lines.append(f"  • {trend}")
                report_lines.append("")
        
        report_lines.append("=" * 80)
        
        with open(output_path / "analytics_summary_report.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"📊 Analytics insights exported to {output_path}")

if __name__ == "__main__":
    # Initialize analytics framework
    data_dir = r"C:\Users\91892\OneDrive\Desktop\City Insights 360\integrated_data"
    analytics = UrbanAnalytics(data_dir)
    
    # Generate comprehensive insights
    insights = analytics.generate_comprehensive_insights()
    
    # Export results
    output_dir = Path(r"C:\Users\91892\OneDrive\Desktop\City Insights 360") / "analytics_output"
    analytics.export_insights(str(output_dir))
    
    print("\n🎯 Analytics framework completed successfully!")
    print(f"📈 Generated insights for {len(analytics.datasets)} datasets")
    print(f"📁 Results saved to: {output_dir}")