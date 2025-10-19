"""
City Insights 360 - Data Integration & Cleaning Pipeline
======================================================

This module provides comprehensive data integration, cleaning, and standardization
for all urban datasets, preparing them for analytics and visualization.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Any, Tuple, Optional
import re
warnings.filterwarnings('ignore')

class DataIntegrator:
    """Comprehensive data integration and cleaning pipeline"""
    
    def __init__(self, data_directory: str, catalog_path: str = None):
        self.data_directory = Path(data_directory)
        self.catalog_path = catalog_path or str(self.data_directory / "data_catalog.json")
        self.catalog = self._load_catalog()
        self.integrated_data = {}
        self.city_master = None
        
    def _load_catalog(self) -> Dict:
        """Load data catalog"""
        try:
            with open(self.catalog_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("⚠️ Catalog not found. Please run data_catalog.py first.")
            return {}
    
    def integrate_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Main integration pipeline for all datasets"""
        print("🔄 Starting data integration pipeline...")
        
        # Step 1: Create city master list
        self._create_city_master()
        
        # Step 2: Integrate datasets by domain
        domains = {
            'air_quality': self._integrate_air_quality_data(),
            'traffic_mobility': self._integrate_traffic_data(),
            'crime_safety': self._integrate_crime_data(),
            'demographics': self._integrate_demographic_data(),
            'healthcare': self._integrate_healthcare_data(),
            'infrastructure': self._integrate_infrastructure_data(),
            'digital_infrastructure': self._integrate_digital_data()
        }
        
        # Step 3: Create unified urban metrics
        unified_metrics = self._create_unified_metrics(domains)
        
        self.integrated_data = {**domains, 'unified_metrics': unified_metrics}
        
        print("✅ Data integration complete!")
        return self.integrated_data
    
    def _create_city_master(self):
        """Create master city list with standardized names and coordinates"""
        print("🏙️ Creating city master list...")
        
        city_sources = []
        
        # Extract cities from different datasets
        datasets_with_cities = [
            ('city_day', 'City'),
            ('smart_mobility_dataset', None),  # Has coordinates
            ('World Largest Cities by Population 2024', 'City'),
            ('ICT_Subdimension_Dataset new', 'City'),
            ('geocode_health_centre', None)  # Geographic data
        ]
        
        for dataset_name, city_col in datasets_with_cities:
            if dataset_name in self.catalog:
                try:
                    df = pd.read_csv(self.catalog[dataset_name]['file_path'])
                    
                    if city_col and city_col in df.columns:
                        cities = df[city_col].dropna().unique()
                        for city in cities:
                            city_sources.append({
                                'city': str(city).strip(),
                                'source': dataset_name,
                                'standardized_name': self._standardize_city_name(str(city))
                            })
                    
                    # Extract coordinates if available
                    coord_pairs = self.catalog[dataset_name]['spatial_info']['coordinate_pairs']
                    if coord_pairs:
                        lat_col, lon_col = coord_pairs[0]
                        if lat_col in df.columns and lon_col in df.columns:
                            coord_data = df[[lat_col, lon_col]].dropna()
                            # Add coordinate info (simplified for demo)
                            
                except Exception as e:
                    print(f"⚠️ Error processing {dataset_name}: {str(e)}")
                    continue
        
        # Create master city DataFrame
        city_df = pd.DataFrame(city_sources)
        if not city_df.empty:
            self.city_master = city_df.groupby('standardized_name').agg({
                'city': 'first',
                'source': lambda x: ', '.join(set(x))
            }).reset_index()
        
        print(f"📍 Created master list with {len(self.city_master) if self.city_master is not None else 0} cities")
    
    def _standardize_city_name(self, city_name: str) -> str:
        """Standardize city names for consistent matching"""
        if pd.isna(city_name):
            return ''
        
        # Remove special characters and normalize
        standardized = re.sub(r'[^\w\s]', '', str(city_name).strip())
        standardized = ' '.join(standardized.split())  # Normalize whitespace
        
        # Common name mappings
        name_mappings = {
            'Bengaluru': 'Bangalore',
            'Kolkata': 'Calcutta',
            'Mumbai': 'Bombay',
            'Chennai': 'Madras',
            'Thiruvananthapuram': 'Trivandrum'
        }
        
        return name_mappings.get(standardized, standardized)
    
    def _integrate_air_quality_data(self) -> pd.DataFrame:
        """Integrate air quality datasets"""
        print("🌬️ Integrating air quality data...")
        
        air_quality_data = []
        
        # Process city_day data
        if 'city_day' in self.catalog:
            try:
                df = pd.read_csv(self.catalog['city_day']['file_path'])
                df['datetime'] = pd.to_datetime(df['Datetime'])
                df['data_granularity'] = 'daily'
                df['standardized_city'] = df['City'].apply(self._standardize_city_name)
                
                # Calculate AQI category scores for analysis
                df['aqi_category_score'] = df['AQI_Bucket'].map({
                    'Good': 1, 'Satisfactory': 2, 'Moderate': 3,
                    'Poor': 4, 'Very Poor': 5, 'Severe': 6
                })
                
                air_quality_data.append(df)
                
            except Exception as e:
                print(f"⚠️ Error processing city_day: {str(e)}")
        
        # Process city_hour data (sample for performance)
        if 'city_hour' in self.catalog:
            try:
                df = pd.read_csv(self.catalog['city_hour']['file_path'])
                # Sample hourly data to reduce size
                df_sample = df.sample(n=min(10000, len(df)), random_state=42)
                df_sample['datetime'] = pd.to_datetime(df_sample['Datetime'])
                df_sample['data_granularity'] = 'hourly'
                df_sample['standardized_city'] = df_sample['City'].apply(self._standardize_city_name)
                
                air_quality_data.append(df_sample)
                
            except Exception as e:
                print(f"⚠️ Error processing city_hour: {str(e)}")
        
        if air_quality_data:
            combined_aq = pd.concat(air_quality_data, ignore_index=True, sort=False)
            return self._clean_air_quality_data(combined_aq)
        
        return pd.DataFrame()
    
    def _clean_air_quality_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize air quality data"""
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Remove extreme outliers (values beyond 99.9th percentile)
        for col in ['PM2.5', 'PM10', 'NO2', 'AQI']:
            if col in df.columns:
                threshold = df[col].quantile(0.999)
                df[col] = df[col].clip(upper=threshold)
        
        # Add derived metrics
        if 'PM2.5' in df.columns and 'PM10' in df.columns:
            df['pm_ratio'] = df['PM2.5'] / df['PM10'].replace(0, np.nan)
        
        return df
    
    def _integrate_traffic_data(self) -> pd.DataFrame:
        """Integrate traffic and mobility data"""
        print("🚦 Integrating traffic data...")
        
        if 'smart_mobility_dataset' not in self.catalog:
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(self.catalog['smart_mobility_dataset']['file_path'])
            
            # Clean timestamp data
            df['timestamp'] = pd.to_datetime(df['Timestamp'])
            df['date'] = df['timestamp'].dt.date
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            
            # Add traffic condition categories
            df['traffic_intensity'] = pd.cut(df['Vehicle_Count'], 
                                           bins=[0, 50, 150, 250, float('inf')],
                                           labels=['Low', 'Medium', 'High', 'Very High'])
            
            # Calculate congestion index
            df['congestion_index'] = (df['Road_Occupancy_%'] * 0.4 + 
                                    (100 - df['Traffic_Speed_kmh']) * 0.6) / 100
            
            # Weather impact on traffic
            weather_impact = {
                'Clear': 1.0, 'Rain': 1.3, 'Snow': 1.5, 'Fog': 1.2
            }
            df['weather_impact_factor'] = df['Weather_Condition'].map(weather_impact)
            
            return df
            
        except Exception as e:
            print(f"⚠️ Error processing traffic data: {str(e)}")
            return pd.DataFrame()
    
    def _integrate_crime_data(self) -> pd.DataFrame:
        """Integrate crime and safety data"""
        print("🔒 Integrating crime data...")
        
        if 'crimes.dataset' not in self.catalog:
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(self.catalog['crimes.dataset']['file_path'])
            
            # Clean year data
            df = df.dropna(subset=['Year'])
            df['Year'] = df['Year'].astype(int)
            
            # Calculate crime rates and trends
            if 'Violent Crimes' in df.columns and 'Population' in df.columns:
                df['violent_crime_rate_per_100k'] = (df['Violent Crimes'] / df['Population']) * 100000
                df['property_crime_rate_per_100k'] = (df['Property\ncrime'] / df['Population']) * 100000
            
            # Calculate year-over-year changes
            df = df.sort_values('Year')
            df['violent_crime_change'] = df['Violent Crimes'].pct_change() * 100
            df['property_crime_change'] = df['Property\ncrime'].pct_change() * 100
            
            return df
            
        except Exception as e:
            print(f"⚠️ Error processing crime data: {str(e)}")
            return pd.DataFrame()
    
    def _integrate_demographic_data(self) -> pd.DataFrame:
        """Integrate demographic and economic data"""
        print("👥 Integrating demographic data...")
        
        demo_data = []
        
        # World cities population data
        if 'World Largest Cities by Population 2024' in self.catalog:
            try:
                df = pd.read_csv(self.catalog['World Largest Cities by Population 2024']['file_path'])
                df['standardized_city'] = df['City'].apply(self._standardize_city_name)
                df['data_source'] = 'world_cities_2024'
                demo_data.append(df)
            except Exception as e:
                print(f"⚠️ Error processing population data: {str(e)}")
        
        # GDP data if available
        if 'cities_by_gdp' in self.catalog:
            try:
                gdp_df = pd.read_csv(self.catalog['cities_by_gdp']['file_path'])
                # Process GDP data based on actual structure
                demo_data.append(gdp_df)
            except Exception as e:
                print(f"⚠️ Error processing GDP data: {str(e)}")
        
        if demo_data:
            return pd.concat(demo_data, ignore_index=True, sort=False)
        
        return pd.DataFrame()
    
    def _integrate_healthcare_data(self) -> pd.DataFrame:
        """Integrate healthcare infrastructure data"""
        print("🏥 Integrating healthcare data...")
        
        healthcare_data = []
        
        # Geocoded health centers
        if 'geocode_health_centre' in self.catalog:
            try:
                df = pd.read_csv(self.catalog['geocode_health_centre']['file_path'])
                
                # Standardize facility types
                df['facility_category'] = df['Facility Type'].str.upper()
                
                # Add urban/rural classification
                df['location_type_clean'] = df['Location Type'].fillna('Unknown')
                
                healthcare_data.append(df)
                
            except Exception as e:
                print(f"⚠️ Error processing health centers: {str(e)}")
        
        # Hospital beds data
        for dataset in ['govthospitalbeds2013jan', 'nin-health-facilities', 'phcdoclabasstpharma2012mar']:
            if dataset in self.catalog:
                try:
                    df = pd.read_csv(self.catalog[dataset]['file_path'])
                    df['data_source'] = dataset
                    healthcare_data.append(df)
                except Exception as e:
                    print(f"⚠️ Error processing {dataset}: {str(e)}")
        
        if healthcare_data:
            return pd.concat(healthcare_data, ignore_index=True, sort=False)
        
        return pd.DataFrame()
    
    def _integrate_infrastructure_data(self) -> pd.DataFrame:
        """Integrate transportation infrastructure data"""
        print("🛣️ Integrating infrastructure data...")
        
        infra_data = []
        
        datasets = [
            'Length_of_National_Highways',
            'Total_and_Surfaced_Length_of_Rural_Roads',
            'Total_Road_Length_by_Category_of_Roads'
        ]
        
        for dataset in datasets:
            if dataset in self.catalog:
                try:
                    df = pd.read_csv(self.catalog[dataset]['file_path'])
                    df['infrastructure_type'] = dataset
                    infra_data.append(df)
                except Exception as e:
                    print(f"⚠️ Error processing {dataset}: {str(e)}")
        
        if infra_data:
            return pd.concat(infra_data, ignore_index=True, sort=False)
        
        return pd.DataFrame()
    
    def _integrate_digital_data(self) -> pd.DataFrame:
        """Integrate digital infrastructure data"""
        print("💻 Integrating digital infrastructure data...")
        
        if 'ICT_Subdimension_Dataset new' not in self.catalog:
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(self.catalog['ICT_Subdimension_Dataset new']['file_path'])
            
            # Clean city names
            df['standardized_city'] = df['City'].apply(self._standardize_city_name)
            
            # Calculate digital readiness score
            digital_cols = [
                'Household Internet Access (%)',
                'Fixed Broadband Subscriptions (%)',
                'Wireless Broadband Coverage 4G (%)',
                'Smart Electricity Meters (%)',
                'e-Government (%)'
            ]
            
            available_cols = [col for col in digital_cols if col in df.columns]
            if available_cols:
                df['digital_readiness_score'] = df[available_cols].mean(axis=1)
            
            return df
            
        except Exception as e:
            print(f"⚠️ Error processing digital infrastructure: {str(e)}")
            return pd.DataFrame()
    
    def _create_unified_metrics(self, domains: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create unified city metrics dashboard"""
        print("📊 Creating unified metrics...")
        
        unified_metrics = []
        
        # Get unique cities from all datasets
        all_cities = set()
        for domain_name, df in domains.items():
            if not df.empty and 'standardized_city' in df.columns:
                all_cities.update(df['standardized_city'].dropna().unique())
        
        for city in all_cities:
            if not city or city == '':
                continue
                
            metrics = {
                'city': city,
                'last_updated': datetime.now().isoformat(),
            }
            
            # Air Quality Metrics
            if not domains['air_quality'].empty:
                city_aq = domains['air_quality'][
                    domains['air_quality']['standardized_city'] == city
                ]
                if not city_aq.empty:
                    metrics.update({
                        'avg_aqi': city_aq['AQI'].mean() if 'AQI' in city_aq.columns else None,
                        'avg_pm25': city_aq['PM2.5'].mean() if 'PM2.5' in city_aq.columns else None,
                        'air_quality_trend': 'improving' if city_aq['AQI'].diff().mean() < 0 else 'declining',
                        'air_quality_records': len(city_aq)
                    })
            
            # Traffic Metrics
            if not domains['traffic_mobility'].empty:
                # Traffic data doesn't have city names, so we'll use aggregate metrics
                traffic_df = domains['traffic_mobility']
                if not traffic_df.empty:
                    metrics.update({
                        'avg_congestion_index': traffic_df['congestion_index'].mean() if 'congestion_index' in traffic_df.columns else None,
                        'avg_traffic_speed': traffic_df['Traffic_Speed_kmh'].mean() if 'Traffic_Speed_kmh' in traffic_df.columns else None
                    })
            
            # Digital Infrastructure
            if not domains['digital_infrastructure'].empty:
                city_digital = domains['digital_infrastructure'][
                    domains['digital_infrastructure']['standardized_city'] == city
                ]
                if not city_digital.empty:
                    latest_year = city_digital['Year'].max()
                    latest_data = city_digital[city_digital['Year'] == latest_year]
                    metrics.update({
                        'digital_readiness_score': latest_data['digital_readiness_score'].mean() if 'digital_readiness_score' in latest_data.columns else None,
                        'internet_access_pct': latest_data['Household Internet Access (%)'].mean() if 'Household Internet Access (%)' in latest_data.columns else None
                    })
            
            unified_metrics.append(metrics)
        
        return pd.DataFrame(unified_metrics) if unified_metrics else pd.DataFrame()
    
    def export_integrated_data(self, output_dir: str):
        """Export integrated datasets"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("💾 Exporting integrated datasets...")
        
        for domain, df in self.integrated_data.items():
            if not df.empty:
                file_path = output_path / f"{domain}_integrated.csv"
                df.to_csv(file_path, index=False)
                print(f"  ✅ {domain}: {len(df)} records → {file_path}")
        
        # Export summary
        summary = {
            'integration_date': datetime.now().isoformat(),
            'datasets_integrated': len(self.integrated_data),
            'total_records': sum(len(df) for df in self.integrated_data.values()),
            'domains': list(self.integrated_data.keys())
        }
        
        with open(output_path / "integration_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📋 Integration summary saved to {output_path / 'integration_summary.json'}")

if __name__ == "__main__":
    # Initialize data integrator
    data_dir = r"C:\Users\91892\OneDrive\Desktop\City Insights 360"
    integrator = DataIntegrator(data_dir)
    
    # Run integration pipeline
    integrated_datasets = integrator.integrate_all_datasets()
    
    # Export results
    output_dir = Path(data_dir) / "integrated_data"
    integrator.export_integrated_data(str(output_dir))
    
    print("\n🎉 Data integration pipeline completed successfully!")
    print(f"📈 Integrated {len(integrated_datasets)} domain datasets")
    print(f"📁 Results saved to: {output_dir}")