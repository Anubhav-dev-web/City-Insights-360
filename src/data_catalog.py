"""
City Insights 360 - Data Catalog and Assessment Module
=====================================================

This module provides comprehensive data cataloging, quality assessment,
and schema documentation for all urban datasets.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class DataCatalog:
    """Comprehensive data catalog for City Insights 360"""
    
    def __init__(self, data_directory: str):
        self.data_directory = Path(data_directory)
        self.datasets = {}
        self.catalog = {}
        
    def scan_datasets(self) -> Dict[str, Any]:
        """Scan all CSV files and create comprehensive catalog"""
        csv_files = list(self.data_directory.glob("*.csv"))
        
        for file_path in csv_files:
            dataset_name = file_path.stem
            print(f"📊 Analyzing {dataset_name}...")
            
            try:
                # Load dataset
                df = pd.read_csv(file_path)
                self.datasets[dataset_name] = df
                
                # Create catalog entry
                self.catalog[dataset_name] = {
                    'file_path': str(file_path),
                    'schema': self._analyze_schema(df),
                    'data_quality': self._assess_quality(df),
                    'temporal_info': self._detect_temporal_columns(df),
                    'spatial_info': self._detect_spatial_columns(df),
                    'key_metrics': self._calculate_metrics(df),
                    'domain': self._classify_domain(dataset_name, df),
                    'last_updated': datetime.now().isoformat()
                }
                
            except Exception as e:
                print(f"❌ Error analyzing {dataset_name}: {str(e)}")
                continue
        
        return self.catalog
    
    def _analyze_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze dataset schema and structure"""
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'column_info': {
                col: {
                    'dtype': str(df[col].dtype),
                    'non_null': int(df[col].count()),
                    'null_percentage': round((df[col].isnull().sum() / len(df)) * 100, 2),
                    'unique_values': int(df[col].nunique()),
                    'sample_values': df[col].dropna().head(3).tolist()
                } for col in df.columns
            }
        }
    
    def _assess_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality metrics"""
        return {
            'completeness': round((df.count().sum() / df.size) * 100, 2),
            'duplicate_rows': int(df.duplicated().sum()),
            'duplicate_percentage': round((df.duplicated().sum() / len(df)) * 100, 2),
            'columns_with_nulls': df.columns[df.isnull().any()].tolist(),
            'data_consistency': self._check_consistency(df)
        }
    
    def _detect_temporal_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect and analyze temporal columns"""
        temporal_cols = []
        time_range = {}
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['date', 'time', 'year', 'month', 'day']):
                temporal_cols.append(col)
                
                # Try to extract time range
                try:
                    if 'year' in col_lower and df[col].dtype in ['int64', 'float64']:
                        time_range[col] = {
                            'min': int(df[col].min()),
                            'max': int(df[col].max()),
                            'range_years': int(df[col].max() - df[col].min())
                        }
                    else:
                        # Try parsing as datetime
                        temp_dates = pd.to_datetime(df[col], errors='coerce')
                        if temp_dates.notna().sum() > 0:
                            time_range[col] = {
                                'min': str(temp_dates.min()),
                                'max': str(temp_dates.max()),
                                'range_days': (temp_dates.max() - temp_dates.min()).days
                            }
                except:
                    pass
        
        return {
            'temporal_columns': temporal_cols,
            'time_ranges': time_range,
            'has_temporal_data': len(temporal_cols) > 0
        }
    
    def _detect_spatial_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect and analyze spatial/geographic columns"""
        spatial_cols = []
        coordinate_pairs = []
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in 
                   ['lat', 'lon', 'city', 'state', 'country', 'location', 'address']):
                spatial_cols.append(col)
        
        # Check for coordinate pairs
        lat_cols = [col for col in df.columns if 'lat' in col.lower()]
        lon_cols = [col for col in df.columns if 'lon' in col.lower()]
        
        for lat_col in lat_cols:
            for lon_col in lon_cols:
                if (df[lat_col].dtype in ['float64', 'int64'] and 
                    df[lon_col].dtype in ['float64', 'int64']):
                    coordinate_pairs.append((lat_col, lon_col))
        
        return {
            'spatial_columns': spatial_cols,
            'coordinate_pairs': coordinate_pairs,
            'has_coordinates': len(coordinate_pairs) > 0,
            'geographic_coverage': self._analyze_geographic_coverage(df, coordinate_pairs)
        }
    
    def _analyze_geographic_coverage(self, df: pd.DataFrame, coord_pairs: List[Tuple]) -> Dict:
        """Analyze geographic coverage of the data"""
        coverage = {}
        
        if coord_pairs:
            lat_col, lon_col = coord_pairs[0]
            valid_coords = df.dropna(subset=[lat_col, lon_col])
            
            if len(valid_coords) > 0:
                coverage = {
                    'lat_range': {
                        'min': float(valid_coords[lat_col].min()),
                        'max': float(valid_coords[lat_col].max())
                    },
                    'lon_range': {
                        'min': float(valid_coords[lon_col].min()),
                        'max': float(valid_coords[lon_col].max())
                    },
                    'coordinate_count': len(valid_coords)
                }
        
        return coverage
    
    def _calculate_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate key statistical metrics"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        metrics = {}
        
        for col in numeric_cols:
            if df[col].count() > 0:  # Only if column has data
                metrics[col] = {
                    'mean': round(float(df[col].mean()), 3),
                    'median': round(float(df[col].median()), 3),
                    'std': round(float(df[col].std()), 3),
                    'min': round(float(df[col].min()), 3),
                    'max': round(float(df[col].max()), 3)
                }
        
        return metrics
    
    def _check_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data consistency issues"""
        issues = {}
        
        # Check for mixed data types in object columns
        for col in df.select_dtypes(include=['object']).columns:
            sample = df[col].dropna().head(100)
            if len(sample) > 0:
                types = set(type(x).__name__ for x in sample)
                if len(types) > 1:
                    issues[f"{col}_mixed_types"] = list(types)
        
        return issues
    
    def _classify_domain(self, dataset_name: str, df: pd.DataFrame) -> str:
        """Classify dataset into urban analytics domain"""
        name_lower = dataset_name.lower()
        columns = ' '.join(df.columns).lower()
        
        if any(term in name_lower for term in ['air', 'pollution', 'aqi', 'pm2.5', 'pm10']):
            return "Air Quality & Environment"
        elif any(term in name_lower for term in ['traffic', 'mobility', 'vehicle', 'transport']):
            return "Smart Mobility & Traffic"
        elif any(term in name_lower for term in ['crime', 'safety']):
            return "Crime & Safety"
        elif any(term in name_lower for term in ['population', 'cities', 'gdp']):
            return "Demographics & Economics"
        elif any(term in name_lower for term in ['health', 'hospital', 'medical']):
            return "Healthcare Infrastructure"
        elif any(term in name_lower for term in ['road', 'highway', 'infrastructure']):
            return "Transportation Infrastructure"
        elif any(term in name_lower for term in ['ict', 'digital', 'broadband', 'smart']):
            return "Digital Infrastructure"
        else:
            return "General Urban Data"
    
    def generate_summary_report(self) -> str:
        """Generate comprehensive summary report"""
        report = []
        report.append("=" * 80)
        report.append("CITY INSIGHTS 360 - DATA CATALOG SUMMARY")
        report.append("=" * 80)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Datasets: {len(self.catalog)}")
        report.append("")
        
        # Domain distribution
        domains = {}
        for dataset, info in self.catalog.items():
            domain = info['domain']
            domains[domain] = domains.get(domain, 0) + 1
        
        report.append("📊 DATASETS BY DOMAIN:")
        for domain, count in sorted(domains.items()):
            report.append(f"  • {domain}: {count} dataset(s)")
        report.append("")
        
        # Data volume summary
        total_rows = sum(info['schema']['rows'] for info in self.catalog.values())
        total_cols = sum(info['schema']['columns'] for info in self.catalog.values())
        
        report.append("📈 DATA VOLUME:")
        report.append(f"  • Total Records: {total_rows:,}")
        report.append(f"  • Total Columns: {total_cols:,}")
        report.append("")
        
        # Temporal coverage
        temporal_datasets = [name for name, info in self.catalog.items() 
                           if info['temporal_info']['has_temporal_data']]
        
        report.append("⏰ TEMPORAL COVERAGE:")
        report.append(f"  • Datasets with time data: {len(temporal_datasets)}")
        for dataset in temporal_datasets[:5]:  # Show first 5
            ranges = self.catalog[dataset]['temporal_info']['time_ranges']
            if ranges:
                for col, range_info in ranges.items():
                    if 'min' in range_info:
                        report.append(f"    - {dataset}: {range_info['min']} to {range_info['max']}")
        report.append("")
        
        # Spatial coverage
        spatial_datasets = [name for name, info in self.catalog.items() 
                          if info['spatial_info']['has_coordinates']]
        
        report.append("🌍 SPATIAL COVERAGE:")
        report.append(f"  • Datasets with coordinates: {len(spatial_datasets)}")
        report.append("")
        
        # Data quality overview
        report.append("✅ DATA QUALITY OVERVIEW:")
        high_quality = sum(1 for info in self.catalog.values() 
                          if info['data_quality']['completeness'] > 90)
        report.append(f"  • High quality datasets (>90% complete): {high_quality}")
        
        avg_completeness = np.mean([info['data_quality']['completeness'] 
                                   for info in self.catalog.values()])
        report.append(f"  • Average completeness: {avg_completeness:.1f}%")
        report.append("")
        
        # Key datasets for analytics
        report.append("🔑 KEY DATASETS FOR ANALYTICS:")
        key_datasets = [
            ("city_day", "Daily air quality metrics across major cities"),
            ("smart_mobility_dataset", "Real-time traffic and mobility data"),
            ("crimes.dataset", "Crime statistics and safety metrics"),
            ("World Largest Cities by Population 2024", "Global urban demographics"),
            ("ICT_Subdimension_Dataset new", "Smart city digital infrastructure")
        ]
        
        for dataset, description in key_datasets:
            if dataset in self.catalog:
                info = self.catalog[dataset]
                report.append(f"  • {dataset}:")
                report.append(f"    - {description}")
                report.append(f"    - {info['schema']['rows']:,} records, {info['schema']['columns']} columns")
                report.append(f"    - {info['data_quality']['completeness']:.1f}% complete")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def export_catalog(self, output_path: str):
        """Export catalog to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.catalog, f, indent=2, default=str)
        print(f"📄 Catalog exported to {output_path}")

if __name__ == "__main__":
    # Initialize data catalog
    data_dir = r"C:\Users\91892\OneDrive\Desktop\City Insights 360"
    catalog = DataCatalog(data_dir)
    
    # Scan all datasets
    print("🔍 Scanning datasets...")
    catalog.scan_datasets()
    
    # Generate and display summary report
    summary = catalog.generate_summary_report()
    print(summary)
    
    # Export catalog
    catalog.export_catalog(os.path.join(data_dir, "data_catalog.json"))
    
    print("\n✅ Data catalog analysis complete!")