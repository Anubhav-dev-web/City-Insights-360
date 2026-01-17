# City Insights 360 - Urban Analytics System

**A comprehensive smart city analytics platform for urban decision-making**

[![Version](https://img.shields.io/badge/Version-1.0.0-blue.svg)](https://github.com/cityinsights360/core)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## 🏙️ Overview

City Insights 360 is a powerful urban analytics system that transforms raw city data into actionable insights for smart city planning and sustainability decisions. Built with Python and optimized for Power BI dashboards, it provides comprehensive analysis across multiple urban domains.

📂 Access all project CSV and Excel files [here](https://drive.google.com/drive/folders/1S-u5HhM2FzlKpNLVfQPsN4xx-DDlQGYB?usp=sharing)


### Key Features

- **🌬️ Air Quality Analysis** - Real-time pollution monitoring and trend analysis
- **🚦 Traffic & Mobility** - Smart traffic pattern analysis and congestion prediction
- **👥 Demographics & Growth** - Population dynamics and urban development insights
- **💻 Digital Infrastructure** - Smart city readiness assessment
- **🏥 Healthcare Access** - Medical facility distribution and accessibility analysis
- **🤖 Predictive Analytics** - ML-powered forecasting for urban challenges
- **📊 Interactive Dashboards** - Power BI integration with automated reporting
- **⚙️ Automated Pipelines** - Scheduled data processing and validation

## 📁 Project Structure

```
City Insights 360/
├── 📄 README.md                    # This documentation
├── 📊 *.csv                       # Raw urban datasets (17 files)
├── 🔧 src/                        # Core system modules
│   ├── data_catalog.py            # Data assessment & cataloging
│   ├── data_integration.py        # Data cleaning & integration
│   ├── analytics_framework.py     # Core analytics engine
│   ├── predictive_models.py       # Machine learning models
│   ├── powerbi_dashboard.py       # Dashboard preparation
│   └── automation_pipeline.py     # Automated processing
├── 📈 integrated_data/            # Cleaned & processed datasets
├── 📊 analytics_output/           # Analysis results & insights
├── 🤖 predictive_models/         # ML models & predictions
├── 📋 powerbi_dashboard/          # Power BI ready datasets
├── 📝 logs/                       # System logs & status
└── ⚙️ config/                     # Configuration files
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** with packages: pandas, numpy, scikit-learn, matplotlib, seaborn
- **Power BI Desktop** (for dashboard creation)
- **Windows 10/11** (recommended for automation features)

### Installation

1. **Clone or download the project**
   ```powershell
   # Navigate to your desired directory
   cd "C:\Users\YourUsername\Desktop"
   ```

2. **Install Python dependencies**
   ```powershell
   pip install pandas numpy scikit-learn matplotlib seaborn joblib schedule
   ```

3. **Verify data files**
   - Ensure all 17 CSV files are in the root directory
   - Key files include: `city_day.csv`, `smart_mobility_dataset.csv`, etc.

### Running the System

**Option 1: Run Individual Components**
```powershell
# Navigate to project directory
cd "City Insights 360"

# 1. Data cataloging and assessment
python src\data_catalog.py

# 2. Data integration and cleaning  
python src\data_integration.py

# 3. Generate analytics insights
python src\analytics_framework.py

# 4. Build predictive models
python src\predictive_models.py

# 5. Prepare Power BI dashboards
python src\powerbi_dashboard.py
```

**Option 2: Full Automated Pipeline**
```powershell
# Run complete pipeline
python src\automation_pipeline.py --mode run

# Health check
python src\automation_pipeline.py --mode health
```

## 📊 Data Sources

The system processes **900,667+ records** across **17 datasets** covering:

### 🌍 Air Quality & Environment
- **city_day.csv** (18,265 records) - Daily air quality measurements
- **city_hour.csv** (Large dataset) - Hourly pollution data
- **station_day.csv** & **station_hour.csv** - Station-level monitoring

### 🚦 Smart Mobility
- **smart_mobility_dataset.csv** (5,000 records) - Traffic patterns, vehicle counts, weather impact

### 👥 Demographics & Economics  
- **World Largest Cities by Population 2024.csv** (801 records) - Global urban populations
- **cities_by_gdp.csv** - Economic indicators

### 🏥 Healthcare Infrastructure
- **geocode_health_centre.csv** (393,414 records) - Healthcare facility locations
- **govthospitalbeds2013jan.csv** - Hospital capacity data

### 🛣️ Transportation Infrastructure
- **Length_of_National_Highways.csv** - Road network data
- **Total_Road_Length_by_Category_of_Roads.csv** - Infrastructure metrics

### 💻 Digital Infrastructure
- **ICT_Subdimension_Dataset new.csv** (180 records) - Smart city digital readiness scores

## 🔍 Analytics Capabilities

### Air Quality Analysis
- **AQI Trends** - Historical and real-time air quality index tracking
- **Pollutant Analysis** - PM2.5, PM10, NO2, SO2, O3 level monitoring  
- **City Rankings** - Cleanest and most polluted urban areas
- **Seasonal Patterns** - Weather-based pollution trend detection
- **WHO Compliance** - International air quality standard assessment

### Traffic & Mobility Intelligence
- **Congestion Patterns** - Rush hour and daily traffic analysis
- **Speed Optimization** - Traffic flow efficiency metrics
- **Weather Impact** - Correlation between weather and traffic conditions
- **Predictive Routing** - ML-based traffic condition forecasting

### Demographics & Urban Growth
- **Population Dynamics** - Growth rate analysis across 1,700+ cities
- **Urban Classification** - City size categorization (Small, Medium, Large, Megacity)
- **Economic Trends** - GDP correlation with urban development
- **Migration Patterns** - Population movement insights

### Digital Readiness Assessment
- **Infrastructure Scoring** - Comprehensive digital maturity evaluation
- **Connectivity Metrics** - Broadband and internet access analysis
- **Smart Services** - E-governance and digital service availability
- **Year-over-year Progress** - Digital transformation tracking

## 🤖 Machine Learning Models

### Air Quality Prediction
- **Models**: Random Forest, Gradient Boosting, Linear Regression
- **Features**: Temporal, meteorological, pollutant concentration
- **Accuracy**: R² scores and MAE evaluation
- **Scenarios**: Current conditions vs. improved air quality projections

### Traffic Congestion Forecasting  
- **Model**: Random Forest Regressor
- **Predictions**: Hourly traffic patterns, congestion index
- **Features**: Time of day, weather conditions, vehicle counts
- **Applications**: Rush hour planning, traffic optimization

### Digital Infrastructure Growth
- **Model**: Random Forest with temporal features
- **Projections**: 3-year digital readiness forecasts
- **Metrics**: Internet access, smart infrastructure adoption
- **City Comparisons**: Benchmarking digital progress

## 📊 Power BI Dashboard Integration

### Dashboard Structure

**📋 Executive Summary Page**
- Key performance indicators (KPIs)  
- City overview metrics
- High-level trend indicators
- Geographic distribution map

**🌬️ Air Quality Analysis Page**  
- Real-time AQI gauges
- Historical trend charts
- Pollutant distribution analysis
- City comparison rankings

**🚦 Traffic & Mobility Page**
- Hourly traffic patterns
- Congestion heat maps  
- Speed vs. time analysis
- Weather impact visualization

**👥 Demographics & Growth Page**
- Population vs. growth rate scatter plots
- City size distribution tree maps
- Economic correlation analysis
- Migration pattern flows

**💻 Digital Readiness Page**
- Digital maturity scorecards
- Infrastructure comparison charts
- Progress tracking over time
- Smart city readiness levels

**🔮 Predictive Insights Page**
- ML-generated forecasts
- Scenario analysis tools  
- What-if planning capabilities
- Future trend projections

### Setting Up Power BI

1. **Import Data Files**
   ```
   powerbi_dashboard/
   ├── executive_summary.csv        # KPIs and metrics
   ├── air_quality_kpis.csv        # Air quality data
   ├── traffic_metrics.csv         # Traffic patterns  
   ├── city_rankings.csv           # City comparisons
   ├── demographic_overview.csv    # Population data
   ├── digital_readiness.csv       # Smart city metrics
   └── time_series_data.csv        # Historical trends
   ```

2. **Create Relationships**
   - Link tables using `city` and `date` fields
   - Enable cross-filtering between visualizations

3. **Build Visualizations**
   - Follow the dashboard specification in `dashboard_specification.json`
   - Use recommended chart types for each metric

4. **Configure Refresh**
   - Set up daily data refresh at 7:00 AM
   - Configure automatic model updates

## ⚙️ Automation & Scheduling

### Pipeline Automation
The system includes a comprehensive automation framework for daily operations:

**🔄 Daily Pipeline (6:00 AM)**
1. Data catalog refresh
2. Integration and cleaning
3. Analytics generation  
4. Predictive model updates
5. Dashboard data preparation
6. Quality validation

**🏥 Health Monitoring (Every 2 hours)**
- Disk space checking
- Data freshness validation  
- Log file size monitoring
- System performance tracking

**🧹 Maintenance (2:00 AM)**
- Log file cleanup
- Temporary file removal
- Data archival processes

### Configuration

Edit `config/pipeline_config.json` to customize:

```json
{
  "schedule": {
    "data_refresh": "06:00",
    "health_check": "*/2",
    "cleanup": "02:00"
  },
  "validation": {
    "min_records_threshold": 100,
    "max_null_percentage": 15.0
  },
  "notifications": {
    "enabled": false,
    "email": {
      "recipients": ["admin@city.gov"]
    }
  }
}
```

## 📈 Key Metrics & KPIs

### System Performance
- **Total Records Processed**: 900,667+
- **Cities Analyzed**: 805 unique locations
- **Data Completeness**: 97.6% average
- **Processing Time**: <5 minutes for full pipeline
- **Update Frequency**: Daily automated refresh

### Urban Analytics Results
- **Air Quality**: 5 major cities monitored, average AQI 251
- **Traffic Analysis**: 24-hour pattern recognition with rush hour identification
- **Demographics**: 1,700+ cities with population and growth analysis
- **Digital Readiness**: 30 cities evaluated across 6 years (2019-2024)
- **Healthcare**: 393,414+ facilities mapped across multiple states

## 🛠️ Technical Architecture

### Core Technologies
- **Python 3.8+** - Primary development language
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - Machine learning models
- **NumPy** - Numerical computing
- **Matplotlib/Seaborn** - Data visualization
- **Power BI** - Dashboard and reporting platform

### Data Processing Pipeline
1. **Ingestion** - CSV file reading and validation
2. **Cataloging** - Schema analysis and quality assessment  
3. **Integration** - Data cleaning, standardization, and merging
4. **Analytics** - Statistical analysis and insight generation
5. **Modeling** - Machine learning prediction and forecasting
6. **Visualization** - Dashboard-ready dataset preparation
7. **Automation** - Scheduled processing and monitoring

### Quality Assurance
- **Data Validation** - Completeness, consistency, and accuracy checks
- **Error Handling** - Robust exception management and logging
- **Performance Monitoring** - Execution time and resource tracking
- **Backup & Recovery** - Automated data preservation

## 📚 Usage Examples

### Basic Analytics
```python
# Load the analytics framework
from src.analytics_framework import UrbanAnalytics

# Initialize with integrated data
analytics = UrbanAnalytics("integrated_data/")

# Generate comprehensive insights
insights = analytics.generate_comprehensive_insights()

# Access specific analysis
air_quality = insights['air_quality_analysis']
city_rankings = insights['city_rankings']
```

### Custom Predictions
```python
# Load predictive models  
from src.predictive_models import UrbanPredictiveModels

# Initialize with data
models = UrbanPredictiveModels("integrated_data/")

# Build custom models
models.build_air_quality_prediction_model()
models.build_traffic_prediction_model()

# Generate forecasts
predictions = models.generate_predictions(forecast_horizon_days=30)
```

### Automation Control
```python
# Control pipeline automation
from src.automation_pipeline import AutomationPipeline

# Initialize pipeline
pipeline = AutomationPipeline("City Insights 360/")

# Run full processing
status = pipeline.run_full_pipeline()

# Check system health
health = pipeline.run_health_check()
```

## 🎯 Use Cases

### 🏛️ Government & Policy
- **Air Quality Monitoring** - Track pollution levels and policy effectiveness
- **Traffic Management** - Optimize signal timing and route planning
- **Urban Planning** - Data-driven city development decisions
- **Public Health** - Healthcare facility placement and capacity planning

### 🏢 Smart City Initiatives  
- **Digital Transformation** - Benchmark and track smart city progress
- **Infrastructure Investment** - Prioritize digital and physical upgrades
- **Citizen Services** - Improve service delivery and accessibility
- **Sustainability Goals** - Monitor environmental and efficiency metrics

### 📊 Research & Academia
- **Urban Studies** - Comprehensive city analysis and comparison
- **Environmental Research** - Pollution trend analysis and modeling
- **Transportation Planning** - Traffic pattern research and optimization
- **Public Policy Analysis** - Evidence-based policy development

### 💼 Private Sector
- **Real Estate Development** - Location assessment and market analysis
- **Logistics & Transportation** - Route optimization and demand forecasting
- **Technology Companies** - Smart city solution development
- **Consulting Services** - Urban analytics and strategic planning

## 🔧 Configuration & Customization

### Data Source Configuration
Modify data sources in `src/data_integration.py`:
```python
# Add new data sources
datasets_with_cities = [
    ('your_dataset', 'City_Column'),
    # Add more datasets as needed
]
```

### Analytics Customization
Extend analytics in `src/analytics_framework.py`:
```python
def analyze_custom_metric(self):
    """Add your custom analysis here"""
    # Custom analytics logic
    return custom_results
```

### Dashboard Customization
Update Power BI specifications in `src/powerbi_dashboard.py`:
```python
# Add new dashboard datasets
'custom_metrics': self._create_custom_metrics(),
```

## 📝 Logging & Monitoring

### Log Files
- **Pipeline Execution**: `logs/pipeline_YYYYMMDD.log`
- **System Health**: `logs/health_check.json`  
- **Pipeline Status**: `logs/latest_pipeline_status.json`
- **Error Tracking**: Detailed exception logging with timestamps

### Monitoring Metrics
- **Data Freshness** - Hours since last update
- **Processing Performance** - Execution time per module
- **Data Quality** - Completeness and consistency scores
- **System Resources** - Disk space and memory usage

## 🚨 Troubleshooting

### Common Issues

**📊 "No data files found"**
- Verify all 17 CSV files are in the root directory
- Check file names match exactly (case-sensitive)
- Ensure files are not corrupted or empty

**🔄 "Pipeline execution failed"**  
- Check logs in `logs/` directory for detailed errors
- Verify Python dependencies are installed
- Ensure sufficient disk space (>5GB recommended)

**📈 "Power BI data import errors"**
- Confirm dashboard CSV files exist in `powerbi_dashboard/`
- Check data types and column formats
- Verify relationships between tables

**⚙️ "Automation not working"**
- Install required packages: `pip install schedule`
- Check configuration in `config/pipeline_config.json`
- Verify Windows Task Scheduler permissions

### Performance Optimization

**🚀 Faster Processing**
- Use SSD storage for improved I/O performance  
- Increase available RAM (8GB+ recommended)
- Process data in chunks for large datasets

**💾 Memory Management**
- Monitor memory usage during processing
- Clear unnecessary variables in Python scripts
- Use data sampling for development and testing

## 🔐 Security & Privacy

### Data Protection
- **Local Processing** - All data remains on your local machine
- **No External Connections** - No data transmitted to external services
- **Access Control** - File system permissions protect sensitive data
- **Audit Trail** - Comprehensive logging of all operations

### Best Practices
- Regularly backup source data and results
- Use version control for configuration changes  
- Implement user access controls as needed
- Monitor logs for unusual activity patterns

## 🤝 Contributing

### Development Guidelines
- Follow Python PEP 8 style guidelines
- Include comprehensive docstrings for new functions
- Add error handling and logging to new features
- Update documentation for any changes

### Testing
- Test new features with sample datasets
- Validate outputs against expected results  
- Check compatibility across different data sizes
- Performance test with full datasets

## 📞 Support & Resources

### Getting Help
- 📧 **Technical Issues**: Check logs and error messages first
- 📚 **Documentation**: Refer to module docstrings and comments
- 🐛 **Bug Reports**: Include log files and reproduction steps
- 💡 **Feature Requests**: Describe use case and expected behavior

### Additional Resources
- **Power BI Documentation**: Microsoft Power BI learning resources
- **Python Libraries**: Pandas, Scikit-learn, NumPy documentation
- **Urban Analytics**: Smart city and urban planning resources
- **Data Science**: Machine learning and predictive modeling guides

## 📄 License & Credits

### License
This project is licensed under the MIT License - see LICENSE file for details.

### Data Sources
- Air quality data from environmental monitoring networks
- Traffic data from smart mobility systems
- Demographics from global urban databases  
- Healthcare infrastructure from public health records
- Digital infrastructure from smart city assessments

### Acknowledgments
- Built for urban planners, city officials, and data analysts
- Designed to support evidence-based city management
- Contributes to sustainable urban development goals
- Supports smart city transformation initiatives


*City Insights 360 - Empowering Smart Cities Through Data Analytics*
