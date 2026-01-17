
# 🌆 City Insights 360 – Urban Analytics & Predictive Modeling

## 📌 Project Overview

To see the project outcome [Project visuals](https://drive.google.com/file/d/1yrnsujoAVuGjxpRYBq3bQM-0bOdqibRF/view)

**City Insights 360** is an end-to-end urban analytics project designed to analyze and predict key indicators that define how modern cities evolve.

The project combines **interactive business intelligence dashboards** with **machine learning–based predictive models** to move beyond descriptive analytics and enable forward-looking insights.

It focuses on understanding cities as **complex systems influenced by environment, mobility, health, demographics, and digital infrastructure**.

---

## 🎯 Objectives

* Analyze multi-domain urban datasets at a global city level
* Identify historical trends and cross-domain relationships
* Build predictive models to forecast future urban challenges
* Translate analytical outputs into decision-ready insights

---

## 🧩 Project Components

### 1️⃣ Descriptive Analytics (Power BI)

Interactive dashboards built using **Power BI** covering:

* 🌫️ **Air Quality**

  * AQI trends
  * PM2.5 / PM10 levels
  * High-risk cities analysis

* 🚦 **Traffic & Mobility**

  * Congestion patterns
  * Vehicle volume vs speed
  * Hourly traffic behavior

* 🏥 **Health & Sustainability**

  * Population exposure to poor AQI
  * Respiratory disease indicators
  * Green cover & clean energy adoption

* 🧍 **Demographics & Urban Growth**

  * Population density
  * Urban sprawl index
  * Growth projections

* 🌐 **Digital Infrastructure**

  * Internet penetration & speed
  * Digital payment adoption
  * Technology ecosystem readiness

Dashboards allow **city-level and region-level drill-downs across multiple years**.

---

## 2️⃣ Predictive Analytics (Python & Machine Learning)

To extend beyond visualization, predictive models were developed using Python.

### 🔮 Predictive Models Implemented

#### 🌬️ Air Quality Prediction

* **Algorithms**

  * Random Forest Regressor
  * Gradient Boosting Regressor
  * Linear Regression
* **Features**

  * Pollutant levels (PM2.5, PM10, NO₂, SO₂, O₃, CO)
  * Temporal features (hour, weekday, month)
  * City encoding
* **Evaluation Metrics**

  * R² Score
  * MAE
  * RMSE

---

#### 🚦 Traffic Congestion Forecasting

* **Algorithm**

  * Random Forest Regressor
* **Features**

  * Vehicle count
  * Traffic speed
  * Time-based patterns
  * Weather impact indicators
* **Outcome**

  * Hourly congestion pattern prediction

---

#### 💻 Digital Readiness Growth Projection

* **Algorithm**

  * Random Forest with cross-validation
* **Purpose**

  * Forecast city-level digital maturity
  * Project digital infrastructure growth over multiple years
* **Validation**

  * Cross-validated R² performance

---

## 🧠 Key Insights

* Descriptive dashboards explain **what has happened**
* Predictive models help estimate **what is likely to happen next**
* Combining BI + ML enables more informed, data-driven urban planning insights

---

## 🛠️ Tech Stack

**Analytics & Visualization**

* Power BI

**Programming & ML**

* Python
* Pandas
* NumPy
* Scikit-learn

**Modeling Techniques**

* Regression models
* Ensemble learning
* Feature engineering
* Cross-validation
* Model evaluation metrics

---

## 📁 Project Structure

```
City-Insights-360/
│
├── dashboards/
│   └── PowerBI_Report.pbix
│
├── integrated_data/
│   ├── air_quality_integrated.csv
│   ├── traffic_mobility_integrated.csv
│   ├── demographics_integrated.csv
│   └── digital_infrastructure_integrated.csv
│
├── predictive_models/
│   ├── predictive_models.py
│   ├── models/
│   ├── predictions.json
│   └── model_summary.json
│
├── README.md
└── requirements.txt
```

---

## 🚀 How to Run the Project

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/city-insights-360.git
cd city-insights-360
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Predictive Models

```bash
python predictive_models.py
```

This will:

* Train all ML models
* Evaluate performance
* Generate forecasts
* Export trained models and predictions

---

## 📊 Output

* Trained ML models (`.joblib`)
* Prediction scenarios (`predictions.json`)
* Model performance summary
* Interactive Power BI dashboards


<<<<<<< HEAD
## 📈 Learning Outcomes

* End-to-end analytics project execution
* Real-world feature engineering
* Model selection based on evaluation metrics
* Bridging BI dashboards with predictive analytics
* Translating ML outputs into business insights


## 👤 Author

**Anubhav**
 Data Analyst | Python | Power BI | Machine Learning


=======
*City Insights 360 - Empowering Smart Cities Through Data Analytics*

