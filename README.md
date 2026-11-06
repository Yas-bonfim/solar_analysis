# Analysis and Forecast of Photovoltaic Potential

This project aims to analyze the potential for solar (photovoltaic) energy generation in the Cuiabá, Mato Grosso region and to build predictive models based on decision trees to estimate solar radiation.

The data is collected directly from the NASA POWER API, with a focus on correcting global radiation by taking local meteorological variables (such as temperature, wind, and cloud cover) into account.

## Context and Findings

Based on the preliminary Exploratory Data Analysis (EDA), we identified the following critical points for the model:

Spatial Resolution: The NASA dataset has a resolution of 1°x1° for solar variables. Therefore, all neighborhoods within the Cuiabá urban area share the same historical radiation time series, justifying the use of a single model for the region.

Climatic Constraints: Cloud cover (CLOUD_AMT) has a strong negative correlation with radiation and drives the drops in power generation during the rainy season (Oct–Apr).

Thermal Loss: Intense heat (especially between Sept–Oct) reduces the efficiency of solar panels. Stronger winds help minimize this loss.

## Repository Structure

solar_analysis/

├── src/

│   ├── __init__.py

│   ├── data_handler.py        # Data collection (NASA API), local cache, and cleanup

│   ├── pv_calculator.py       # Physical Formulas (NOCT, Specific Photovoltaic Potential)

│   └── model_comparator.py    # Feature engineering and ML training (XGBoost, LightGBM, etc.)

├── main.py                    # Main pipeline orchestration script

└── requirements.txt           # Project dependencies

## Instalação e Uso

1. Clone the repository:
   ```
   git clone https://github.com/seu-usuario/solar_analysis.git
   cd solar_analysis
   ```
2. Install the dependencies:

We recommend using a virtual environment (venv or conda).
```
pip install -r requirements.txt
```
3. Run the pipeline:
   
The main script will extract the data, calculate the physical potential, and compare the regression algorithms (Random Forest, Gradient Boosting, XGBoost, and LightGBM).
```
python main.py
```
