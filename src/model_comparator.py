import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,root_mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import time

class SolarTreeModelComparator:
    def __init__(self, target_col: str = "ALLSKY_SFC_SW_DWN"):
        self.target_col = target_col
        self.features = [
            "CLOUD_AMT", "T2M_MAX", "T2M_MIN", "RH2M", 
            "PRECTOTCORR", "WS2M", "month", "day_of_year", "is_rainy_season"
        ]
        
        self.models = {
            "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=150, learning_rate=0.1, random_state=42, n_jobs=-1),
            "LightGBM": LGBMRegressor(n_estimators=150, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
        }
        self.results = []

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["month"] = df.index.month
        df["day_of_year"] = df.index.dayofyear
        df["is_rainy_season"] = np.where(df["month"].isin([5, 6, 7, 8, 9]), 0, 1)
        return df.dropna(subset=self.features + [self.target_col])

    def run_comparison(self, df: pd.DataFrame):
        print("Preparando os dados e extraindo features...")
        df_processed = self.feature_engineering(df)
        
        X = df_processed[self.features]
        y = df_processed[self.target_col]
        
        split_idx = int(len(df_processed) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"Treino: {len(X_train)} dias | Teste: {len(X_test)} dias\n")
        print("-" * 50)
        
        for name, model in self.models.items():
            start_time = time.time()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            elapsed_time = time.time() - start_time
            mae = mean_absolute_error(y_test, preds)
            rmse = root_mean_squared_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            
            self.results.append({
                "Modelo": name,
                "R2 Score": round(r2, 4),
                "MAE (kWh/m²)": round(mae, 4),
                "RMSE (kWh/m²)": round(rmse, 4),
                "Tempo de Treino (s)": round(elapsed_time, 3)
            })
            
            print(f"[{name}] Concluído em {elapsed_time:.2f}s")
            
        print("-" * 50)
        return pd.DataFrame(self.results).sort_values(by="RMSE (kWh/m²)", ascending=True)

    def get_feature_importances(self, best_model_name: str) -> pd.Series:
        model = self.models.get(best_model_name)
        if not model:
            raise ValueError(f"Modelo {best_model_name} não encontrado.")
        importances = model.feature_importances_
        return pd.Series(importances, index=self.features).sort_values(ascending=False)