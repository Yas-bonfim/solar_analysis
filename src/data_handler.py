import pandas as pd
import requests
import numpy as np
from pathlib import Path
from io import StringIO

class NasaPowerDataLoader:
    URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
    PARAMS = ["T2M_MAX", "T2M_MIN", "T2M", "RH2M", "PS", "WS2M", "WD2M",
              "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "T2MDEW", "CLOUD_AMT"]
    
    FAIXAS = {
        "T2M": (-10, 50), "T2M_MAX": (-10, 55), "T2M_MIN": (-15, 45), 
        "RH2M": (0, 100), "PS": (80, 110), "WS2M": (0, 40), "WD2M": (0, 360),
        "PRECTOTCORR": (0, 500), "ALLSKY_SFC_SW_DWN": (0, 12), "CLOUD_AMT": (0, 100),
    }

    def __init__(self, cache_dir: str = "data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def fetch_data(self, point_name: str, lat: float, lon: float, start_year: int = 1981, end_year: int = 2024, step: int = 5) -> pd.DataFrame:
        file_path = self.cache_dir / f"power_{point_name}.csv"
        
        if file_path.exists():
            return pd.read_csv(file_path, index_col="date", parse_dates=True)
            
        parts = []
        for year in range(start_year, end_year + 1, step):
            end_step = min(year + step - 1, end_year)
            response = requests.get(self.URL, params={
                "parameters": ",".join(self.PARAMS),
                "community": "RE",
                "latitude": lat,
                "longitude": lon,
                "start": f"{year}0101",
                "end": f"{end_step}1231",
                "format": "CSV"
            }, timeout=180)
            response.raise_for_status()
            
            csv_data = StringIO(response.text.split("-END HEADER-")[1])
            parts.append(pd.read_csv(csv_data))
            
        df = pd.concat(parts, ignore_index=True)
        df["date"] = pd.to_datetime(df[["YEAR", "MO", "DY"]].rename(
            columns={"YEAR": "year", "MO": "month", "DY": "day"}))
        df = df.drop(columns=["YEAR", "MO", "DY"]).set_index("date").astype(float)
        
        df.to_csv(file_path)
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.replace([-999, -999.0, -99, -8999], np.nan).copy()
        for col, (low, high) in self.FAIXAS.items():
            if col in df.columns:
                df.loc[(df[col] < low) | (df[col] > high), col] = np.nan
        return df.dropna()