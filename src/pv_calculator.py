import numpy as np
import pandas as pd

class PhotovoltaicCalculator:
    def __init__(self, gamma: float = -0.0040, noct: float = 45.0, pr_others: float = 0.86):
        self.gamma = gamma
        self.noct = noct
        self.pr_others = pr_others

    def _astronomical_sun_hours(self, lat: float, doy: np.ndarray) -> np.ndarray:
        dec = np.deg2rad(23.44) * np.sin(2 * np.pi * (284 + doy) / 365)
        x = np.clip(-np.tan(np.deg2rad(lat)) * np.tan(dec), -1, 1)
        return 24 / np.pi * np.arccos(x)

    def calculate_potential(self, df: pd.DataFrame, lat: float) -> pd.DataFrame:
        df = df.copy()
        doy = df.index.dayofyear.values
        sun_hours = self._astronomical_sun_hours(lat, doy)
        
        t_day = (df["T2M"] + df["T2M_MAX"]) / 2
        g_day = df["ALLSKY_SFC_SW_DWN"] * 1000 / sun_hours
        f_wind = 9.5 / (5.7 + 3.8 * df["WS2M"])
        
        t_cell = t_day + (self.noct - 20) / 800 * g_day * f_wind
        f_temp = 1 + self.gamma * (t_cell - 25)
        
        df["T_CELULA"] = t_cell
        df["Y_KWH_KWP"] = df["ALLSKY_SFC_SW_DWN"] * f_temp * self.pr_others
        
        return df