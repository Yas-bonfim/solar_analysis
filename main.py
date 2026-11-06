from src.data_handler import NasaPowerDataLoader
from src.pv_calculator import PhotovoltaicCalculator
from src.model_comparator import SolarTreeModelComparator

def main():
    LAT, LON = -15.601, -56.097
    LOCATION_NAME = "Cuiaba_urbano"
    
    # 1. Carregamento e Limpeza
    loader = NasaPowerDataLoader()
    raw_data = loader.fetch_data(LOCATION_NAME, LAT, LON)
    clean_data = loader.clean_data(raw_data)
    
    # 2. Cálculo do Potencial Solar (Opcional para a predição da radiação, mas útil para análise)
    pv_calc = PhotovoltaicCalculator()
    enriched_data = pv_calc.calculate_potential(clean_data, LAT)
    
    # 3. Comparação dos Modelos
    comparator = SolarTreeModelComparator(target_col="ALLSKY_SFC_SW_DWN")
    results_df = comparator.run_comparison(enriched_data)
    
    print("\n=== Ranking dos Modelos ===")
    print(results_df.to_string(index=False))
    
    # 4. Análise do Modelo Vencedor
    best_model = results_df.iloc[0]["Modelo"]
    print(f"\n=== Importância das Variáveis ({best_model}) ===")
    importances = comparator.get_feature_importances(best_model)
    print(importances.round(4).to_string())

if __name__ == "__main__":
    main()