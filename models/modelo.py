import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from models.select_data_db import consultar_datos
from models.logger import get_logger
import yaml
import os


# Configurar logger
logger = get_logger(__name__)

def modelo_prediccion():
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'settings.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            output_path=config["output_path"]
        logger.info("🚀 Configuración cargada correctamente.")

        # Cargar el modelo entrenado
        encoder = joblib.load("models/data/encoder.pkl")
        modelo = joblib.load("models/data/outputmodelo_muerte_materna.pkl")
        logger.info("🚀 cargada Modelo.")

        # Cargar nuevos datos
        query=""" SELECT edad,ocupacion,etnia,departamento,semanas,causa,categoria_causa FROM temp_carga;"""

        df = consultar_datos(query)
        logger.info("✅  Carga de dataset a Modelo.")

        # Ajustar categoría según semanas de gestación
        df['categoria_ajustada'] = df['categoria_causa']
        df.loc[df['semanas'] == 42, 'categoria_ajustada'] = 'Tardia'

        # Seleccionar columnas relevantes
        columnas_modelo = ['edad', 'ocupacion', 'etnia', 'departamento', 'semanas','causa']
        df_modelo = df[columnas_modelo].dropna()

        # Codificar variables categóricas
        categoricas = ['ocupacion', 'etnia', 'departamento']
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded = encoder.fit_transform(df_modelo[categoricas])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categoricas))
        logger.info("✅  Asiganar variables a Modelo.")

        # Combinar con variables numéricas
        df_final = pd.concat([df_modelo[['edad', 'semanas']].reset_index(drop=True), encoded_df], axis=1)

        # Realizar predicciones
        predicciones = modelo.predict(df_final)

        # Agregar predicciones al DataFrame original
        df_modelo['prediccion_categoria'] = predicciones

        # Guardar resultados

        archivo_salida = os.path.join(output_path, "predicciones_muerte_materna.csv")
        df_modelo.to_csv(archivo_salida, index=False)

        df_modelo.to_csv("predicciones_muerte_materna.csv", index=False)
        logger.info("✅  Predición de Modelo.")
        print("Predicciones guardadas en 'predicciones_muerte_materna.csv'")
        
    except Exception as e:
        logger.critical(f"Error en la predicción del modelo: {e}", exc_info=True)




