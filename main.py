import os
import yaml
import pandas as pd
import joblib
from joblib import dump
from models.logger import get_logger
from models.select_data_db import consultar_datos
from models.training_random_forest import entrenar_modelo

logger = get_logger("main")

def main():
    try:
        # Cargar configuración
        with open("config/settings.yaml") as f:
            config = yaml.safe_load(f)
        output_path = config["output_path"]
        modelo_path = os.path.join(output_path, "modelo_muerte_materna.pkl")
        logger.info("🚀 Configuración cargada correctamente.")

        # Consulta de datos para entrenamiento
        query_entrenamiento = """
        SELECT id_registro, fecha_evento, edad, ocupacion, etnia, semanas, departamento, causa, categoria_causa
        FROM tbl_maternas_mt;
        """
        df_entrenamiento = consultar_datos(query_entrenamiento)
        logger.info(f"📊 Datos de entrenamiento cargados: {df_entrenamiento.shape[0]} registros.")

        # Entrenar modelo
        modelo = entrenar_modelo(df_entrenamiento)
        dump(modelo, modelo_path)
        logger.info(f"✅ Modelo entrenado y guardado en: {modelo_path}")

        # Consulta de datos para predicción
        query_prediccion = """
        SELECT edad, ocupacion, etnia, departamento, semanas, causa
        FROM tbl_maternas_mt
        WHERE id_registro = 5682;
        """
        df_nuevo = consultar_datos(query_prediccion)

        # Validar columnas
        columnas_esperadas = ['edad', 'semanas', 'ocupacion', 'etnia', 'departamento', 'causa']
        if not all(col in df_nuevo.columns for col in columnas_esperadas):
            raise ValueError(f"❌ Las columnas del nuevo dataset no coinciden con las esperadas: {columnas_esperadas}")

        # Cargar modelo
        print(modelo_path)
        if not os.path.exists(modelo_path):
            raise FileNotFoundError(f"❌ No se encontró el modelo en la ruta: {modelo_path}")
        pipeline = joblib.load(modelo_path)

        # Realizar predicción
        df_pred = df_nuevo[columnas_esperadas]        
        encoder = joblib.load("models/data/output/encoder_categoria_causa.pkl")
        logger.info(f"✅ Datos Entrada para predicción: {df_pred.shape[0]} registros.")
        print(df_pred.head(1))
        prediccion = pipeline.predict(df_pred)
        clase_predicha = encoder.inverse_transform(prediccion)
        logger.info(f"✅ Predicción realizada: {prediccion}")
        print(f"Predicción: {clase_predicha}")


    except Exception as e:
        logger.critical(f"❌ Error en el proceso: {e}", exc_info=True)

if __name__ == "__main__":
    main()