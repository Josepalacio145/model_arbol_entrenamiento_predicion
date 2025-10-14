import sys
import os
import yaml
from models.logger import get_logger
from sqlalchemy import create_engine
import pandas as pd


# Configurar logger
logger = get_logger(__name__)

def consultar_datos(cosnulta:str,tipo:bool= True):
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'settings.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)
        logger.info("🚀 Configuración cargada correctamente.")
        
        engine = create_engine(config["db_connection"])
        
        with engine.connect() as connection:
            if tipo:
                df = pd.read_sql(cosnulta, connection)
                logger.info("✅ script ejecutado correctamente.")
                return df
            else:
                connection.execute(cosnulta)
                logger.info("✅ Consulta ejecutada correctamente.")
                return None
    except Exception as e:
        logger.critical(f"Error al cargar la configuración: {e}", exc_info=True)