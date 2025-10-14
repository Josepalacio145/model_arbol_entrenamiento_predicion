import os
import yaml
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from models.logger import get_logger


logger = get_logger(__name__)


import sklearn
print(sklearn.__version__)


def entrenar_modelo(datafr=None):
    try:
        # Cargar configuración
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'settings.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            output_path=config["output_path"]
        logger.info("🚀 Configuración cargada correctamente.")

                
        df_sample = datafr
        # El entrenamiento se realizar con el 30% de los datos
        df= df_sample.sample(frac=0.9, random_state=42)
        # df=df_sample
        logger.info(f"📊 Datos cargados para entrenamiento: {df.shape[0]} registros.")

        # Preprocesamiento
        df = df.dropna(subset=['edad', 'ocupacion', 'etnia', 'departamento', 'semanas','causa', 'categoria_causa'])
        df = df[df['categoria_causa'].isin(['Temprana', 'Tardia'])]
        df.loc[df['semanas'] == 42, 'categoria_causa'] = 'Tardia'

        # Codificación
        # X = pd.get_dummies(df[['edad', 'ocupacion', 'etnia', 'departamento', 'semanas','causa']])
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder

        # Definir columnas categóricas y numéricas
        categoricas = ['ocupacion', 'etnia', 'departamento', 'causa']
        numericas = ['edad', 'semanas']

        # Crear el preprocesador
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categoricas)
            ],
            remainder='passthrough'  # Deja pasar las columnas numéricas
        )

        # Crear pipeline completo
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=1000, random_state=42))
        ])

        # Preparar datos
        X = df[numericas + categoricas]
        #y = LabelEncoder().fit_transform(df['categoria_causa'])
        encoder = LabelEncoder()
        y = encoder.fit_transform(df['categoria_causa'])

        # División de datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Entrenamiento
        pipeline.fit(X_train, y_train)

        # Evaluación
        y_pred = pipeline.predict(X_test)
        reporte = classification_report(y_test, y_pred)
        logger.info(f"📊 Reporte de clasificación:\n{reporte}")
        
        


        # Guardar pipeline completo
        joblib.dump(encoder, os.path.join(output_path, "encoder_categoria_causa.pkl"))
        joblib.dump(pipeline, os.path.join(output_path, "modelo_muerte_materna.pkl"))
        logger.info(f"✅ Modelo guardado en: {os.path.join(output_path, 'modelo_muerte_materna.pkl')}")

       # y = LabelEncoder().fit_transform(df['categoria_causa'])

        # División de datos
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Entrenamiento
        #modelo = RandomForestClassifier(n_estimators=1000, random_state=42)
        #modelo.fit(X_train, y_train)

        # Evaluación
        #y_pred = modelo.predict(X_test)
        #reporte = classification_report(y_test, y_pred)
        #logger.info(f"📊 Reporte de clasificación:\n{reporte}")


        # Guardar modelo
        #output_model_path = os.path.join("models", "modelo_muerte_materna.pkl")
        #joblib.dump(encoder, output_path+"encoder.pkl")
        #joblib.dump(modelo, output_path+"modelo_muerte_materna.pkl")
        #logger.info(f"✅ Modelo guardado en: {output_path+'modelo_muerte_materna.pkl'}")  
        #joblib.dump(modelo, output_model_path)
        #logger.info(f"✅ Modelo guardado en: {output_model_path}")

        # Visualización de importancia de características
        plt.figure(figsize=(12, 6))
        #plt.bar(X.columns, pipeline.named_steps['classifier'].feature_importances_)
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        plt.xticks(rotation=45, ha='right')
        plt.title("Importancia de las características")
        plt.xlabel("Características")
        plt.ylabel("Importancia")
        output_plot_path = os.path.join("models", "importancia_caracteristicas.png")
        plt.tight_layout()
        #plt.savefig(output_plot_path)
        plt.savefig(output_path+"importancia_caracteristicas.png")
        logger.info(f"📈 Gráfico guardado en: {output_plot_path}")

        return pipeline

    except Exception as e:
        logger.critical(f"❌ Error durante el entrenamiento del modelo: {e}", exc_info=True)