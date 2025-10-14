import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from models.select_data import consultar_tabla

# Cargar el modelo entrenado
modelo = joblib.load("modelo_muerte_materna.pkl")

# Cargar nuevos datos
df = pd.read_csv("data_General.csv")

# Ajustar categoría según semanas de gestación
df['categoria_ajustada'] = df['categoria_causa']
df.loc[df['semanas'] == 42, 'categoria_ajustada'] = 'Tardia'

# Seleccionar columnas relevantes
columnas_modelo = ['edad', 'ocupacion', 'etnia', 'departamento', 'semanas']
df_modelo = df[columnas_modelo].dropna()

# Codificar variables categóricas
categoricas = ['ocupacion', 'etnia', 'departamento']
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoded = encoder.fit_transform(df_modelo[categoricas])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categoricas))

# Combinar con variables numéricas
df_final = pd.concat([df_modelo[['edad', 'semanas']].reset_index(drop=True), encoded_df], axis=1)

# Realizar predicciones
predicciones = modelo.predict(df_final)

# Agregar predicciones al DataFrame original
df_modelo['prediccion_categoria'] = predicciones

# Guardar resultados
df_modelo.to_csv("predicciones_muerte_materna.csv", index=False)
print("Predicciones guardadas en 'predicciones_muerte_materna.csv'")