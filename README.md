# Proyecto Modelo entrenamiento y clasificación ML

## 📂 Estructura del Proyecto
```
/MODEL_ARBOL/
│
├── config/                          # Archivo de configuración "variables de entorno"
│   └── settings.yaml
├── models/                          # Scripts de Extracción, Transformación y Carga
│   ├── data/
│       ├──output/                   # Archivo de salida del  entrenamiento 
│   ├── __init__.py    
│   ├── logger.py                    # Control de logs de ejecución
│   ├── modelo.py                    # Modelo para clasicar muertes tempranas o tardias
│   ├── random_forest_maternas.py    # Modelo prueba inicial
│   ├── select_data_db.py            # clase para consutlar datos en db
│   └── training_random_forest.py    # Modelo de entrenamiento
├── logs/                            # Logs de errores, ejecución, validaciones
├── README.md                        # Explicación general del proyecto
├── requirements.txt                 # Librerías usadas
└── main.py                          # Script principal para modelo
```
    
## ⚙️ Requisitos Previos
Asegúrate de tener instalado:

```
instalar las librerarias necesarias
Tener backup restaurado 
cambiar conexiones
```

## 🛠️ Instalación de librerias del proyecto
Desde la raiz del proyecto, ejecuta el siguiente comando:

## ⚙️ Ejecutar Modelo

desde la linea de comando ejecute pyhton main.py