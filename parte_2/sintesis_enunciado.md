* Probemos varios modelos (al menos 5 tipos distintos) reportando cual de todos fue el mejor (según la métrica AUC-ROC)
* Pretende que utilicemos técnicas para buscar la mejor configuración de hiperparámetros
* Que intentemos hacer al menos un ensamble
* Que utilicemos cross-validation para comparar los modelos 
* Que presentemos varias métricas del modelo final
    * AUC-ROC
    * Matriz de confusión
    * Accuracy
    * Precisión
    * Recall


* Que dejemos muy explícitos los pasos de pre-procesamiento/feature engineering que usamos en cada modelo
* Que dejemos toda la lógica del preprocesado en un archivo python llamado preprocesing.py 
    * Ahí estaran todas las funciones utilizadas para preprocesamiento
* Se espera que apliquemos al menos dos tecnicas de preprocesamiento distintos por cada tipo de modelo 
* Se espera que si dos modelos tienen el mismo preprocesado entonces usen la misma función en preprocessing.py


* Se espera que cada Nombre Modelo este en un notebook separado con el nombre "Nombre Modelo".ipynb
    * Dentro del mismo esté de forma clara la llamada a los preprocesados, su entrenamiento, la evaluación del mismo y finalmente una predicción en formato csv de un archivo nuevo localizado en: https://docs.google.com/spreadsheets/d/1ObsojtXfzvwicsFieGINPx500oGbUoaVTERTc69pzxE


* Se espera que por cada modelo listado en la tabla, hagamos las predicciones de este archivo y en la entrega junto con los notebook también entreguemos todas las predicciones. El nombre del archivo con las predicciones tiene que ser "Nombre Modelo".csv
* El formato esperado para las predicciones realizadas en cada .csv es igual al del archivo de ejemplo https://docs.google.com/spreadsheets/d/1jc4bfOyp80opnBnTBupqXnJajyF3a9NVuS9_c8XR7zU en donde por cada línea del archivo se tiene:
    * "id" "tiene_alto_valor_adquisitivo"


* Todas las dependencias de librerías deben estar en un requirements.txt
* La entrega se tiene que realizar en el mismo repositorio de la primera entrega, en una carpeta llamada parte_2
* Las predicciones de cada modelo se deberan guardar en el directorio parte_2/predicciones