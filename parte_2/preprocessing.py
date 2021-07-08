import requests 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_curve, auc

def obtener_datasets():
    """Se obtiene los datasets descargados desde Google Drive.

    Retorno
    --------
           df_train -> pd.DataFrame: El dataset del TP1, descargado desde Google Drive.
        df_holdhout -> pd.DataFrame: El dataset presentando en el TP2 que se usará como holdout, descargado desde Google Drive.
    """
    with requests.get("https://docs.google.com/spreadsheets/d/1-DWTP8uwVS-dZY402-dm0F9ICw_6PNqDGLmH0u8Eqa0/export?format=csv") as r, open("fiufip_dataset_old.csv", "wb") as f:
        for chunk in r.iter_content():
            f.write(chunk)
            
    with requests.get("https://docs.google.com/spreadsheets/d/1ObsojtXfzvwicsFieGINPx500oGbUoaVTERTc69pzxE/export?format=csv") as r, open("fiufip_dataset_new.csv", "wb") as f:
        for chunk in r.iter_content():
            f.write(chunk)
            
    df_train = pd.read_csv("fiufip_dataset_old.csv")
    df_holdout = pd.read_csv("fiufip_dataset_new.csv")
    
    return df_train, df_holdout


def aplicar_preparacion(df_train: pd.DataFrame):
    """Se prepara el dataset acorde al Análisis Exploratorio realizado en el TP1.

    Retorno
    --------
         X_train -> pd.DataFrame: El dataset preparado, solamente con los features sin la target.
        y_target -> numpy.ndarray: La feature target eliminada del dataset de las features.
    """
    
    # Renombración:
    X_train = df_train
    X_train.rename(columns={'ganancia_perdida_declarada_bolsa_argentina':'suma_declarada_bolsa_argentina'},inplace=True)
    X_train['rol_familiar_registrado'].mask((X_train.rol_familiar_registrado == 'casado' ) | (X_train.rol_familiar_registrado == 'casada'), 'casado_a', inplace=True)
    X_train['estado_marital'].mask(X_train.estado_marital == 'divorciado' , 'divorciado_a', inplace=True)
    X_train['estado_marital'].mask(X_train.estado_marital == 'separado', 'separado_a', inplace=True)
    
    # Missings:
    X_train['categoria_de_trabajo'].replace(np.nan,'No contesta', inplace=True)
    X_train['trabajo'].replace(np.nan,'No contesta', inplace=True)
   
    
    # Aplicando las transformacioens del TP1:
    X_train['categoria_de_trabajo'] = X_train['categoria_de_trabajo'].apply(generalizar_empleados_publicos)
    X_train['educacion_alcanzada'] = X_train['educacion_alcanzada'].apply(agrupar_educacion_alcanzada)
    #X_train['anios_estudiados'] =  X_train['anios_estudiados'].apply(agrupacion_anios_estudiados)
    
    # Eliminación de variables irrelevantes
    # TODO: Llevar a Barrios => Palermo y No-Palermo
    X_train.drop(columns=['barrio'], inplace=True) 
    

    # NO HACE FALTA:
    
    # Conversión tipos datos para optimización de memoria
        # "category": to more efficiently store the data -> https://pbpython.com/pandas_dtypes_cat.html#:~:text=The%20category%20data%20type%20in,more%20efficiently%20store%20the%20data.
    X_train = X_train.astype({
            "trabajo": "category", 
            "categoria_de_trabajo": "category",
            "genero": "category",
            "religion": "category",
            "educacion_alcanzada": "category",
            "estado_marital": "category",
            "rol_familiar_registrado": "category",        
            }) 

        # ubyte: [0, 255) -> https://numpy.org/devdocs/reference/arrays.scalars.html#numpy.ubyte 
    X_train = X_train.astype({
            "tiene_alto_valor_adquisitivo": np.ubyte, 
            "edad": np.ubyte,
            "anios_estudiados": np.ubyte,
            "horas_trabajo_registradas": np.ubyte,
            }) 


    # Obtención de feature de validación: 
    y_target = np.array(X_train[['tiene_alto_valor_adquisitivo']]).ravel()
    X_train.drop(columns=['tiene_alto_valor_adquisitivo'], inplace=True)

    return X_train, y_target

def conversion_numerica(X_train: pd.DataFrame):
    """Recibe el dataset X_train preparado y retorna el dataset modificado con sus features de tipo "category" a numéricas con OneHotEncoding y OrdinalEncoder.

    A las features con noción de orden se les asigna numero según el orden dado (por ejemplo, la feature de educacion_alcanzada)

    """
    # OneHoteo algunas:
    X_train = pd.get_dummies(X_train, drop_first=True, columns=[
        'genero', 
        'estado_marital', 
        'trabajo', 
        'categoria_de_trabajo',
        'religion', 
        'rol_familiar_registrado',
        ])

    # La que tiene noción de orden la hago de 0,1,..,6        
    X = [['Jardin', 
        'Primaria - [1,6] grado', 
        'Primaria - (6,9] grado',
        'Secundario - [1,3] anio', 
        'Secundario - (3,6] anio', 
        'Universitario - [1,3] anio',
        'Universitario - (3,6] anio']]
    enc = OrdinalEncoder(categories = X)
    X_train['educacion_alcanzada'] = enc.fit_transform(X_train.loc[:,['educacion_alcanzada']])
    X_train = X_train.astype({"educacion_alcanzada": np.ubyte})

    # ¡Para ver la inversa!
    #enc.inverse_transform(X_train[['educacion_alcanzada']])

    return X_train



def conversion_numerica_tp2(X_train: pd.DataFrame):
    """Recibe el dataset X_train preparado y retorna el dataset modificado con sus features de tipo "category" a numéricas con OneHotEncoding y OrdinalEncoder.

    A las features con noción de orden se les asigna numero según el orden dado (por ejemplo, la feature de educacion_alcanzada)

    """
    # OneHoteo algunas:
    X_train = pd.get_dummies(X_train, drop_first=True, columns=[
        'genero', 
        'estado_marital', 
        'trabajo', 
        'barrio', 
        'categoria_de_trabajo',
        'religion', 
        'rol_familiar_registrado',
        ])

    # La que tiene noción de orden la hago de 0,1,..,6        
    X = [['ciclo_inicial', 
        'segundo_ciclo', 
        'universitario_inicial',
        'universitario_avanzado']]
    enc = OrdinalEncoder(categories = X)
    X_train['educacion_alcanzada'] = enc.fit_transform(X_train.loc[:,['educacion_alcanzada']])
    X_train = X_train.astype({"educacion_alcanzada": np.ubyte})

    # ¡Para ver la inversa!
    #enc.inverse_transform(X_train[['educacion_alcanzada']])

    return X_train    


def aplicar_preparacion_tp2(df_train: pd.DataFrame):
    """Se prepara el dataset acorde al Análisis Exploratorio realizado en el TP1 y el nuevo en el TP2.

    Retorno
    --------
         X_train -> pd.DataFrame: El dataset preparado, solamente con los features sin la target.
        y_target -> numpy.ndarray: La feature target eliminada del dataset de las features.
    """
    
    # Renombración:
    X_train = df_train
    X_train.rename(columns={'ganancia_perdida_declarada_bolsa_argentina':'suma_declarada_bolsa_argentina'},inplace=True)
    X_train['rol_familiar_registrado'].mask((X_train.rol_familiar_registrado == 'casado' ) | (X_train.rol_familiar_registrado == 'casada'), 'casado_a', inplace=True)
    X_train['estado_marital'].mask(X_train.estado_marital == 'divorciado' , 'divorciado_a', inplace=True)
    X_train['estado_marital'].mask(X_train.estado_marital == 'separado', 'separado_a', inplace=True)
    
    # Missings:
    X_train['categoria_de_trabajo'].replace(np.nan,'No contesta', inplace=True)
    X_train['trabajo'].replace(np.nan,'No contesta', inplace=True)
   
    
    # Aplicando las transformacioens del TP1:
    X_train['categoria_de_trabajo'] = X_train['categoria_de_trabajo'].apply(generalizar_empleados_publicos)
    X_train['educacion_alcanzada'] = X_train['educacion_alcanzada'].apply(agrupar_educacion_alcanzada_tp2)
    X_train['barrio'] =  X_train['barrio'].apply(agrupar_palermo_o_no_tp2)
    X_train['religion'] =  X_train['religion'].apply(agrupar_cristiano_o_no_tp2)   

    # NO HACE FALTA:
    
    # Conversión tipos datos para optimización de memoria
        # "category": to more efficiently store the data -> https://pbpython.com/pandas_dtypes_cat.html#:~:text=The%20category%20data%20type%20in,more%20efficiently%20store%20the%20data.
    X_train = X_train.astype({
            "trabajo": "category", 
            "categoria_de_trabajo": "category",
            "genero": "category",
            "barrio": "category",
            "religion": "category",
            "educacion_alcanzada": "category",
            "estado_marital": "category",
            "rol_familiar_registrado": "category",        
            }) 

        # ubyte: [0, 255) -> https://numpy.org/devdocs/reference/arrays.scalars.html#numpy.ubyte 
    X_train = X_train.astype({
            "tiene_alto_valor_adquisitivo": np.ubyte, 
            "edad": np.ubyte,
            "anios_estudiados": np.ubyte,
            "horas_trabajo_registradas": np.ubyte,
            }) 


    # Obtención de feature de validación: 
    y_target = np.array(X_train[['tiene_alto_valor_adquisitivo']]).ravel()
    X_train.drop(columns=['tiene_alto_valor_adquisitivo'], inplace=True)

    return X_train, y_target


####### Se generaron nuevas features en ciertos analisis del TP1. En este preprocessing.py están dichas funciones que generan esas nuevas features.


def generalizar_empleados_publicos(categoria):
    """
        En el TP1 se usó así:

        df['categoria_de_trabajo'] = df['categoria_de_trabajo'].apply(generalizar_empleados_publicos)
        
    """
    
    if categoria in ['empleado_municipal','empleado_provincial','empleadao_estatal']:
        return 'empleado_publico'
    return categoria


def agrupar_educacion_alcanzada(categoria):
    """
        En el TP1 se usó así:

        df['educacion_alcanzada_agrupada'] = df.educacion_alcanzada.apply(agrupacion_educacion)
        
    """
        
    if categoria in ['universidad_4_anio','universiada_5_anio','universiada_6_anio']:
        return 'Universitario - (3,6] anio'
    if categoria in ['universidad_1_anio','universidad_2_anio','universidad_3_anio']:
        return 'Universitario - [1,3] anio'
    if categoria in ['1_anio','2_anio','3_anio']:
        return 'Secundario - [1,3] anio'
    if categoria in ['4_anio','5_anio','6_anio']:
        return 'Secundario - (3,6] anio'
    if categoria in ['1-4_grado','5-6_grado']:
        return 'Primaria - [1,6] grado'
    if categoria in ['7-8_grado','9_grado']:
        return 'Primaria - (6,9] grado'
    if categoria in ['preescolar']:
        return 'Jardin'    
    return categoria


def agrupar_educacion_alcanzada_tp2(categoria):
    """
        Nueva generación
        
    """
        
    if categoria in ['universidad_4_anio','universiada_5_anio','universiada_6_anio']:
        return 'universitario_avanzado'
    if categoria in ['universidad_1_anio','universidad_2_anio','universidad_3_anio']:
        return 'universitario_inicial'
    if categoria in ['1_anio','2_anio','3_anio']:
        return 'segundo_ciclo'
    if categoria in ['4_anio','5_anio','6_anio']:
        return 'segundo_ciclo'
    if categoria in ['1-4_grado','5-6_grado']:
        return 'ciclo_inicial'
    if categoria in ['7-8_grado','9_grado']:
        return 'ciclo_inicial'
    if categoria in ['preescolar']:
        return 'ciclo_inicial'    
    return categoria


def agrupar_cristiano_o_no_tp2(categoria):
    """
        Nueva generación
        
    """
    if categoria not in ['cristianismo']:
        return 'no_cristianismo'
    return categoria

def agrupar_palermo_o_no_tp2(categoria):
    """
        Nueva generación
        
    """
    if categoria not in ['Palermo']:
        return 'no_palermo'
    return categoria


def agrupacion_anios_estudiados(categoria):
    """
        En el TP1 se usó así:

        df['Anios estudiados generalizada'] = df.anios_estudiados.apply(agrupacion_anios_estudiados)
        
    """
        
    if categoria in np.arange(1,6).tolist():
        return '(0,5]'
    if categoria in np.arange(6,9).tolist():
        return '(5,8]'
    if categoria in np.arange(9,12).tolist():
        return '(8,11]'
    if categoria in np.arange(12,15).tolist():
        return '(11,14]'
    if categoria in np.arange(15,18).tolist():
        return '(14,17]'
    if categoria in np.arange(18,21).tolist():
        return '(17,20]'
    return categoria


def declaracion_actividad_en_bolsa(df: pd.DataFrame):       
    df['declaro_actividad_en_bolsa'] = df['suma_declarada_bolsa_argentina'] != 0
    df['declaro_actividad_en_bolsa'].replace(False,0)
    
    
def agrupar_edad_por_rangos(df: pd.DataFrame):
    rango_edades =  np.arange(10,100,10)
    pd.Series(pd.cut(df['edad'], bins = rango_edades))
 

################## FUNCIONES AUXILIARES USADAS

def graficar_matriz_confusion(y_true, y_pred):
    """
        Grafica la matriz de confusión para una clasificación binaria de 0 y 1.

        Siendo 0 como 'Bajo valor adquisitivo', y el caso contrario para 1.
        
        ### Parametros recibidos:
            *  y_true: array de 1xN con los N valores binarios reales.
            *  y_pred: array de 1xN con los N valores predichos por algún modelo.
    """

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(dpi=110)
    sns.heatmap(cm, 
                annot=True, 
                annot_kws={"size": 16}, 
                fmt='g', 
                square=True, 
                cmap=plt.cm.Blues,
                yticklabels=['Bajo valor', 'Alto valor'],
                xticklabels=['Bajo valor', 'Alto valor']
                )
    plt.grid(False)
    plt.title("Matriz de confusión", fontsize=16)
    plt.xlabel("Predicho", fontsize=16)
    plt.ylabel("Real", fontsize=16)
    plt.show()


def plot_roc_curves(clf, XX_test, yy_test, XX_train, yy_train):
    plt.figure(dpi=110)

    # For Test
    fpr_test, tpr_test, _ = roc_curve(yy_test, clf.predict_proba(XX_test)[:, 1])
    roc_auc_test = auc(fpr_test, tpr_test)
    # For Train
    fpr_train, tpr_train, _  = roc_curve(yy_train, clf.predict_proba(XX_train)[:, 1])
    roc_auc_train = auc(fpr_train, tpr_train)

    # Ploting
    plt.plot(
        fpr_test, tpr_test, color='red', lw=1, label=f'ROC curve for Test (area = {roc_auc_test:.2f})'
    )
    plt.plot(
        fpr_train, tpr_train, color='green', lw=1, label=f'ROC curve for Train (area = {roc_auc_train:.2f})'
    )

    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', weight= "bold")
    plt.ylabel('True Positive Rate', weight= "bold")
    plt.title('Curva ROC', weight= "bold")
    plt.legend(loc="lower right")
    plt.show()
