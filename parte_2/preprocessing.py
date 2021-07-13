import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import KBinsDiscretizer

def obtener_datasets():
    """Se obtiene los datasets descargados desde Google Drive de la materia.

    Retorno
    --------
           df_train -> pd.DataFrame: El dataset del TP1 que se usará para entrenar.
        df_holdhout -> pd.DataFrame: El dataset presentando en el TP2 que se usará como holdout.
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


def aplicar_preparacion_holdout(df_for_prediction: pd.DataFrame, generalizada: bool):
    """
    Prepara el dataset para predecir. Elimina aquellas features que durante el entrenamiento no se tuvieron en cuenta, es decir la 'id' y 'representatividad_poblacional'.
    Además se le aplica la función de 'aplicar_preparacion()' o 'aplicar_preparacion_generalizado()' según el booleano recibido.
    
    Parametros recibidos
    --------
        df -> pd.DataFrame: El dataset obtenido mediante 'obtener_datasets()'
        generalizada -> bool: Un booleano que indica si se desea aplicar la preparacion generalizada o no.

    Retorno
    --------
        X_df -> pd.DataFrame: El dataset listo para predecir, solamente con los features sin la target.
    """
    #por las dudas ordenamos las columnas según el id recibido
    df_new = df_for_prediction.sort_values('id')
    
    # borramos las columnas que no se usaran para la predicción
    df_new = df_new.drop(columns=['id', 'representatividad_poblacional'])
    
    # agrego una columna de si tiene alto valor adquisitivo totalmente random para aplicar la función de preparacion que ya teniamos, no nos interesará esta variable al aplicar la preparación.
    df_new['tiene_alto_valor_adquisitivo'] = np.random.randint(0, 2, df_new.shape[0])
    
    #por las dudas ordenamos las columnas según el id recibido
    if not generalizada: 
        X_df, _ = aplicar_preparacion(df_new)
    else:
        X_df, _ = aplicar_preparacion_generalizado(df_new)

    return X_df


def aplicar_preparacion(df: pd.DataFrame):
    """Se prepara el dataset para entrenar acorde al Análisis Exploratorio realizado en el TP1.

    Parametros recibidos
    --------
        df -> pd.DataFrame: El dataset obtenido mediante 'obtener_datasets()'

    Retorno
    --------
        X_train -> pd.DataFrame: El dataset preparado, solamente con los features sin la target.
        y_target -> numpy.ndarray: La feature target eliminada del dataset de las features.
    """
    
    renombrar_variables(df)
    solucionar_missings(df)
   
    # Aplicando las transformacioens del TP1:
    df['categoria_de_trabajo'] = df['categoria_de_trabajo'].apply(generalizar_empleados_publicos)
    df['educacion_alcanzada'] = df['educacion_alcanzada'].apply(agrupar_educacion_alcanzada)
    
    # Eliminación de variables irrelevantes
    df.drop(columns=['barrio'], inplace=True) 
    
    conversion_tipos(df)

    # Obtención de feature de validación: 
    y_target = obtener_feature_validacion(df)

    return df, y_target


def aplicar_preparacion_generalizado(df: pd.DataFrame):
    """Se prepara el dataset para entrenar acorde a un nuevo Análisis Exploratorio.

    Parametros recibidos
    --------
        df -> pd.DataFrame: El dataset obtenido mediante 'obtener_datasets()'

    Retorno
    --------
        X_train -> pd.DataFrame: El dataset preparado, solamente con los features sin la target.
        y_target -> numpy.ndarray: La feature target eliminada del dataset de las features.
    """
    
    renombrar_variables(df)
    solucionar_missings(df)
   
    # Aplicando nueva transformación.
    df['barrio'] =  df['barrio'].apply(agrupar_palermo_o_no_tp2)
    
    conversion_tipos(df)
    df = df.astype({"barrio": "category"}) 

    # Obtención de feature de validación: 
    y_target = obtener_feature_validacion(df)

    return df, y_target


def conversion_numerica(X_train: pd.DataFrame):
    """ 
    A las features con noción de orden se les asigna numero según el orden dado (por ejemplo, la feature de educacion_alcanzada) con OrdinalEncoder.
    Y a las features sin orden se le aplica OneHotEncoding.
    
    Parametros recibidos
    --------
        X_train -> pd.DataFrame: el dataset preparado con 'aplicar_preparacion()'

    Retorno
    --------
        X_train -> pd.DataFrame: El dataset modificando las features de tipo "category" a numéricas con OneHotEncoding y OrdinalEncoder.
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


    print("Aplicando 'conversion_numerica' en las variables categóricas.")
    return X_train


def conversion_numerica_generalizada(X_train: pd.DataFrame):
    """ 
    A las features con noción de orden se les asigna numero según el orden dado (por ejemplo, la feature de educacion_alcanzada) con OrdinalEncoder.
    Y a las features sin orden se le aplica OneHotEncoding.
    
    Parametros recibidos
    --------
        X_train -> pd.DataFrame: el dataset preparado con 'aplicar_preparacion_generalizado()'

    Retorno
    --------
        X_train -> pd.DataFrame: El dataset modificando las features de tipo "category" a numéricas con OneHotEncoding y OrdinalEncoder.
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
    X = [['preescolar', 
        '1-4_grado','5-6_grado','7-8_grado','9_grado', 
        '1_anio','2_anio','3_anio','4_anio','5_anio','6_anio',
        'universidad_1_anio','universidad_2_anio','universidad_3_anio','universidad_4_anio','universiada_5_anio','universiada_6_anio']]
    enc = OrdinalEncoder(categories = X)
    X_train['educacion_alcanzada'] = enc.fit_transform(X_train.loc[:,['educacion_alcanzada']])
    X_train = X_train.astype({"educacion_alcanzada": np.ubyte})

    print("Aplicando 'conversion_numerica_generalizada' en las variables categóricas.")
    return X_train    


def get_dataframe_polynomial(df: pd.DataFrame, grade: np.uint8, interaction_only: bool):
    """ 
    Expande el dataset con PolynomialFeatures teniendo en cuentas las 4 columnas numéricas del mismo.
    
    Parametros recibidos
    --------
        df -> pd.DataFrame: El dataset obtenido mediante 'obtener_datasets()'.
        grade -> np.uint8: el numero de grado para aplicar con PolynomialFeatures
        interaction_only -> bool: indicador para para las potencias entre las nuevas features mediante

    Retorno
    --------
        df_poly -> pd.DataFrame: retorna el dataset recibido agregandole las nuevas features generadas a partir de PolynomialFeature segun los parametros recibidos.
    """
    
    print('Dataset inicial con', len(df.columns), 'features...')

    to_expand = df[['anios_estudiados', 'edad', 'suma_declarada_bolsa_argentina', 'horas_trabajo_registradas']]
    df_old = df.drop(columns =['anios_estudiados', 'edad', 'suma_declarada_bolsa_argentina', 'horas_trabajo_registradas'] )
    poly = PolynomialFeatures(grade, interaction_only = interaction_only)
    df_expand = pd.DataFrame(poly.fit_transform(to_expand))
    df_expand = filter_by_variance(df_expand, 0)
    df_poly = pd.concat([df_expand,df_old], axis=1)
    print('Dataset nuevo con PolynomialFeature con', len(df_poly.columns), 'features...')
    return df_poly



def get_dataframe_polynomial_all(df: pd.DataFrame, grade: np.uint8, interaction_only: bool):
    """ 
    Expande el dataset con PolynomialFeatures teniendo en cuenta todas las columnas del dataset recibido.
    
    Parametros recibidos
    --------
        df -> pd.DataFrame: El dataset obtenido mediante 'obtener_datasets()'.
        grade -> np.uint8: el numero de grado para aplicar con PolynomialFeatures
        interaction_only -> bool: indicador para para las potencias entre las nuevas features mediante

    Retorno
    --------
        df_poly -> pd.DataFrame: retorna el dataset recibido agregandole las nuevas features generadas a partir de PolynomialFeature segun los parametros recibidos.
    """
    
    print('Dataset inicial con', len(df.columns), 'features...')
    poly = PolynomialFeatures(grade, interaction_only = interaction_only)
    df_expand = pd.DataFrame(poly.fit_transform(df))
    df_expand = filter_by_variance(df_expand, 0)
    print('Dataset nuevo con PolynomialFeature con', len(df_expand.columns), 'features...')
    return df_expand    

def reduccion_numerica(df: pd.DataFrame, varianza_explicada: np.uint8 = 0.95):
    """ 
    Reduce la dimensionalidad del dataset recibido mediante TruncatedSVD manteniendo un 95% de varianza por default.
    Antes de aplicar TruncatedSVD, se re-escala los datos mediante MinMaxScaler().
            
    Parametros recibidos
    --------
        df -> pd.DataFrame: recibe el dataset con el preprocesamiento aplicado por la función de 'aplicar_preparacion()' y, 'conversion_numerica' ó 'conversion_numerica_generalizada'
        varianza_explicada -> np.uint8: por default 0.95, pero se puede recibir la cantidad deseada entre 0 y 1.

    Retorno
    --------
        X_df_numerico_scaled_svd -> pd.DataFrame: retorna el dataset reducido.
    """
    scaled = MinMaxScaler().fit_transform(df)
    X_df_numerico_scaled = pd.DataFrame(scaled)
    print("Aplicando MinMaxScaler previo al TruncatedSVD...")
    svd = TruncatedSVD(n_components=X_df_numerico_scaled.shape[1]-1, n_iter=10, random_state=10)
    svd.fit(X_df_numerico_scaled)

    var_cumu = np.cumsum(svd.explained_variance_ratio_) 
    k = np.argmax(var_cumu > varianza_explicada )
        
    svd = TruncatedSVD(n_components=k +1 , n_iter=10, random_state=10)
    X_df_numerico_scaled_svd = pd.DataFrame(svd.fit_transform(X_df_numerico_scaled))
    print('TruncatedSVD aplicado con',  k+1, 'componentes finales se explica una varianza de: %.10f' % svd.explained_variance_ratio_.sum())
    return X_df_numerico_scaled_svd


def reduccion_rfecv(estimator, X_df, y_df, min_features_to_select, step, n_jobs, scoring, cv):
    selector = RFECV(
                    estimator=estimator,
                    min_features_to_select=min_features_to_select,
                    step=step,
                    n_jobs=n_jobs,
                    scoring=scoring,
                    cv=cv,
                    )
    selector = selector.fit(X_df, y_df)

    f = selector.get_support(1) 
    X_reduced = X_df[X_df.columns[f]] 
    return X_reduced

def obtener_features_continuas(df: pd.DataFrame):
    return df[['edad', 'suma_declarada_bolsa_argentina']]

def get_dataframe_scaled(df, scaler_r):
    scaled = scaler_r.fit_transform(df)
    return pd.DataFrame(scaled, index = df.index, columns = df.columns)

def obtener_features_discretas(df: pd.DataFrame):
    df_d = df[[
        'edad',
        'anios_estudiados',
        'categoria_de_trabajo',
        'educacion_alcanzada', 
        'estado_marital', 
        'genero','religion', 
        'rol_familiar_registrado', 
        'suma_declarada_bolsa_argentina',
        'horas_trabajo_registradas',
        'trabajo']].copy()
    df_d['edad'] = KBinsDiscretizer(n_bins=10, encode='ordinal',strategy = "kmeans").fit_transform(df_d.loc[:,['edad']])
    df_d['suma_declarada_bolsa_argentina'] = KBinsDiscretizer(n_bins=8, encode='ordinal',strategy = "kmeans").fit_transform(df_d.loc[:,['suma_declarada_bolsa_argentina']])
    df_d['horas_trabajo_registradas'] = KBinsDiscretizer(n_bins=6, encode='ordinal',strategy = "kmeans").fit_transform(df_d.loc[:,['horas_trabajo_registradas']])

    df_d_n = conversion_numerica(df_d)
    return df_d_n


#######  FUNCIONES GENERACIÓN DE FEATURES Y DEMÁS AUXILIARES


def filter_by_variance(df, threshold):
    cols_con_varianza = df.var().index.values
    _df = df[cols_con_varianza].copy()
    selector = VarianceThreshold(threshold=threshold)
    vt = selector.fit(_df)
    _df = _df.loc[:, vt.get_support()]
    return _df


def renombrar_variables(df: pd.DataFrame):
    df.rename(columns={'ganancia_perdida_declarada_bolsa_argentina':'suma_declarada_bolsa_argentina'},inplace=True)
    df['rol_familiar_registrado'].mask((df.rol_familiar_registrado == 'casado' ) | (df.rol_familiar_registrado == 'casada'), 'casado_a', inplace=True)
    df['estado_marital'].mask(df.estado_marital == 'divorciado' , 'divorciado_a', inplace=True)
    df['estado_marital'].mask(df.estado_marital == 'separado', 'separado_a', inplace=True)
    
def solucionar_missings(df: pd.DataFrame):
    df['categoria_de_trabajo'].replace(np.nan,'No contesta', inplace=True)
    df['trabajo'].replace(np.nan,'No contesta', inplace=True)
   
def conversion_tipos(df: pd.DataFrame):
    # Conversión tipos datos para optimización de memoria
        # "category": to more efficiently store the data -> https://pbpython.com/pandas_dtypes_cat.html#:~:text=The%20category%20data%20type%20in,more%20efficiently%20store%20the%20data.
    df = df.astype({
            "trabajo": "category", 
            "categoria_de_trabajo": "category",
            "genero": "category",
            "religion": "category",
            "educacion_alcanzada": "category",
            "estado_marital": "category",
            "rol_familiar_registrado": "category",        
            }) 

        # ubyte: [0, 255) -> https://numpy.org/devdocs/reference/arrays.scalars.html#numpy.ubyte 
    df = df.astype({
            "tiene_alto_valor_adquisitivo": np.ubyte, 
            "edad": np.ubyte,
            "anios_estudiados": np.ubyte,
            "horas_trabajo_registradas": np.ubyte,
            }) 

def obtener_feature_validacion(df: pd.DataFrame):
    y_target = np.array(df[['tiene_alto_valor_adquisitivo']]).ravel()
    df.drop(columns=['tiene_alto_valor_adquisitivo'], inplace=True)
    return y_target



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
 

# MAS FUNCIONES AUXILIARES USADAS

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


def plot_roc_curves_red(clf, XX_test, yy_test, XX_train, yy_train):
    plt.figure(dpi=110)

    # For Test
    fpr_test, tpr_test, _ = roc_curve(yy_test, clf.predict(XX_test))
    roc_auc_test = auc(fpr_test, tpr_test)
    # For Train
    fpr_train, tpr_train, _  = roc_curve(yy_train, clf.predict(XX_train))
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
    
def plot_roc_curves(clf, XX_test, yy_test, XX_train, yy_train):
    plt.figure(dpi=110)

    # For Test
    fpr_test, tpr_test, _ = roc_curve(yy_test, clf.predict_proba(XX_test)[:,1])
    roc_auc_test = auc(fpr_test, tpr_test)
    # For Train
    fpr_train, tpr_train, _  = roc_curve(yy_train, clf.predict_proba(XX_train)[:,1])
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
