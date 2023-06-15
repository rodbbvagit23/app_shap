# ---- Libraries ---------

import streamlit as st
import pandas as pd
import pickle
from PIL import Image
from pyhere import here
from modules.otras_func import *
import mapie
# showWarningOnDirectExecution = false
# import warnings
# warnings.filterwarnings("ignore")

# ---- Ejemplos -------

st.set_page_config(layout="wide")
st.set_option("deprecation.showPyplotGlobalUse", False)
st.write(
    """<style>
    [data-testid="stHorizontalBlock"] {
        align-items: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css(here('style.css'))


st.markdown(
    "<h1 style='text-align: center; color: black;'>Ejemplos 🪐 </h1>",
    unsafe_allow_html=True)

st.markdown("<div style='text-align: justify;'> Para ejemplificar la interpretación de los gráficos generados \
             apartir de los shap values proponemos dos ejemplos, para una tarea de clasificación usaremos \
             el bien conocido dataset de Iris de Fisher pero en esta ocasión será filtrado solamente para dos tipos  \
             de flores virginica y versicolor\
             . Para el problema de regresión usaremos el data set de Diabetes de Sklearn el cuál posee \
             un registro de indicadores médicos que se le han hecho a pacientes con diabetes y a los cuáles un año después \
             les fue medido de forma cuantitativa el progreso de su enfermedad.</div> ", unsafe_allow_html = True)


#--------------------------- Modelo de Clasificación -----------------
st.markdown("<h2 style='text-align: left; color: black;\'>\
            Modelo de Clasificación</h2>",
            unsafe_allow_html=True)
    
st.markdown("<div style='text-align: justify;'>Para la construcción del data set de Iris\
             se observaron cuatro especies diferentes de Iris y se midieron cuatro rasgos diferentes de ellas \
             petal length, petal with, sepal length y sepal with;  \
             por lo que contamos con cuatro variables que nos ayudan a identificar el tipo de Iris que corresponde\
             a cada observación del DataSet. Cómo se comentó en la introducción, nos limitaremos a filtrar el data\
             set por las Iris Virginica y Versicolor de tal forma que esto se trate de un problema de clasificación binaria. \
             Para poder construir los modelos, virgínica ha sido etiquetada cómo 1 y versicolor cómo 0. Y sólo queda señalar que los gráficos \
             están basados/interpretan en la probabilidad del evento y entonces, la probabilidad de que la flor sea etiquetada cómo Virgínica.    \
             </div> ", unsafe_allow_html = True)

st.markdown("<div style='text-align: justify;'>Para este problema se entrenó un modelo KNN con 5 vecinos como\
             hiperparámetro y los demás hiperparámetros estandar del modelo KNN de sklearn.  </div> ", unsafe_allow_html = True)
             
st.markdown("<h3 style='text-align: center;'> Beeswarm Plot </h3> ", unsafe_allow_html = True)

beeswarm_picture = Image.open(here('otros_insumos', "Beeswarmplot-Iris.png"))
col1, col2, col3 = st.columns([.2, 1, .2])
col2.image(beeswarm_picture,
       use_column_width=True,
       width = 1)
st.markdown("<div style='text-align: justify;'> Podemos darle la siguiente interpretación al gráfico.</div> ", unsafe_allow_html = True)
st.markdown(
"""
- las observaciones que se encuentran en el círculo amarillo poseen un tono azul, esto representa valores\
          bajos de la variable petal length dentro de su rango y el hecho de que estén del lado derecho del 0 \
          significa que estos SHAP values son positivos lo que quiere decir que afectan a nuestro modelo de forma positiva\
          es decir que mueven la media de la predicción (en este caso la probabilidad de que sea Virginica) hacia 1.
- las observaciones que se encuentran en el círculo verde poseen un tono rojizo, esto representa valores\
          altos de la variable petal length dentro de su rango y el hecho de que estén del lado izquierdo del 0 \
          significa que estos SHAP values son negativos lo que quiere decir que afectan a nuestro modelo de forma negativa\
          es decir que mueven la media de la predicción (en este caso la probabilidad de que sea Virginica) hacia 0.
- las observaciones que se encuentran en el círculo anaranjado poseen un tono azul, esto representa valores\
          bajos de la variable sepal with dentro de su rango y el hecho de que estén del lado izquierdo del 0 \
          significa que estos SHAP values son negativos lo que quiere decir que afectan a nuestro modelo de forma negativa\
          es decir que mueven la media de la predicción (en este caso la probabilidad de que sea Virginica) hacia 0.
- las observaciones que se encuentran en el círculo azul poseen un tono rojizo, esto representa valores\
          altos de la variable sepal with dentro de su rango y el hecho de que estén del lado derecho del 0 \
          significa que estos SHAP values son negativos lo que quiere decir que afectan a nuestro modelo de forma positiva\
          es decir que mueven la media de la predicción (en este caso la probabilidad de que sea Virginica) hacia 1.
"""
)
#=========================================================================================
st.markdown("<h3 style='text-align: center;'> Bar Plot </h3> ", unsafe_allow_html = True)
beeswarm_picture = Image.open(here('otros_insumos', "Barplot-Iris.png"))
col1, col2, col3 = st.columns([.2, 1, .2])
col2.image(beeswarm_picture,
       use_column_width=True,
       width = 1)
st.markdown("<div style='text-align: justify;'> Podemos darle la siguiente interpretación al gráfico.</div> ", unsafe_allow_html = True)
st.markdown(
"""
- La variable más importante influyente en nuestro modelo es petal length.
- La variable menos influyente en nuestro modelo es sepal with.
"""    
)
#=======================================================================================
st.markdown("<h3 style='text-align: center;'> Waterfall plot </h3> ", unsafe_allow_html = True)
beeswarm_picture = Image.open(here('otros_insumos', "Waterfallplot-Iris.png"))
col1, col2, col3 = st.columns([.2, 1, .2])
col2.image(beeswarm_picture,
       use_column_width=True,
       width = 1)
st.markdown("<div style='text-align: justify;'> Podemos darle la siguiente interpretación al gráfico.</div> ", unsafe_allow_html = True)
st.markdown(
"""
- El valor medio de la predicción, en este caso de la probabilidad, de que la flor sea Virginica de todas las observaciones del data set 
  es de 0.523 (el valor en morado).
- La variable más influyente para la predicción de esta observación es la variable petal length (el valor en naranja) dado
  el largo de su barrar la cuál tiene una longitud de 0.4131 unidades (este número es el SHAP value), esta barra
  se encuentra en color rojo lo que indica que impacta al modelo de forma positiva y entonces 
  mueven la media de la predicción (en este caso la probabilidad de que sea Virginica) hacia 1.
- La variable menos influyente para la predicción de esta observación es la variable sepal with (el valor en verde) dado
  el largo de su barra la cuál tiene una longitud de 0.0007 unidades (este número es el SHAP value), esta barra
  se encuentra en color azul lo que indica que impacta al modelo de forma negativa y entonces 
  mueven la media de la predicción (en este caso la probabilidad de que sea Virginica) hacia 0.
- La suma de todos los SHAP values, que son los valores dentro o aun costado de las barras, más el valor medio de la predicción\
  es igual a la probabilidad predicha para esta observación, es decir, 0.8 y que tomando cómo threshold para la clasificación 0.5 entonces \
  esta flor sería clasificada cómo Virgínica.
"""    
)


#--------------------------- Modelo de Regresión ---------------------
st.markdown("<h2 style='text-align: left; color: black;\'>\
            Modelo de Regresión</h2>",
            unsafe_allow_html=True)
    
st.markdown("<div style='text-align: justify;'>El data set de Diabetes de Sklearn cuenta con \
             un registro del progreso de enfermedad de pacientes enfermos de diabetes un año después de \
             tomarles una serie de indicadores médicos, estos últimos mencionados serán nuestros 10 regresores \
             y nuestra variable target para la tarea de regresión será el progreso de la enfermedad. \
             </div> ", unsafe_allow_html = True)
             
st.markdown("<h3 style='text-align: center;'> Beeswarm Plot </h3> ", unsafe_allow_html = True)

beeswarm_picture = Image.open(here('otros_insumos', "Beeswarmplot-Diabetes.png"))
col1, col2, col3 = st.columns([.2, 1, .2])
col2.image(beeswarm_picture,
       use_column_width=True,
       width = 1)

st.markdown("<div style='text-align: justify;'> Podemos darle la siguiente interpretación al gráfico.</div> ", unsafe_allow_html = True)
st.markdown(
"""
- las observaciones que se encuentran en el recuadro verde poseen un tono rojizo, esto representa valores\
          altos de la variable s5 dentro de su rango y el hecho de que estén del lado derecho del 0 \
          significa que estos SHAP values son positivos lo que quiere decir que afectan a nuestro modelo de forma positiva\
          es decir que mueven la media de la predicción (en este caso el progreso de la enfermedad) hacia un valor más alto.
          En resumen, a mayor valor de la variable s5 la predicción aumenta.
- las observaciones que se encuentran en el recuadro naranja poseen un tono azulado, esto representa valores\
          bajos de la variable s5 dentro de su rango y el hecho de que estén del lado izquierdo del 0 \
          significa que estos SHAP values son negativos lo que quiere decir que afectan a nuestro modelo de forma negativa\
          es decir que mueven la media de la predicción (en este caso el progreso de la enfermedad) hacia un valor más bajo.
          En resumen, a menos valor de la variable s5 la predicción disminuye.
- Observando los SHAP values mostrados en el recuadro morado notamos que estos tienden a presentar valores muy cercanos a cero,
          esto significa que en general esta variable no es muy influyente en comparación a las otras variables, hecho que 
          también debería poder ser apreciable en el Bar plot que sigue. 
"""
)

#=========================================================================================
st.markdown("<h3 style='text-align: center;'> Bar Plot </h3> ", unsafe_allow_html = True)
beeswarm_picture = Image.open(here('otros_insumos', "Barplot-Diabetes.png"))
col1, col2, col3 = st.columns([.2, 1, .2])
col2.image(beeswarm_picture,
       use_column_width=True,
       width = 1)
st.markdown("<div style='text-align: justify;'> Podemos darle la siguiente interpretación al gráfico.</div> ", unsafe_allow_html = True)
st.markdown(
"""
- La variable más influyente en nuestro modelo es \"s5\".
- La variable menos influyente en nuestro modelo es \"sex\".
"""    
)

#========================================================================================
st.markdown("<h3 style='text-align: center;'> Waterfall plot </h3> ", unsafe_allow_html = True)
beeswarm_picture = Image.open(here('otros_insumos', "Waterfallplot-Diabetes.png"))
col1, col2, col3 = st.columns([.2, 1, .2])
col2.image(beeswarm_picture,
       use_column_width=True,
       width = 1)
st.markdown("<div style='text-align: justify;'> Podemos darle la siguiente interpretación al gráfico.</div> ", unsafe_allow_html = True)
st.markdown(
"""
- El valor medio de la predicción, en este caso de la progresión de la enfermedad, sobre todas las observaciones del data set 
  es de 132.729 (el valor en el recuadro morado).
- La variable más influyente para la predicción de esta observación es la variable pb (el valor subrayado en naranja) dado
  el largo de su barrar la cuál tiene una longitud de 24.0242 unidades (este número es el SHAP value), esta barra
  se encuentra en color azul lo que indica que impacta al modelo de forma negativa y entonces 
  mueven la media de la predicción (el progreso de la enfermedad) hacia abajo.
- La segunda variable más influyente para la predicción de esta observación es la variable bmi (el valor subrayado en verde) dado
  el largo de su barra la cuál tiene una longitud de 10.0416 unidades (este número es el SHAP value), esta barra
  se encuentra en color rojo lo que indica que impacta al modelo de forma positiva y entonces 
  mueve la media de la predicción (el progreso de la enfermedad) hacia arriba.
- La suma de todos los SHAP values, que son los valores dentro o aun costado de las barras, más el valor medio de la predicción\
  es igual a la probabilidad predicha para esta observación, es decir, 102.41 (el valor en el recuadro rosa).
"""    
)

#--------------------------- Descarga de Archivos --------------------
st.markdown("<h2 style='text-align: left; color: black;\'>\
            Es tu turno! </h2>",
            unsafe_allow_html=True)
st.markdown("<div style='text-align: justify;'> Así como lo viste en el video de la pestaña de Home\
             puedes descargar los data sets usados para generar los gráficos y cargarlos en la pestaña\
             \"Interpreta\" de la aplicación.</div> ", unsafe_allow_html = True)

Xtrain_iris = pd.read_csv(here('otros_insumos', "Xtrain_iris.csv"))
ytrain_iris = pd.read_csv(here('otros_insumos', "ytrain_iris.csv"))
Xtrain_diabetes = pd.read_csv(here('otros_insumos', "Xtrain_diabetes.csv"))
ytrain_diabetes = pd.read_csv(here('otros_insumos', "ytrain_diabetes.csv"))

col1, col2, col3 = st.columns(3)
select_task = False
    


with st.expander("Download Files"):
    select_task = st.radio("¿Se trata de una tarea de Clasificación o de Regresión?",
                 ('Clasificacion', 'Regresion'), horizontal = True)
    select_task = select_task.lower()
    
    t_data, t_models = st.tabs(['Data', 'Models']) # Tabs
    
    with t_data:
        
        if select_task == 'regresion':
            # Descripción
            
            st.write(
                "Se seleccionó interpretar un problema de regresión, para esto, se utiliza el"
                " dataset de ..."
            )
            
            st.write(
                ""
            )
            
        else:
            # Descripción
            
            st.write(
                "Se seleccionó interpretar un problema de clasificación, para esto, se utiliza el"
                " dataset de Iris."
            )
            
            st.write(
                ""
            )
            
        # Data construction 
        X, y =  [Xtrain_diabetes, ytrain_diabetes] if select_task == 'regresion' else [Xtrain_iris, ytrain_iris]
        
        
        # X_train, X_test, y_train, y_test = itp.train_test_split(X,
        #                                                         y,
        #                                                         test_size = 0.20,
        #                                                         random_state = 222023)
        
        # complete_data = pd.concat([X, y], 
        #                           axis = 1)
        
        csv_x_train = convert_df(X)
        csv_y_train = convert_df(y)
        
        # Display data and download button
    
        # st.dataframe(complete_data.head())
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(X.head(), 600)
        with col2:
            st.dataframe(y.head(), 600)
        
        _, c_X_download, c_Y_download, _ = st.columns(4)
        
        if select_task == 'regresion':
            name1 = 'Xtrain_Diabetes.csv'
            name2 = 'ytrain_Diabetes.csv'
        else:
            name1 = 'Xtrain_Iris.csv'
            name2 = 'ytrain_Iris.csv'
        
        
        with c_X_download:
            
            st.download_button('Download X train', csv_x_train, name1)
            
        with c_Y_download:
            
            st.download_button('Download y train', csv_y_train, name2)
    with t_models:
        
        st.write(
            "A continuación, se muestran una lista de modelos que pueden ser entrenados para poder"
            " generar un pronóstico de nuestra variable objetivo. Es importante aclarar que no se realiza"
            " ningún procedimiento para la obtención de hiperparámetros."
        )
        
        # Select model to train
        model_options_r = ['RandomForestRegressor']
        model_options_c = ['KNN']
        
        selected_model = st.selectbox(label = 'Seleccione Modelo a Entrenar:',
                                      options = model_options_r if select_task == 'regresion' else model_options_c,
                                      index = 0)
        
        # Training model
        dict_models_regressor = {'RandomForestRegressor':pickle.load(open(here('otros_insumos', "RandomForestRegressor-diabetes.pkl"), 'rb'))}
        dict_models_classification = {'KNN': pickle.load(open(here('otros_insumos', "KNN-Iris.pkl"), 'rb'))}
        
        mapped_selected_model = dict_models_regressor[selected_model] if select_task == 'regresion' else dict_models_classification[selected_model]
        
        # model_trained_conf, X_train, y_train = itp.train_model(mapped_selected_model, X, y)
        
        _, c_model_download, _ = st.columns(3)
        
        with c_model_download:
            
            st.download_button('Download Model',
                               data = download_model(mapped_selected_model),
                               file_name = 'model_config.sav')
     


