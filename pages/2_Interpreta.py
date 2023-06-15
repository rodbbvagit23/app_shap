#--------------------------Dependencias------------------------
import streamlit as st 
import pandas as pd
import shap
import pickle
from modules.otras_func import * 
from modules.interpretabilidad import *
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from pyhere import here
import webbrowser
import mapie



st.set_page_config(layout="wide")
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        
local_css(here('style.css'))
# ---------------------------------------------------------------------------------------

if ('button_clicked_1'not in st.session_state)or ('button_clicked_2'not in st.session_state)\
    or ('button_clicked_3' not in st.session_state):
    st.session_state.button_clicked_1 = False
    st.session_state.button_clicked_2 = False
    st.session_state.button_clicked_3 = False
    

def callback_1():
    st.session_state.button_clicked_1 = True
def callback_2():
    st.session_state.button_clicked_2 = True
def callback_3():
    st.session_state.button_clicked_3 = True
    
# ---------------------------------------------------------------------------------------

st.markdown(
    "<h1 style='text-align: center; color: black;'>Interpreta 🚀</h1>",
    unsafe_allow_html=True,
)

#------------------------------- Explicación -----------------------
st.write('En el siguiente desplegable deberás cargar un csv separado por comas de\
         tu conjunto de validación o entrenamiento, también deberás cargar un csv con tus valores target, y finalmente\
          el modelo en formato pickle con el que entrenaste. Una vez realizado lo anterior,\
         de click en el botón "Cargar Archivos"')

#------------------ Carga de modelo, csv ---------------
with st.expander("Upload Files"):
    x_column, y_column, model_column = st.columns(3)
    with x_column:
        uploaded_file = st.file_uploader("Cargue su matriz de predictoras (train o valid set)")
    with y_column:
        target = st.file_uploader('Cargue sus valores target')
        
    with model_column:
        modelo = st.file_uploader("Cargue su modelo")
    with st.form('confirm_data'):
        
        _, _, c_complete,_, _ = st.columns(5)
        with c_complete:
            task = st.radio("¿Se trata de una tarea de Clasificación o de Regresión?",
        ('Clasificacion', 'Regresion'), horizontal = True)
            submit_button_upload =  st.form_submit_button(label = 'Cargar Archivos', on_click=callback_1)
  

if (submit_button_upload) or (st.session_state.button_clicked_1):
#------------------------- Ejecuta el Explainer ---------------------

    if (uploaded_file is not None) &\
            (modelo is not None) &\
                (target is not None):
        X_train = pd.read_csv(uploaded_file)
        y = pd.read_csv(target)
        modelo = pickle.load(modelo)
        shap.initjs()
        
        st.markdown("<h3 style='text-align: center; color: black;\'>\
                    Tablas Cargadas </h3>", unsafe_allow_html = True)
        
        st.markdown("<div style='text-align: justify;'> A continuación, una muestra de la matriz de predictoras\
                    y su respectiva tabla con los valores target que nos ha proporcinado. De favor a continuación, indique los nombres\
                    de las columnas que poseen los índices de las tablas (si es que poseen una), de lo contrario, seleccione la opción \"Ninguna\".\
                    De la misma forma, indique\
                    el nombre de la columna en la que se encuentra la variable target. Una vez hecho esto, seleccione el botón \"Interpreta\" </div> ", unsafe_allow_html = True)
                    
        col1, col2, col3 = st.columns([.8, .8, .5])

        with col1: 
            st.markdown(
                "<h6 style='text-align: center; color: black;'>Matriz de Predictoras</h6>",
                unsafe_allow_html=True,
            )
            st.dataframe(X_train.head(10), 600)
            
        with col2:
            st.markdown(
                "<h6 style='text-align: center; color: black;'>Matriz con valores Target</h6>",
                unsafe_allow_html=True,
            )
            st.dataframe(y.head(10), 600) 
        
        
# Escoge las columna target y los índices
        with col3:
            inserta_espacios(7)
            select_index = st.selectbox(
                label ="Escoge el índice de las tablas:",
                options = list(X_train.columns) + ['Ninguna'],
                help="la columna que es el índice del data frame",
            )
            
            select_vart_target = st.selectbox(
                label ="Escoge la columna que contiene la target:",
                options = list(y.columns),
                help="Escoge la columna que contiene la target:",
            )
        
       
        _, _, c_interpreta, _, _ = st.columns(5)
        with c_interpreta:
            interpreta_button =  st.button(label = 'Interpreta', on_click=callback_2)
        
        
        
        if (interpreta_button)or (st.session_state.button_clicked_2):
            
            if select_index != 'Ninguna':
                X_train.set_index(select_index, inplace = True)
                y.set_index(select_index, inplace = True)
            
            y = y[select_vart_target]
            
            indice, X_train, y = ordena(X_train, y)
                
            if type(modelo) == GridSearchCV:
                modelo = modelo.best_estimator_
                if type(modelo) == Pipeline:
                    modelo, X_train = gridsearch_pipeline(modelo, X_train, y)
            #==================================================================
            st.markdown("<h3 style='text-align: center; color: black;\'>\
                        Introduction </h3>", unsafe_allow_html = True)
            col1, col2 = st.columns([.75, .25])
            with col1:
                intro = st.text_area("Enter some text")
            with col2:
                inserta_espacios(4)
                boton_intro = st.button('Guardar una intro')
                if boton_intro:
                    with open(r'otros_insumos\imagenes_temporales\intro.pickle', 'wb') as f:
                        pickle.dump(intro, f)
            
            #==================================================================
            st.markdown("<h3 style='text-align: center; color: black;\'>\
                        SHAP Plots </h3>", unsafe_allow_html = True)
            if task == 'Regresion':
                general_explainer_regression(modelo.predict, X_train, y, indice)
            else:
                general_explainer_clasif(modelo, X_train, y, indice)
                # with open(r'otros_insumos\imagenes_temporales\dict_texto.pickle', 'rb') as f:
                #     my_dict_texto = pickle.load(f)
                # st.write('aqui el diccionario')
                # st.write(my_dict_texto)
             
            
            _, _, col, _, _ = st.columns(5)
            with col:
                st.download_button('Download CSV', here('pages','reporte.html'))
#                 gen_reporte = st.button(label = 'Genera Reporte')
#                 if (gen_reporte):
#                     add_picture_and_text(here('pages','reporte.html'), here('otros_insumos','imagenes_temporales'))
#                     genera_reporte(here('pages','reporte.html')) 
        
