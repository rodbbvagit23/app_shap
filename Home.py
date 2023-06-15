# ---- Libraries -----------

import streamlit as st
from PIL import Image
import shap
import os
from pyhere import here

# ---- Page Config -----------


st.set_page_config(layout="wide")
st.set_option("deprecation.showPyplotGlobalUse", False)
#--------------------------------

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css(here('style.css'))

st.write(
    """<style>
    [data-testid="stHorizontalBlock"] {
        align-items: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---- Main Interface -----------

st.markdown(
    "<h1 style='text-align: center; color: black;'>Interpretabilidad Modelos ML 📚 </h1>",
    unsafe_allow_html=True)

        
st.markdown('<div style="text-align: justify;">Bienvenido a la aplicación de Interpretabilidad de ML.\
         El fin de esta app es proporcionar al ususario una herramienta\
         interactiva para visualizar e interpetar\
         las predicciones de modelos de Machine learning mediante\
         el uso del método de Shapley Values. <br> </div>',
         unsafe_allow_html=True)

#--------------------------- Introducción -----------------
st.markdown("<h2 style='text-align: left; color: black;\'>\
            Introducción</h2>",
            unsafe_allow_html=True)
         
st.markdown('<div style="text-align: justify;"> Si esta es la primera vez que utilizas esta aplicación, te recomendamos\
         leer las notas que se encuentran a continuación y revisar la pestaña de \"Ejemplo\", de los contrario,\
         da click en la pestaña "Interpreta" y empieza con la interpretación\
         de tu modelo. En esta hoja encontrarás respuesta a las preguntas\
         ¿Por qué es importante la interpretabilidad en el Machine Learning?\
         ¿Por qué usar SHAP como método explicativo? así como una\
         breve explicación de los gráficos que pueden ser construidos haciendo\
         uso de los SHAP values. En la hoja de Ejemplo podrás encontrar dos ejemplos \
         de cómo interpretar los gráficos generados apartir de los SHAP values y finalmente en la\
         página Interpreta, podrás interpretar cualquier modelo de machine learning que desees. </div>', unsafe_allow_html=True)

image = Image.open(here("otros_insumos", "shap_interp.png"))
st.image(image)

#------------------------ Importancia Interpretabilidad ML ---------------------

st.markdown("<h2 style='text-align: left; color: black;\'>\
            ¿Por qué es tan importante la interpretabilidad \
            en el Machine Learning? 🤔</h2>"\
            ,unsafe_allow_html=True)


st.markdown( "<div style='text-align: justify;'> En años recientes la necesidad\
             de tomar decisiones basadas en datos se ha vuelto cada vez más apremiante, y en ocasiones, \
             la solución a problemas complejos va a compañada del uso de modelos complejos. Desgraciadamente,\
             a diferencia de los modelos lineales, modelos como XGBosst o SGD no son interpretables lo que \
             ocasiona que el usuario sea reactivo al uso de este tipo de modelos. Afortunadamente, hoy en día\
             la barrera entre interpretabilidad  y precisión en los modelos de machine learning se ha ido \
             derrumbando gracias a algunos métodos que nos permiten interpretar\
             una predicción de forma local, esto mediante un modelo auxiliar al que nos referímos cómo 'modelo explicativo' que\
             se trata de cualquier tipo de modelo que sea interpretable, la regresión lineal o un arbol de decisión por dar algunos ejemplos. </div>", unsafe_allow_html=True)
url = "https://arxiv.org/pdf/1705.07874.pdf"
st.markdown("Todos los detalles técnicos de estas notas los puedes consultar el paper [An Unified Approach to Interpreting Model Predictions](%s)." %url)

        
          
# ------------------------ Feature Attribution Methods ------------------
st.markdown("<h2 style='text-align: left; color: black;\'>\
            Feature Attribution Methods 💡</h2>"\
            ,unsafe_allow_html=True)
    
# ----->>>> No se entiende el mensaje que se quiere dar en esta sección
    
st.markdown("<div style='text-align: justify;'>Se ha mencionado en ¿Por qué es tan importante la interpretabilidad\
             en el Machine Learning? la existencia de algunos métodos que tienen cómo fin explicar un modelo de forma \
             local. En particular, aquellos que utilizan cómo modelo explicativo una regresión lineal de variables binarias para explican una predicción\
             son denominados \"Additive feature attribution methods\", los cuáles,\
             dado que sus variables se encuentran en una misma escala, sus respectivos coeficiente son comparables y más aún, son interpretables.</div>", unsafe_allow_html=True)
st.latex("g(z') = \phi_{0} + \sum_{i = 1}^{M} \phi_{i}z'_{i}")
    

    
# ------------------------ Empieza Teoría SHAP values ---------------------------
st.markdown("<h2 style='text-align: left; color: black;\'>\
            Shapley Values 👩‍🏫</h2>",
            unsafe_allow_html=True)
    
st.markdown("<div style='text-align: justify;'>Los shapley values son un resultado\
             derivado de la teoría de juegos.\
             Se plantea un juego de N jugadores que trabajan en equipo para obtener un pago,\
             la pregunta que se plantea es ¿Cómo se cuantifica la contribución de cada jugador? \
             </div>",
             unsafe_allow_html=True)
                 

st.markdown("<div style='text-align: justify;'><br> Podemos extrapolar el probelma de los N jugadores a un modelo de machine learning\
             pensando que los N jugadores son las N variables de nuestro modelo, y el payoff del problema planteado\
             es el output o la predicción, de esta forma en el paper \"An Unified Approach to Interpreting Model Predictions\" se \
             explica cómo, la solución a los coeficientes de los feature attribution methods son un símil a la solución de los pagos que se le asignan\
             a cada jugador en el problema planteado. Los pagos asignados a cada jugador vienen dados por los siguientes coeficientes: </div>" ,
             unsafe_allow_html=True)
    
st.latex(r'''
    \phi_{i}(x) = \sum_{S \subset N\{i\}} \frac{|S|!(n - |S| -1)!}{n!}(v(S \cup \{i\}) - v(S) )
    ''')
st.markdown("<div style='text-align: justify;'> Dónde v  es una función tal que al evaluar un subconjunto S de jugadores en esta\
             nos expresa la suma total esperada de los pagos que se pueden obtener mediante el trabajo colaborativo de los S miembros del equipo.</div>" ,
             unsafe_allow_html=True)

st.markdown("<div style='text-align: justify;'> Algo a destacar del paper mencionado es que menciona que al suponer ciertas condiciones deseables, recordando la definición de los feature attribution methods,\
             tenemos la siguiente igualdad al usar los coeficientes de los SHAP values:</div>" \
            , unsafe_allow_html=True)
    
st.latex(r'''
         \sum_{i = 1}^{M} \phi_{i} = f(x) - E[f(x)]
         ''')
         
st.markdown("<div style='text-align: justify;'> <b>¡En palabras simples!</b> los shapley values nos indican cuántas unidades mueve respecto a la media de la predicción, \
             una variable del modelo. Es impartante mencionar que los shapley values se calculan cuándo el output del modelo es continuo, por lo\
             tanto en caso de trabajarse con un modelo de clasificación, el modelo deberá predecir la probabilidad de pertenecer o no a una categoría, en lugar de\
             la etiqueta.</div>",
             unsafe_allow_html=True)
    
    
#-----------------------Empieza Sección Graficos con SHAP values -----------------

st.markdown("<h2 style='text-align: left; color: black;\'>\
            <br> Gráficos de Interpretabilidad 📊</h2>"\
            ,unsafe_allow_html=True)
    
    
# st.markdown( "<div style='text-align: justify;'>Los siguientes gráficos fueron generados apartir del Data\
#          Set de 'IRIS' y se plantea el problema de clasificar si una\
#          flor es Virginica o Versicolor. En caso de que la probabilidad predicha sea mayor que 0.5 entonces la flor será Virginica,\
#          en caso contrario será Versicolor.</div>", unsafe_allow_html=True)

#--------------------------Empieza Beeswarm plot----------------------------
st.markdown("<h3 style='text-align: center; color: black;\'>\
            Beeswarm Plot </h3>"\
            ,unsafe_allow_html=True)
    
col1, col2 = st.columns(2)

with col1:
    st.markdown( "<div style='text-align: justify;'> En el Beeswarm plot se grafican los SHAP values de una muestra\
            de las instancias del Data Set. \
            El rango de cada variables se\
            aprecia gracias al HeatMap que aparece del lado derecho del gráfico,\
            por lo que, entre más grande sea el valor de una variable para una\
            instancia, el punto del SHAP value de esta será más rojo\
            y lo mismo ocurrirá con los puntos azules.</div>", unsafe_allow_html=True)
    st.markdown( "<div style='text-align: justify;'> Recordando la definición de los SHAP values, interpretamos el hecho\
            de que un punto este a la derecha o a la izquierda del cero cómo\
            que la variable empuja la predicción hacia un valor más bajo o más alto\
            de la media.</div>", unsafe_allow_html=True)
    # st.markdown("<div style='text-align: justify;'> En el contexto del problema que se ha planteado, \
    #              decimos que valores altos de la variable 'petal with' empuja la predicción de probabiliad\
    #              hacia 1, mientras que valores bajos lo empujan hacia el valor 0. De la misma forma,\
    #              a valores bajos de la variable 'sepal with' la predicción es empujada hacia 1.</div>", unsafe_allow_html=True)




with col2:
    beswarmplot_picture = Image.open(here('otros_insumos', "beeswarmplot_example.png"))
    col2.image(beswarmplot_picture,
           use_column_width=True,
           width = .2)

#------------------------ Empieza SHAP Feature Importance --------------------
st.markdown("<h3 style='text-align: center; color: black;\'>\
            Feature Importance Plot </h3>", unsafe_allow_html = True)
            
col1, col2 = st.columns(2)

with col1: 
    st.markdown("<div style='text-align: justify;'> Si afirmamos que una variable es importante para un modelo\
             entonces es de esperarse que esta sea importante para las predicciones de cada una de las instancias\
             (observaciones a predecir), por lo que sacar el promedio de los valores absolutos de los\
             SHAP values para una variable a lo largo de todas las instancias, tiene sentido.</div> ", unsafe_allow_html = True)
    st.latex(r'''
             I_{j} = \frac{1}{n}\sum_{i = 1}^{n}|\phi_{j}^{i}|
             ''')
    # st.markdown("<div style='text-align: justify;'> Con este gráfico observamos que, en promedio,\
    #              la variable de \"Petal Width\" es la más importante para las predicciones de nuestro modelo y la\
    #              menos importante es \"Sepal Width\".</div> ", unsafe_allow_html = True)


with col2:
    featimportance_picture = Image.open(here("otros_insumos", "feature_importance_plot.png"))
    col2.image(featimportance_picture,
           use_column_width=True,
           width = .2)    
    
#----------------------- Empieza Partial Dependence Plot----------------------
st.markdown("<h3 style='text-align: center; color: black;\'>\
            Partial Dependence Plot </h3>", unsafe_allow_html = True)
          
col1, col2 = st.columns(2)

with col1: 
    st.markdown("<div style='text-align: justify;'> Con este gráfico, nos es posible visualizar\
                 cómo se mueve la media del output de nuestro modelo una vez que dejamos fijo el valor de una\
                 variable. En el eje de las \"x\" observamos el rango de la variable en cuestión mientras que\
                 en el eje \"y\" el la media de las predicciones.</div> ", unsafe_allow_html = True)
    # st.markdown("<div style='text-align: justify;'> En este gráfico observamos que, dejando fijas \
    #              todas las variables excepto \"Sepal Length\" el promedio de nuestras predicciones (probabilidades)\
    #              pasan de casi .515 a .485 cuándo la variable \"Sepal Length\" pasa de 4.5 a 5.5, es decir\
    #              disminuye la probabilidad de que la flor sea Virginica.</div> ", unsafe_allow_html = True)
    
    

with col2:
    # featimportance_picture = Image.open(here("Documents", "interpretabilidad_ml", "data", "images", 'partial_dependence_plot.png'))
    featimportance_picture = Image.open(here("otros_insumos", 'partial_dependence_plot.png'))
    col2.image(featimportance_picture,
           use_column_width=True,
           width = .2) 

    
#----------------------- Empieza SHAP Waterfall plot -------------------------
st.markdown("<h3 style='text-align: center; color: black;\'>\
            Waterfall Plot </h3>", unsafe_allow_html = True)
          
col1, col2 = st.columns(2)

with col1: 
    st.markdown("<div style='text-align: justify;'> Este gráfico sirve para observar cómo, partiendo desde la media,\
                cada una de las variables contribuye a una predicción en particular. Recuérdese la identidad: </div> ", unsafe_allow_html = True)
    st.latex(r'''
             \sum_{i = 1}^{M} \phi_{i} = f(x) - E[f(x)]
             ''')
    st.markdown("<div style='text-align: justify;'> Las phi son los SHAP values, son los valores que se encuentran dentro de las barras,\
                 estas están ordenadas de forma descendiente de acuerdo a su valor absoluto. </div>", unsafe_allow_html = True)
    # st.markdown("<div style='text-align: justify;'> Por lo que, partiendo de la media, podemos ver que el hecho de que\
    #              sepal width sea 2.9 no cambia la predicción desd ela media, una vez que agregamos la condición de \
    #              \sepal length= 6.3 la predicción se empuja hacia 1 y así sucesivamente con petal length y petal width.</div> ", unsafe_allow_html = True)


with col2:
    # featimportance_picture = Image.open(here("Documents", "interpretabilidad_ml", "data", "images", 'waterfall_plot.png'))
    featimportance_picture = Image.open(here('otros_insumos', 'waterfall_plot.png'))

    col2.image(featimportance_picture,
           use_column_width=True,
           width = .2)    
    
st.markdown("<div style='text-align: justify;'> En la página \"Ejemplo\" podrás encontrar dos ejemplos sobre \
             cómo interpretar estos gráficos. </div> ", unsafe_allow_html = True)


    
    
    
