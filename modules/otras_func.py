import streamlit as st 
from sklearn.pipeline import Pipeline
import pickle
import mapie 
import webbrowser
from pyhere import here
import os 
import pickle



@st.cache
def ordena(X_train, y):
    Sn = X_train.copy()
    x_cols = X_train.columns
    Sn['target'] = y
    Sn = Sn.sort_index(ascending = True)
    indice = list(Sn.index)
    X_train = Sn[x_cols]
    y = Sn['target']
    return indice, X_train, y

@st.cache
def gridsearch_pipeline(model, X_train, y_train):
    sklearn_model = model.steps[-1][1]
    preprocessing_steps = model.steps[:-1]
    pipe = Pipeline(preprocessing_steps)
    X = pipe.fit_transform(X_train)
    sklearn_model = sklearn_model.fit(X, y_train)
    return sklearn_model, X

@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')

@st.cache
def download_model(model_config):
    pickle_model = pickle.dumps(model_config)
    return pickle_model


def inserta_espacios(x):
    for i in range(0, x):
        st.write('\n')
    
def genera_reporte(x):
    webbrowser.open_new_tab(x)
    
def add_picture_and_text(file, link_imgtemp):
    
    with open(r'otros_insumos\imagenes_temporales\dict_texto.pickle', 'rb') as f:
        my_dict_texto = pickle.load(f)
        
    with open(r'otros_insumos\imagenes_temporales\intro.pickle', 'rb') as f:
        intro = pickle.load(f)
   
    mensaje_part1 = '''<html>
    <head><style>
		.center {
			display: flex;
			flex-direction: column;
			align-items: center;
			justify-content: center;
			text-align: center;
		}
        p {
            font-family: Verdana, sans-serif;
        }
	</style></head>
    '''
    
    mensaje_part1 = mensaje_part1 + f'''
    <body>
    <div class="center">
    <h1>Reporte de Interpretabilidad</h1>
    </div>
    <div style="max-width: 800px; margin: 0 auto; text-align: justify;">
    <p>{intro}</p>
    </div>
    '''
    mensaje_part2 = '''
    </body>
    </html>'''
    name_plots = [i for i in os.listdir(link_imgtemp) if i[-3:] == 'png']
    imagenes = ''
    for i in name_plots:
        imagenes+='<div style="max-width: 800px; margin: 0 auto; text-align: justify;">\n\t'+\
            f'''<h2>{i[:-4]}</h2>'''+\
            f'''<p>{my_dict_texto[i]}</p></div>'''+\
            '''<div class="center">'''+\
            f'''<img src="..\otros_insumos\imagenes_temporales\{i}" alt="{i}" style="page-break-after: always;" height="500px" width="900px">\n'''+\
            '''</div>'''
    mensaje = mensaje_part1 + imagenes + mensaje_part2
    f = open(file,'w')
    f.write(mensaje)
    f.close()
    
def actualiza_diccionario_texto(dictio):
    if len(dictio)>0:
        archivos = os.listdir(r'otros_insumos\imagenes_temporales')
        if 'dict_texto.pickle' not in archivos:
            dictio
            with open(r'otros_insumos\imagenes_temporales\dict_texto.pickle', 'wb') as f:
                pickle.dump(dictio, f)
        else:
            with open(r'otros_insumos\imagenes_temporales\dict_texto.pickle', 'rb') as f:
                old_dictio = pickle.load(f)
            old_dictio.update(dictio)
            with open(r'otros_insumos\imagenes_temporales\dict_texto.pickle', 'wb') as f:
                pickle.dump(old_dictio, f)
    else:
        pass
        
    
