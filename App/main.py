import streamlit as st 
from PIL import Image

from sklearn import datasets
from sklearn.datasets import make_blobs
import Classification as cl
import Regression as rg
import Logging

log = Logging.app_logging()

im = Image.open("favicon.png")
st.set_page_config(
page_title="ML Model Visualizer",
page_icon=im,
layout="wide",
initial_sidebar_state="expanded",
menu_items={
        "Get Help": "https://github.com/abhimanyubhowmik",
        "Report a bug": "https://linkedin.com/in/bhowmikabhimanyu",
        "About": "### A Machine Learning Models Visualisation Application \n :copyright: Abhimayu Bhowmik \n\n Github: https://github.com/abhimanyubhowmik \n\n Linkedin: https://linkedin.com/in/bhowmikabhimanyu ",
    }
)

st.title('Machine Learning Model Vizualizer')
problem_name = st.sidebar.selectbox('Select Problem', ('Classification', 'Regression'))

def get_problem(problem_name):
    if problem_name == 'Classification':
        dataset_name = st.sidebar.selectbox('Select Dataset', ('Iris', 'Random'))
        classifier_name = st.sidebar.selectbox('Select classifier',('Logistic Regression', 'KNN', 'SVM', 'Random Forest'))
        test_size = st.sidebar.slider('Test Size',min_value= 10,max_value= 90)
        return classifier_name,dataset_name,test_size
    else:
        dataset_name = st.sidebar.selectbox('Select Dataset', ('House price', 'Diabetes'))
        regressor_name = st.sidebar.selectbox('Select Regressor',('Linear Regression', 'Ridge Regression','Lasso Regression', 'Elastic Net Regression'))
        test_size = st.sidebar.slider('Test Size',min_value= 10,max_value= 90)
        return regressor_name,dataset_name,test_size


def get_dataset(name):
    try:
        data = None
        if name == 'Iris':
            data = datasets.load_iris()
            log.info('Iris dataset loaded')
        elif name == 'Random':
            data = make_blobs()
            log.info('Random dataset loaded')
            X, y = data
            return X, y
        elif name == 'House price':
            data = datasets.load_boston()
            log.info('Boston House price dataset loaded')
        elif name == 'Diabetes':
            data = datasets.load_diabetes()
            log.info('Diabetes dataset loaded')
        else:
            log.error('Wrong Dataset')

        X = data.data
        y = data.target
        log.info('Dataset splitted into targer and data.')
        return X, y
    except Exception as e:
        log.error(e)

def prediction(problem, model_name, X, y,test_size):
    if problem == 'Classification':
        classification  = cl.ClassifierModels(model_name, X, y,test_size)
        classification.classification_data_details()
        parms = classification.classification_parameter_ui()
        clf = classification.get_classifier(parms)
        classification.classification_report(clf)
        classification.plot_dataset(clf)
    else:
        regression = rg.RegressorModels(model_name, X, y,test_size)
        regression.regression_data_details()
        parms = regression.regression_parameter_ui()
        reg = regression.get_regressor(parms)
        regression.regression_report(reg)
        regression.plot_dataset(reg)
    


if __name__ == "__main__":
    model_name,dataset_name,test_size = get_problem(problem_name)
    X,y = get_dataset(dataset_name)
    prediction(problem_name,model_name,X,y,test_size)

