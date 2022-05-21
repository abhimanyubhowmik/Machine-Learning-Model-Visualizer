import streamlit as st 
import numpy as np

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
import Logging

log = Logging.app_logging()

class ClassifierModels:

    def __init__(self,clf_name,X,y,test_size):
        self.clf_name = clf_name
        self.X = X
        self.y = y
        self.test_size = test_size

    def classification_parameter_ui(self):
        params = dict()
        if self.clf_name == 'SVM':
            try:
                C = st.sidebar.slider('C', 0.01, 10.0)
                params['C'] = C
                Kernel = st.sidebar.radio('Kernel',('linear','poly','rbf','sigmoid'))
                params['Kernel'] = Kernel
                log.info('Classification parms Successfully Displayed')
            except Exception as e:
                log.error(e)
        elif self.clf_name == 'KNN':
            try:
                K = st.sidebar.slider('K', 1, 15)
                params['K'] = K
                log.info('Classification parms Successfully Displayed')
            except Exception as e:
                log.error(e)
        elif self.clf_name == 'Logistic Regression':
            try:
                solver = st.sidebar.selectbox('Solver',('lbfgs','newton-cg','liblinear','saga'))
                params['Solver'] = solver
                if solver == 'lbfgs' or solver == 'newton-cg':
                    penalty = st.sidebar.radio('Regularization',('l2','none'))
                    params['Penalty'] = penalty
                elif solver == 'liblinear':
                    penalty = st.sidebar.radio('Regularization',('l1','l2','none'))
                    params['Penalty'] = penalty
                else:
                    penalty = st.sidebar.radio('Regularization',('l1','l2','elasticnet','none'))
                    params['Penalty'] = penalty
                C = st.sidebar.slider('C', 0.01, 10.0)
                params['C'] = C
                log.info('Classification parms Successfully Displayed')
            except Exception as e:
                log.error(e)

        else:
            try:
                max_depth = st.sidebar.slider('max_depth', 2, 15)
                params['max_depth'] = max_depth
                n_estimators = st.sidebar.slider('n_estimators', 1, 100)
                params['n_estimators'] = n_estimators
                log.info('Classification parms Successfully Displayed')
            except Exception as e:
                log.error(e)
        return params

    def classification_data_details(self):
        try:
            st.write('Shape of dataset:', self.X.shape)
            st.write('Number of classes:', len(np.unique(self.y)))
            log.info('Dataset Details Successfully Displayed')
        except Exception as e:
            log.error(e)

    def get_classifier(self, params):
        clf = None
        if self.clf_name == 'SVM':
            try:
                clf = SVC(C=params['C'],kernel=params['Kernel'])
                log.info('SVM Model Successfully Created')
            except Exception as e:
                log.error(e)
        elif self.clf_name == 'KNN':
            try:
                clf = KNeighborsClassifier(n_neighbors=params['K'])
                log.info('KNN Model Successfully Created')
            except Exception as e:
                log.error(e)
        elif self.clf_name == 'Logistic Regression':
            try:
                clf = LogisticRegression(penalty=params['Penalty'],C=params['C'],solver = params['Solver'],l1_ratio= 0,max_iter = 50)
                log.info('Logistic Regression Successfully Created')
            except Exception as e:
                log.error(e)
        else:
            try:
                clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
                    max_depth=params['max_depth'], random_state=1234)
                log.info('Rnadom Forest Model Successfully Created')
            except Exception as e:
                log.error(e)
        return clf


    def classification_report(self,clf):
        try:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=1234)

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)

            st.write(f'Classifier = {self.clf_name}')
            st.write(f'Test Size = {self.test_size}')
            st.write(f'Accuracy =', acc)
            st.markdown('''
                    _Accuracy according to complete dataset and test size_
            ''')
            log.info('Classification report Successfully Displayed')
        except Exception as e:
            log.error(e)

    def plot_dataset(self,clf):
        try:
            pca = PCA(2)
            h = 0.02
            X = self.X
            X_projected = pca.fit_transform(X)

            x1 = X_projected[:, 0]
            x2 = X_projected[:, 1]
            x1_new = x1.reshape(len(x1),1)
            x2_new = x2.reshape(len(x2),1)
            X_new = np.concatenate((x1_new, x2_new),axis=1)
            # x1 = X[:,0]
            # x2 = X[:,1]
            # X_new = X[:, :2]
            X_new = StandardScaler().fit_transform(X_new)
            log.info('PCA Successfully generated')

            try:

                x_min, x_max = x1.min() - .5, x1.max() + .5
                y_min, y_max = x2.min() - .5, x2.max() + .5
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
                y_ = np.arange(y_min, y_max, h)

                clf.fit(X_new, self.y)
                log.info('PCA ML Model created successfully')

                try:
                    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)
                    log.info('Model Results successfully predicted')

                    trace1 = go.Heatmap(x=xx[0], y=y_, 
                                        z=Z,
                                        colorscale='Jet',
                                        showscale=True)

                    trace2 = go.Scatter(x=x1, y=x2,
                                        mode='markers',
                                        showlegend=False,
                                        marker=dict(size=10,
                                                    color=self.y, 
                                                    colorscale='Jet',
                                                    reversescale = True,
                                                    line=dict(color='black', width=1))
                                    )

                    layout= go.Layout(
                        autosize= True,
                        xaxis = dict(title = 'Principal Component 1'),
                        yaxis = dict(title = 'Principal Component 2'),
                        title= 'Sample Model Vizualization',
                        hovermode= 'closest',
                        showlegend= False
                    )

                    data = [trace1, trace2]
                    fig = go.Figure(data=data,layout= layout)   
                    log.info('Model figures generated successfully.')


                    st.plotly_chart(fig)

                except Exception as e:
                    log.error(e)
            except Exception as e:
                log.error(e)
        except Exception as e:
            log.error(e)