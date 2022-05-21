import streamlit as st 
import numpy as np

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
import Logging

log = Logging.app_logging()


class RegressorModels:

    def __init__(self,reg_name,X,y,test_size):
        self.reg_name = reg_name
        self.X = X
        self.y = y
        self.test_size = test_size

    def regression_parameter_ui(self):
        params = dict()
        if self.reg_name == 'Linear Regression':
            log.info('No Parameter UI for linear regression')
            pass
        elif self.reg_name == 'Ridge Regression':
            try:
                solver = st.sidebar.selectbox('Solver',('svd', 'cholesky', 'lsqr', 'sparse_cg', 'saga', 'lbfgs'))
                params['solver'] = solver
                alpha = st.sidebar.slider('Alpha', 0.01, 10.0)
                params['alpha'] = alpha
                log.info('Regression parms Successfully Displayed')
            except Exception as e:
                log.error(e)
        elif self.reg_name == 'Elastic Net Regression':
            try:
                alpha = st.sidebar.slider('Alpha', 0.01, 10.0)
                params['alpha'] = alpha
                l1_ratio = st.sidebar.slider('L1 ratio',0,1)
                params['l1_ratio'] = l1_ratio
                log.info('Regression parms Successfully Displayed')
            except Exception as e:
                log.error(e)

        else:
            try:
                alpha = st.sidebar.slider('Alpha', 0.01, 10.0)
                params['alpha'] = alpha
                max_iter = st.sidebar.slider('Maximum iteration',100,1000)
                params['max_iter'] = max_iter
                log.info('Regression parms Successfully Displayed')
            except Exception as e:
                log.error(e)
        return params

    def regression_data_details(self):
        try:
            st.write('Shape of dataset:', self.X.shape)
            log.info('Dataset Details Successfully Displayed')
        except Exception as e:
            log.error(e)

    def get_regressor(self, params):
        reg = None
        if self.reg_name == 'Linear Regression':
            try:
                reg = LinearRegression()
                log.info('Linear Regression model Successfully Created')
            except Exception as e:
                log.error(e)
        elif self.reg_name == 'Ridge Regression':
            try:
                reg = Ridge(alpha=params['alpha'],solver = params['solver'])
                log.info('Ridge Regression Model Successfully Created')
            except Exception as e:
                log.error(e)
        elif self.reg_name == 'Lasso Regression':
            try:
                reg = Lasso(alpha=params['alpha'],max_iter = params['max_iter'])
                log.info('Lasso Regression Successfully Created')
            except Exception as e:
                log.error(e)
        else:
            try:
                reg = ElasticNet(alpha=params['alpha'], l1_ratio = params['l1_ratio'])
                log.info('Elastic Net Regression Successfully Created')
            except Exception as e:
                log.error(e)
        return reg


    def regression_report(self,reg):
        try:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=1234)

            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)

            r2 = r2_score(y_test, y_pred)

            st.write(f'Classifier = {self.reg_name}')
            st.write(f'Test Size = {self.test_size}')
            st.write(f'Mean Squered Error = {mse}') 
            st.write(f'R2 Score = {r2}')
            st.markdown('''
                    _Accuracy according to complete dataset and test size_
            ''')
            log.info('Regression report Successfully Displayed')
        except Exception as e:
            log.error(e)

    def plot_dataset(self,reg):
        try:
            pca = PCA()
            X = self.X
            X_projected = pca.fit_transform(X)

            X = X_projected[:, 0]
            X_2d = X.reshape(len(X),1)
            log.info('PCA Successfully generated')

            try:

                reg.fit(X_2d, self.y)
                log.info('PCA ML Model created successfully')

                try:
                    y_pred = reg.predict(X_2d)
                    log.info('Model Results successfully predicted')

                    trace1 = go.Scatter(x = X, y = y_pred, mode = 'lines')

                    trace2 = go.Scatter(x=X, y=self.y,
                                        mode='markers',
                                        showlegend=False,
                                        marker=dict(size=7,
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

                    data = [trace2, trace1]
                    fig = go.Figure(data=data,layout= layout)   
                    log.info('Model figures generated successfully.')


                    st.plotly_chart(fig)

                except Exception as e:
                    log.error(e)
            except Exception as e:
                log.error(e)
        except Exception as e:
            log.error(e)

