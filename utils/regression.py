# original script by
# Author: Matt Hall
# Email: matt@agilescientific.com
# License: BSD 3 clause

# modified by
# Achmad Ginanjar

from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import json



class regressions:
    def __init__(self,x_train,y_train,x_val,y_val,x_test,y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        

        self.N = 60
        self.random_state = 13

        self.models = {
            '': dict(),
            'Linear': dict(model=Ridge(), pen='alpha', mi=0, ma=10),
            # 'Polynomial': dict(model=make_pipeline(PolynomialFeatures(2), Ridge()), pen='ridge__alpha', mi=0, ma=10),
            'Huber': dict(model=HuberRegressor(), pen='alpha', mi=0, ma=10),
            'Nearest Neighbours': dict(model=KNeighborsRegressor(), pen='n_neighbors', mi=3, ma=9),
            # 'Linear SVM': dict(model=SVR(kernel='linear'), pen='C', mi=1e6, ma=1),
            'SVR': dict(model=SVR()),
            'Gaussian Process': dict(model=GaussianProcessRegressor(random_state=self.random_state), pen='alpha', mi=1e-12, ma=1),
            'Decision Tree': dict(model=DecisionTreeRegressor(random_state=self.random_state), pen='max_depth', mi=20, ma=3),
            'Random Forest': dict(model=RandomForestRegressor(random_state=self.random_state), pen='max_depth', mi=20, ma=4),
            'Neural Net': dict(model=MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, tol=0.01, random_state=self.random_state), pen='alpha', mi=0, ma=10),    
        }
                        

    def fit(self):
        x_train,y_train = self.x_train, self.y_train
        x_val,y_val = self.x_val, self.y_val
        x_test,y_test = self.x_test, self.y_test
        scores = dict()
        y_std = np.std(y_train)

        for modelname, model in self.models.items():
            if modelname=="":
                continue
            print('Evaluations on', modelname)
            minScore = dict()
            maxScore = dict()

                
            if (m := model.get('model')) is not None:
                
                # xm = np.linspace(-2.5, 2.5).reshape(-1, 1)
                if (pen := model.get('pen')) is not None:
                    m.set_params(**{pen: model['mi']})  # Min regularization.
                m.fit(x_train, y_train)
                ŷm = m.predict(x_train)
                ŷm_ = m.predict(x_val)
                ŷ = m.predict(x_test)

                mscore = np.sqrt(mean_squared_error(y_train, ŷm)) *y_std
                m_score = np.sqrt(mean_squared_error(y_val, ŷm_)) *y_std
                score = np.sqrt(mean_squared_error(y_test, ŷ)) *y_std

                minScore['train'] = float(mscore)
                minScore['val'] = float(m_score)
                minScore['test'] = float(score)
              
            if (pen := model.get('pen')) is not None:
                m.set_params(**{pen: model['ma']})  # Max regularization.
                r = m.fit(x_train, y_train)
                ŷr = r.predict(x_train)
                ŷr_ = r.predict(x_val)

                ŷ = r.predict(x_test)
                mscore = np.sqrt(mean_squared_error(y_train, ŷr)) *y_std
                m_score = np.sqrt(mean_squared_error(y_val, ŷr_)) *y_std
                score = np.sqrt(mean_squared_error(y_test, ŷ))*y_std

                maxScore['train'] = float(mscore)
                maxScore['val'] = float(m_score)
                maxScore['test'] = float(score)
            scores[modelname] = dict()
            scores[modelname]['minScore'] = minScore
            scores[modelname]['maxScore'] = maxScore
            print('Minimum Settings', json.dumps(scores[modelname]['minScore']))
            print('Maximum Settings', json.dumps(scores[modelname]['maxScore']))

        
        return scores


        