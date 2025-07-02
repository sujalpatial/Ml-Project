import os 
import sys
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
import pickle

def load_object(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)

def save_object(file_path,obj):
    try:
        dir_path =os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open (file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(x_train,y_train,x_test,y_test,models,params):
    try:
        report={}
        
        for model_name,model in models.items():
            param_grid=params.get(model_name, {})
        if param_grid:
            gs=GridSearchCV(model,param_grid,cv=3,n_jobs=-1,scoring='r2',verbose=0)
            gs.fit(x_train,y_train)
            best_model=gs.best_estimator_
        else:
            model.fit(x_train,y_train)
            
            y_test_pred=best_model.predict(x_test)
            test_model_score=r2_score(y_test,y_test_pred)

            report[model_name] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e,sys)
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e  # ✅ correctly raises the exception
