import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score
from sys import argv

import mlflow
import mlflow.sklearn

import os



#df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../wine_data.csv"))
#print(df.shape)

wine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wine_data.csv")


feature_names=['Type','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols'
          ,'Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue'
          ,'OD280/OD315 of diluted wines','Proline']

data = pd.read_csv(wine_path,names=feature_names)




y=data['Type']
del data['Type']
data_train, data_test, class_train, class_test = train_test_split(data, y, test_size=0.3)



num_tree=int(argv[1]) if len(argv)>1 else 10
max_depth=int(argv[2]) if len(argv)>2 else 3

with mlflow.start_run():
    rf=RandomForestClassifier(n_estimators=num_tree,max_depth=max_depth)
    rf.fit(data_train,class_train)
    pred=rf.predict(data_test)
    accuracy=accuracy_score(class_test,pred)
    
    print("max_depth", max_depth)
    print("num_tree", num_tree)
    print("accuracy", accuracy)
    print(rf, "model")
    
    
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("num_tree", num_tree)

    mlflow.log_metric("accuracy", accuracy)
  
    mlflow.sklearn.log_model(rf, "model")
    