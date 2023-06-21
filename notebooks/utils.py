from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

import pandas as pd

class ScalerDf(BaseEstimator, TransformerMixin):
    def __init__(self, method):
        self.method = method

    def transform(self,X,y=None):
        if self.method == 'minmax':
            scaler = MinMaxScaler()
        elif self.method == 'standard':
            scaler = StandardScaler()
        elif self.method == 'none': # Agregar condición para no hacer nada
            return X
        scaler.fit(X)
        X = pd.DataFrame(
                scaler.transform(X),
                columns=X.columns,
                index = X.index
            )
        return X

    def fit(self, X, y=None):
        return self 
    
def split_data(X, y, test_size=0.2, val_size=0.25, random_state=42):
    # Dividir los datos en entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Dividir los datos de entrenamiento en entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test