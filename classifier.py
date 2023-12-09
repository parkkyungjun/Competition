from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
# from sklearn.ensemble import VotingClassifier
import pickle

class model():
  def __init__(self, model, param):
    if model == 'xgb':
      self.model = XGBClassifier(**param)
    elif model == 'lgb':
      self.model = LGBMClassifier(**param)
    elif model == 'cat':
      self.model = CatBoostClassifier(**param)
    self.mod = model
    self.param = param

  def fit(self, X_train, y_train, param, save=False, name='model'):
    self.model.fit(X_train, y_train, **param)
    if save:
      with open(f'{name}.pkl', 'wb') as name:
        pickle.dump(self.model, name)

  def fold(self, X, y, n):
    kfold = KFold(n_splits=n, shuffle=True, random_state=42)
    scores = []
    
    for train_index, test_index in kfold.split(X):
      if isinstance(X, pd.DataFrame):
        X_train, X_test = X.loc[train_index], X.loc[test_index] 
        y_train, y_test = y.loc[train_index], y.loc[test_index]
      elif isinstance(X, np.array):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

      model = model(self.mod, param)
      model.fit(X_train, y_train)
    
  def predict(self, X_test):
    return self.model.predict(X_test)
    
  def predict_proba(self, X_test):
    return self.model.predict_proba(X_test)[:, 1]
