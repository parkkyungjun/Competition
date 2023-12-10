from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from skopt import BayesSearchCV
import optuna

# from skopt.space import Real, Integer
# from scipy.stats import randint as sp_randint
# from scipy.stats import uniform as sp_uniform
# stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
"""
lgbm
https://lightgbm.readthedocs.io/en/latest/Parameters.html
param = { num_iterations
    'max_depth': sp_randint(15, 30), # -1, x>=0
    'colsample_bytree': sp_uniform(0.5, 0.5), # 1, 0~1
    'colsample_bynode': sp_uniform(0.5, 0.5), # 1, 0~1
    'reg_alpha' : sp_uniform(0, 1), # 0, x>=0 -> l1
    'reg_lambda' : sp_uniform(0, 1), # 0, x>=0 -> l2
    'min_child_samples' : sp_randint(10, 150), # 20, x>=0
    'max_bin': sp_randint(150, 1500), # 255, x>=1
}

xgb
https://xgboost.readthedocs.io/en/stable/parameter.html
param = { n_estimators, 
    'max_depth': trial.suggest_int('max_depth', 3, 10), # 6, x>=0
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0), # 1, 0~1
    'reg_alpha': trial.suggest_float('alpha', 0, 1), # 0, x>=0
    'reg_lambda': trial.suggest_float('lambda', 1, 10), # 1, x>=0
    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10), # 1, x>=0
}

cat
https://catboost.ai/en/docs/references/training-parameters/
param = { n_estimators
    'max_depth': trial.suggest_int('max_depth', 3, 10), # 6, x>=0
    'reg_lambda': trial.suggest_float('lambda', 1, 10), # 3, x>=0
    'min_child_samples': trial.suggest_int('min_child_samples', 1, 10), # 1 x>=0
}
"""
def log_best(serch):
  print("Best parameters found: ", serch.best_params_)
  print("Best score: ", serch.best_score_)
  
def random_serch(X_train, y_train, model, param, serch_param):
  """
  param = { 
      'max_depth': sp_randint(15, 30), -> 15~30
      'colsample_bytree': sp_uniform(0.5, 0.5), -> 0.5 + (0 ~ 0.5)
  }
  random_search = RandomizedSearchCV(lgb, param_distributions=param, n_iter=100, cv=3, random_state=42, n_jobs=-1, scoring='roc_auc')
  """
  random_search = RandomizedSearchCV(model, param_distributions=param, **serch_param)
  random_search.fit(X_train, y_train)
  log_best(random_search)


def grid_serch(X_train, y_train, model, param, serch_param):
  """
  param = {
    'max_depth': [15, 20, 25, 30],
    'colsample_bytree': [0.5, 0.7, 0.9, 1.0],
  }
  GridSearchCV(lgb, param_grid=param, cv=stratified_kfold or 5, n_jobs=-1, scoring='roc_auc')
  """
  grid_search = GridSearchCV(model, param_grid=param, **serch_param) # cv=stratified_kfold, n_jobs=-1, scoring='roc_auc'
  grid_search.fit(X_train, y_train)
  log_best(grid_search)

def bayes_search(X_train, y_train, model, param, serch_param):
  """
  param = {
    'max_depth': Integer(15, 30),
    'colsample_bytree': Real(0.5, 1.0),
  }
  bayes_search = BayesSearchCV(lgb, param, n_iter=32, cv=stratified_kfold or 5, n_jobs=-1, scoring='roc_auc')
  """
  bayes_search = BayesSearchCV(model, param, **serch_param) # n_iter=32, cv=stratified_kfold, n_jobs=-1, scoring='roc_auc'
  bayes_search.fit(X_train, y_train)
  log_best(bayes_search)

def optuna(objective, direction, n_trials):
  """
  def objective(trial):
      param = {
          'max_depth': trial.suggest_int('max_depth', 15, 30),
          'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
      }
  
      lgbm = LGBMClassifier(**param)
      stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
      scores = cross_val_score(lgbm, X, y, cv=stratified_kfold, scoring='roc_auc')
  
      return scores.mean()
  """

  
  study = optuna.create_study(direction=direction)
  study.optimize(objective, n_trials=n_trials)
  
  print("Best parameters: ", study.best_trial.params)
  print("Best score: ", study.best_value)

  
