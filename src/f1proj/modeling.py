# Random Forest Feature Importance
# =========================================================
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from f1proj.eval import years_series_from_master

def train_rf(X, y, random_state=42):
    model = RandomForestRegressor(n_estimators=500, random_state=random_state, n_jobs=-1)
    model.fit(X, y)
    return model

def rf_feature_importance(X, y, top_k=20, random_state=42, n_estimators=500):
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return imp.head(top_k), rf

def years_series_from_master(master_df):
    """
    master_df içinde yıl bilgisini döndürür.
    Tercih sırası: 'year' -> 'race_year' -> 'season'
    """
    for col in ["year", "race_year", "season"]:
        if col in master_df.columns:
            return master_df[col]
    raise KeyError("Yıl kolonu bulunamadı: master_df 'year' / 'race_year' / 'season' içermiyor.")

def split_train_test_by_last_n_years(master_df, X_df, y_ser, holdout_years=2):
    years = years_series_from_master(master_df)
    uniq_years = sorted(years.unique())
    if holdout_years >= len(uniq_years):
        raise ValueError("holdout_years, mevcut yıl sayısından küçük olmalı")
    test_years = uniq_years[-holdout_years:]
    train_mask = ~years.isin(test_years)
    test_mask  = years.isin(test_years)
    return (
        X_df.loc[train_mask], X_df.loc[test_mask],
        y_ser.loc[train_mask], y_ser.loc[test_mask],
        test_years
    )

def time_series_cv_scores(master_df, X_df, y_ser, n_splits=5, random_state=42):
    years = years_series_from_master(master_df)
    gkf = GroupKFold(n_splits=n_splits)
    rmses, maes, r2s = [], [], []
    for tr_idx, val_idx in gkf.split(X_df, y_ser, groups=years):
        X_tr, X_val = X_df.iloc[tr_idx], X_df.iloc[val_idx]
        y_tr, y_val = y_ser.iloc[tr_idx], y_ser.iloc[val_idx]
        rf_cv = RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1
        )
        rf_cv.fit(X_tr, y_tr)
        pred = rf_cv.predict(X_val)
        rmse = mean_squared_error(y_val, pred, squared=False)
        mae  = mean_absolute_error(y_val, pred)
        r2   = r2_score(y_val, pred)
        rmses.append(rmse); maes.append(mae); r2s.append(r2)
    return dict(RMSE=np.mean(rmses), MAE=np.mean(maes), R2=np.mean(r2s),
                RMSE_std=np.std(rmses), MAE_std=np.std(maes), R2_std=np.std(r2s))
