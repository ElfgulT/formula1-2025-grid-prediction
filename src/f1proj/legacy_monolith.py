# =========================================================
# F1 puan tahmini – UÇTAN UCA TEK DOSYA
# =========================================================
# Bu dosyada arşiv verisini okuttum, temizledim, feature ürettim, modeli eğittim
# 2023–2024 hold-out değerlendirmesi yaptım, 2025 verisini FastF1 ile çekip tahmin tablosu ürettim.
#Bazı yerlerde grafik oluşturmak için kalabalık yapan kodlar mevcuttur.


ARCHIVE_DIR = "/Users/elifgultopcu/Downloads/formula1/archive"
LIVE2025_DIR = "/Users/elifgultopcu/Downloads/formula1/2025_live"
KAGGLE_2025_DIR = "/Users/elifgultopcu/Downloads/formula1/kaggle_2025"

USE_GRID_FEATURE = True #gridsiz model için false


MANUAL_NAME_OVERRIDES_2025: dict[int, str] = {}

import os
import re
import warnings
import numpy as np
import pandas as pd

from pandas.api.types import (is_categorical_dtype,is_object_dtype,is_numeric_dtype)
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
import joblib
from sklearn.impute import KNNImputer, SimpleImputer


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 180)
pd.set_option('display.float_format', lambda x: f"{x:.3f}")

# =========================================================
# Yardımcılar: zaman dönüşümü, duration doldurma
# =========================================================
def parse_lap_time_to_seconds(s):
    if pd.isna(s):
        return np.nan
    s = str(s)
    try:
        parts = s.split(':')
        if len(parts) == 2:  # m:ss.xxx
            m = float(parts[0]); sec = float(parts[1])
            return m*60 + sec
        elif len(parts) == 3:  # h:mm:ss.xxx
            h = float(parts[0]); m = float(parts[1]); sec = float(parts[2])
            return h*3600 + m*60 + sec
        else:
            return float(s)  # belki zaten saniye
    except:
        return np.nan

def parse_duration_to_seconds(s):
    if pd.isna(s):
        return np.nan
    s = str(s)
    try:
        return float(s)  # "2.345" gibi saniye
    except:
        return parse_lap_time_to_seconds(s)

# =========================================================
# Temizlik
# =========================================================
def clean_dataset(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    df = df.copy()
    df = df.replace('\\N', np.nan)
    name = dataset_name.lower().strip()

    if name in ['circuits','circuit']:
        for col in ['circuitId','circuitRef','name','location','country','url']:
            if col in df.columns:
                if col.endswith('Id') or col.endswith('Ref'):
                    df[col] = df[col].astype('category')
                else:
                    df[col] = df[col].astype('object')
        for col in ['lat','lng','alt']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    elif name in ['constructors','constructor']:
        for col in ['constructorId','constructorRef','name','nationality','url']:
            if col in df.columns:
                if col.endswith('Id') or col.endswith('Ref'):
                    df[col] = df[col].astype('category')
                else:
                    df[col] = df[col].astype('object')

    elif name in ['constructor_results','constructor-results','constructorresults']:
        for col in ['constructorResultsId','raceId','constructorId']:
            if col in df.columns:
                df[col] = df[col].astype('category')
        if 'points' in df.columns:
            df['points'] = pd.to_numeric(df['points'], errors='coerce')
        if 'status' in df.columns:
            df['status'] = df['status'].astype('object')

    elif name in ['constructor_standings','constructor-standings','constructorstandings']:
        for col in ['constructorStandingsId','raceId','constructorId']:
            if col in df.columns:
                df[col] = df[col].astype('category')
        for col in ['points','position','wins']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    elif name in ['drivers','driver']:
        for col in ['driverId','driverRef','number','code','forename','surname','nationality','url']:
            if col in df.columns:
                if col in ['driverId','driverRef','number','code']:
                    df[col] = df[col].astype('category')
                else:
                    df[col] = df[col].astype('object')
        if 'dob' in df.columns:
            df['dob'] = pd.to_datetime(df['dob'], errors='coerce')

    elif name in ['driver_standings','driver-standings','driverstandings']:
        for col in ['driverStandingsId','raceId','driverId']:
            if col in df.columns:
                df[col] = df[col].astype('category')
        for col in ['points','position','wins']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    elif name in ['lap_times','laptimes','lap-times']:
        for col in ['raceId','driverId']:
            if col in df.columns:
                df[col] = df[col].astype('category')
        for col in ['lap','position','milliseconds']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'time' in df.columns:
            df['time_seconds'] = df['time'].apply(parse_lap_time_to_seconds)

    elif name in ['pit_stops','pitstops','pit-stops']:
        for col in ['raceId','driverId']:
            if col in df.columns:
                df[col] = df[col].astype('category')
        for col in ['stop','lap','milliseconds']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'duration' in df.columns:
            duration_num = pd.to_numeric(df['duration'], errors='coerce')
            duration_sec_parsed = df['duration'].where(duration_num.notna(), df['duration'])
            duration_sec_parsed = duration_sec_parsed.apply(parse_duration_to_seconds)
            df['duration_seconds'] = np.where(duration_num.notna(), duration_num, duration_sec_parsed)
        else:
            df['duration_seconds'] = np.nan

        if 'milliseconds' in df.columns:
            mask_dur = df['duration_seconds'].isna() & df['milliseconds'].notna()
            df.loc[mask_dur, 'duration_seconds'] = df.loc[mask_dur, 'milliseconds'] / 1000.0
            mask_ms = df['milliseconds'].isna() & df['duration_seconds'].notna()
            df.loc[mask_ms, 'milliseconds'] = (df.loc[mask_ms, 'duration_seconds'] * 1000.0).round()

    elif name in ['qualifying','qualify']:
        for col in ['qualifyId','raceId','driverId','constructorId']:
            if col in df.columns:
                df[col] = df[col].astype('category')
        for col in ['position','number']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        for col in ['q1','q2','q3']:
            if col in df.columns:
                df[col + '_seconds'] = df[col].apply(parse_lap_time_to_seconds)

        if 'q1_seconds' in df.columns:
            df['q1_milliseconds'] = df['q1_seconds'] * 1000
            valid = df['q1_milliseconds'].dropna()
            if valid.shape[0] >= 100:
                q1 = valid.quantile(0.01)
                q3 = valid.quantile(0.99)
                iqr = q3 - q1
                low = q1 - 1.5*iqr
                up  = q3 + 1.5*iqr
                out_idx = df.index[(df['q1_milliseconds'] < low) | (df['q1_milliseconds'] > up)]
                if len(out_idx) <= 5:
                    df = df.drop(out_idx)

    elif name in ['races','race']:
        for col in ['raceId','round','circuitId','name']:
            if col in df.columns:
                if col in ['raceId','round','circuitId']:
                    df[col] = df[col].astype('category')
                else:
                    df[col] = df[col].astype('object')
        if 'year' in df.columns:
            df['year'] = pd.to_datetime(df['year'], format='%Y', errors='coerce')
        for col in ['date','fp1_date','fp2_date','fp3_date','quali_date','sprint_date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        for col in ['time','fp1_time','fp2_time','fp3_time','quali_time','sprint_time','url']:
            if col in df.columns:
                df[col] = df[col].astype('object')

    elif name in ['results','result']:
        for col in ['resultId','raceId','driverId','constructorId','statusId','number']:
            if col in df.columns:
                df[col] = df[col].astype('category')
        for col in ['grid','position','positionOrder','points','laps','rank','fastestLap','milliseconds','fastestLapSpeed']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'positionText' in df.columns:
            df['positionText'] = df['positionText'].astype('category')
        if 'time' in df.columns:
            df['time'] = df['time'].astype('object')
        if 'fastestLapTime' in df.columns:
            df['fastestLapTime_seconds'] = df['fastestLapTime'].apply(parse_lap_time_to_seconds)

    elif name in ['seasons','season']:
        if 'year' in df.columns:
            df['year'] = pd.to_datetime(df['year'], format='%Y', errors='coerce')
        if 'url' in df.columns:
            df['url'] = df['url'].astype('object')

    elif name in ['sprint_results','sprint-results','sprintresults']:
        for col in ['resultId','raceId','driverId','constructorId','statusId','number']:
            if col in df.columns:
                df[col] = df[col].astype('category')
        for col in ['grid','position','positionOrder','points','laps','rank','fastestLap','milliseconds','fastestLapSpeed']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'positionText' in df.columns:
            df['positionText'] = df['positionText'].astype('category')
        if 'time' in df.columns:
            df['time'] = df['time'].astype('object')
        if 'fastestLapTime' in df.columns:
            df['fastestLapTime_seconds'] = df['fastestLapTime'].apply(parse_lap_time_to_seconds)

    return df

# =========================================================
# Hamları okur & temizleri bellekte döndürür
# =========================================================
def load_raw_and_clean(base_dir: str):
    def _read(name, file):
        path = os.path.join(base_dir, file)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return clean_dataset(pd.read_csv(path), name)

    return dict(
        circuits              = _read('circuits', 'circuits.csv'),
        constructors          = _read('constructors', 'constructors.csv'),
        constructor_results   = _read('constructor_results', 'constructor_results.csv'),
        constructor_standings = _read('constructor_standings', 'constructor_standings.csv'),
        drivers               = _read('drivers', 'drivers.csv'),
        driver_standings      = _read('driver_standings', 'driver_standings.csv'),
        lap_times             = _read('lap_times', 'lap_times.csv'),
        pit_stops             = _read('pit_stops', 'pit_stops.csv'),
        qualifying            = _read('qualifying', 'qualifying.csv'),
        races                 = _read('races', 'races.csv'),
        results               = _read('results', 'results.csv'),
        seasons               = _read('seasons', 'seasons.csv'),
        sprint_results        = _read('sprint_results', 'sprint_results.csv'),
        status                = _read('status', 'status.csv')
    )

# =========================================================
# Feature Engineeringe başlamadan
# =========================================================
def fill_cats_with_unknown(df, cols):
    for col in cols:
        if col not in df.columns:
            continue
        if is_categorical_dtype(df[col]):
            if "Unknown" not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories(['Unknown'])
            df[col] = df[col].fillna('Unknown')
        else:
            df[col] = df[col].astype('object').fillna('Unknown')
    return df

def collapse_rare_cats(df, cols, min_count, exclude_values=('Unknown','Other')):
    for col in cols:
        if col not in df.columns:
            continue
        vc = df[col].value_counts(dropna=False)
        rare = vc[(vc < min_count) & (~vc.index.isin(exclude_values))].index
        if len(rare) > 0:
            df[col] = df[col].apply(lambda x: 'Other' if x in rare else x)
    return df

def safe_group_shift_mean(df, by, col, window=5, min_periods=1):
    return df.groupby(by, sort=False)[col].transform(
        lambda s: s.shift().rolling(window, min_periods=min_periods).mean()
    )

def safe_group_shift_std(df, by, col, window=5, min_periods=2):
    return df.groupby(by, sort=False)[col].transform(
        lambda s: s.shift().rolling(window, min_periods=min_periods).std()
    )

def safe_group_shift_expanding_mean(df, by, col):
    return df.groupby(by, sort=False)[col].transform(
        lambda s: s.shift().expanding().mean()
    )

def encode_by_cardinality(master, max_cardinality=40, id_cols=('raceId','driverId','constructorId')):
    cat_cols = [c for c in master.columns if is_object_dtype(master[c]) and (c not in id_cols)]
    low_card  = [c for c in cat_cols if master[c].nunique(dropna=False) <= max_cardinality]
    high_card = [c for c in cat_cols if master[c].nunique(dropna=False) >  max_cardinality]

    le_map = {}
    for col in high_card:
        le = LabelEncoder()
        master[col] = le.fit_transform(master[col].astype(str))
        le_map[col] = le
    if len(low_card) > 0:
        master = pd.get_dummies(master, columns=low_card, drop_first=True)
    return master, le_map, low_card, high_card

def cast_remaining_categories_to_numeric(df):
    """object dışındaki category sütunları (örn: round) -> kod (int)."""
    for c in df.columns:
        if is_categorical_dtype(df[c]) and not is_object_dtype(df[c]):
            df[c] = df[c].cat.codes.astype('int64')
    return df

def drop_leaky_and_datetime_cols(X, leak_cols=('status','position'), drop_year_if_datetime=True):
    for c in leak_cols:
        if c in X.columns:
            X = X.drop(columns=[c])
    dt_cols = X.select_dtypes(include=['datetime64[ns]','datetimetz']).columns.tolist()
    if drop_year_if_datetime and 'year' in X.columns and str(X['year'].dtype).startswith('datetime64'):
        dt_cols.append('year')
    if dt_cols:
        X = X.drop(columns=list(set(dt_cols)))
    return X



def knn_impute_numeric(df, exclude_cols=()):
    df = df.copy()

    # Sayısal (object olmayan) sütunları seç
    num_cols = [c for c in df.columns
                if (c not in exclude_cols)
                and is_numeric_dtype(df[c])
                and not is_object_dtype(df[c])]

    # Kenar durum: satır yok veya numerik yok
    if df.shape[0] == 0 or len(num_cols) == 0:
        return df

    # Tüm-nan kolonları 0.0 ile doldurur (KNN tüm-nan kabul etmez)
    all_nan_cols = [c for c in num_cols if df[c].isna().all()]
    if all_nan_cols:
        df[all_nan_cols] = 0.0

    # Sonsuz -> NaN
    for c in num_cols:
        if np.isinf(df[c]).any():
            df.loc[np.isinf(df[c]), c] = np.nan

    # Sadece numerikleri float olarak alır
    X_num = df[num_cols].astype(float)
    cols_used = list(X_num.columns)

    try:
        imputer = KNNImputer(n_neighbors=3)
        values = imputer.fit_transform(X_num)
        imputed_df = pd.DataFrame(values, index=df.index, columns=cols_used)
        df[cols_used] = imputed_df
    except Exception:
        # Geri düş: sabit doldurma
        simp = SimpleImputer(strategy='constant', fill_value=0.0)
        values = simp.fit_transform(X_num)
        imputed_df = pd.DataFrame(values, index=df.index, columns=cols_used)
        df[cols_used] = imputed_df

    return df

# =========================================================
# Feature Engineering Ana Fonksiyonuy
# =========================================================
def create_features(
    results, races, drivers, constructors, status, circuits,
    driver_standings, constructor_standings,
    *, max_cardinality=40, rare_min_count=None, add_extra_features=True
):
    res, rac, drv, cons, stat, circ = [x.copy() for x in [results, races, drivers, constructors, status, circuits]]
    dstd, cstd = driver_standings.copy(), constructor_standings.copy()

    master = (
        res.merge(rac[['raceId','year','round','circuitId']], on='raceId', how='left')
           .merge(drv[['driverId','driverRef','nationality','dob','forename','surname']], on='driverId', how='left')
           .merge(cons[['constructorId','constructorRef','nationality','name']], on='constructorId', how='left',
                  suffixes=('_driver','_constructor'))
           .merge(stat[['statusId','status']], on='statusId', how='left')
           .merge(circ[['circuitId','name','location','country']], on='circuitId', how='left', suffixes=('','_circuit'))
           .merge(dstd[['raceId','driverId','points','position']], on=['raceId','driverId'], how='left',
                  suffixes=('','_driver_stand'))
           .merge(cstd[['raceId','constructorId','points','position']], on=['raceId','constructorId'], how='left',
                  suffixes=('','_constructor_stand'))
    )

    master = master.rename(columns={
        'points_driver_stand'        : 'driver_total_points',
        'position_driver_stand'      : 'driver_stand_position',
        'points_constructor_stand'   : 'constructor_total_points',
        'position_constructor_stand' : 'constructor_stand_position',
        'name'                       : 'circuit_name',
        'name_constructor'           : 'constructor_name'
    })

    # İsme dair sütunları düşer (gürültü)
    master = master.drop(columns=['forename','surname','constructor_name'], errors='ignore')

    # Gereksizleri düş
    master = master.drop(columns=[
        'url','url_driver','url_const','url_circuit',
        'resultId','number','milliseconds',
        'fastestLapTime','fastestLapSpeed','fastestLap',
        'rank','time','statusId','positionText','positionOrder'
    ], errors='ignore')

    # Hedef target
    if 'points' not in master.columns:
        raise ValueError("results içinde 'points' yok—hedef üretilemedi.")
    master['points_norm']   = master.groupby('year')['points'].transform(lambda x: x / x.max())
    master['target_points'] = master['points'].astype(float)
    master = master.drop(columns=['points'], errors='ignore')

    master = master.sort_values(['driverId','year','round'])

    # leakage-safe geçmiş metrikler
    master['driver_last5_points']       = safe_group_shift_mean(master,'driverId','target_points',5,1)
    master['constructor_last5_points']  = safe_group_shift_mean(master,'constructorId','target_points',5,1)
    master['driver_last3_grid']         = safe_group_shift_mean(master,'driverId','grid',3,1)
    master['constructor_last3_grid']    = safe_group_shift_mean(master,'constructorId','grid',3,1)
    master['driver_circuit_avg_points']      = safe_group_shift_expanding_mean(master,['driverId','circuitId'],'target_points')
    master['constructor_circuit_avg_points'] = safe_group_shift_expanding_mean(master,['constructorId','circuitId'],'target_points')
    master['driver_consistency_std5']        = safe_group_shift_std(master,'driverId','target_points',5,2)
    master['constructor_consistency_std5']   = safe_group_shift_std(master,'constructorId','target_points',5,2)
    master['driver_last_status']      = master.groupby('driverId')['status'].shift()
    master['constructor_last_status'] = master.groupby('constructorId')['status'].shift()
    master['driver_first_race']       = (master.groupby('driverId').cumcount() == 0).astype(int)
    master['constructor_first_race']  = (master.groupby('constructorId').cumcount() == 0).astype(int)

    # yaş
    master['year_race']  = master['year'].dt.year if str(master['year'].dtype).startswith('datetime64') else master['year']
    master['driver_age'] = master['year_race'] - pd.to_datetime(master['dob'], errors='coerce').dt.year
    master['driver_age'] = master['driver_age'].fillna(master['driver_age'].median())
    master = master.drop(columns=['dob','year_race'], errors='ignore')

    # standings'i geçmişe kaydır
    master['pre_driver_total_points']        = master.groupby(['driverId','year'])['driver_total_points'].shift()
    master['pre_driver_stand_position']      = master.groupby(['driverId','year'])['driver_stand_position'].shift()
    master['pre_constructor_total_points']   = master.groupby(['constructorId','year'])['constructor_total_points'].shift()
    master['pre_constructor_stand_position'] = master.groupby(['constructorId','year'])['constructor_stand_position'].shift()

    # Ek özellikler
    if add_extra_features:
        master['driver_form_delta']        = master['driver_last5_points'] - master['driver_last3_grid']
        master['constructor_form_delta']   = master['constructor_last5_points'] - master['constructor_last3_grid']
        master['combined_circuit_affinity'] = np.nanmean(
            master[['driver_circuit_avg_points','constructor_circuit_avg_points']].values, axis=1
        )
        master['driver_last5_minus_last3'] = master['driver_last5_points'] - safe_group_shift_mean(
            master,'driverId','target_points',3,1
        )

    # Rolling/expanding NaN -> 0
    for c in [
        'driver_last5_points','constructor_last5_points','driver_last3_grid','constructor_last3_grid',
        'driver_circuit_avg_points','constructor_circuit_avg_points',
        'driver_consistency_std5','constructor_consistency_std5',
        'pre_driver_total_points','pre_driver_stand_position',
        'pre_constructor_total_points','pre_constructor_stand_position',
        'driver_form_delta','constructor_form_delta','combined_circuit_affinity','driver_last5_minus_last3'
    ]:
        if c in master.columns and not is_object_dtype(master[c]):
            master[c] = master[c].fillna(0)

    # Kategorikler
    cat_missing_cols = [
        'driverRef','constructorRef','circuit_name','location','country',
        'driver_last_status','constructor_last_status'
    ]
    master = fill_cats_with_unknown(master, cat_missing_cols)

    if rare_min_count is None:
        rare_min_count = max(5, int(0.005 * len(master)))
    master = collapse_rare_cats(master,
                                ['driverRef','constructorRef','circuit_name','location','country'],
                                min_count=rare_min_count,
                                exclude_values=('Unknown','Other'))

    for col in cat_missing_cols:
        if col in master.columns:
            master[col + '_missing'] = (master[col] == 'Unknown').astype(int)

    # Leak içerikli kolonları encode öncesi düşer
    master = master.drop(columns=[c for c in ['status','position'] if c in master.columns], errors='ignore')

    # Encoding (object)
    master, le_map, low_card_cols, high_card_cols = encode_by_cardinality(
        master, max_cardinality=max_cardinality,
        id_cols=('raceId','driverId','constructorId')
    )

    # Kalan category dtype'ları → int
    master = cast_remaining_categories_to_numeric(master)

    # KNN impute (sadece sayısal; ID/target hariç)
    master = knn_impute_numeric(master, exclude_cols={'target_points','raceId','driverId','constructorId'})

    # X, y
    y = master['target_points'].astype(float)
    X = master.drop(columns=['target_points'], errors='ignore')

    # datetime kırp
    X = drop_leaky_and_datetime_cols(X, leak_cols=(), drop_year_if_datetime=True)

    # Hedef türevi sızıntı
    for c in ['points_norm']:
        if c in X.columns:
            X = X.drop(columns=[c])

    # ID kolonları
    for c in ['raceId', 'driverId', 'constructorId']:
        if c in X.columns:
            X = X.drop(columns=[c])

    # Yarış sonrası oluşan kolonlar
    for c in ['laps', 'fastestLapTime_seconds']:
        if c in X.columns:
            X = X.drop(columns=[c])

    # standings ham kolonları (pre_* kalsın)
    for c in ['driver_total_points', 'driver_stand_position',
              'constructor_total_points', 'constructor_stand_position']:
        if c in X.columns:
            X = X.drop(columns=[c])

    # pist ezberi
    if 'circuitId' in X.columns:
        X = X.drop(columns=['circuitId'])

    # grid kullanma
    if not USE_GRID_FEATURE and 'grid' in X.columns:
        X = X.drop(columns=['grid'])

    meta = {
        'encoders'       : le_map,
        'low_card_cols'  : low_card_cols,
        'high_card_cols' : high_card_cols,
        'columns_X'      : X.columns.tolist(),
        'id_to_driver'   : drv[['driverId','forename','surname']].copy() if {'forename','surname'}.issubset(drivers.columns) else None,
        'id_to_team'     : cons[['constructorId','name']].copy() if 'name' in cons.columns else None
    }
    return master, X, y, meta

# =========================================================
# Random Forest Feature Importance
# =========================================================
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

# =========================================================
# Zaman bilinçli CrossValidat. / Split / Görselleştirme
# =========================================================
def years_series_from_master(master_df):
    yr = master_df['year']
    if str(yr.dtype).startswith('datetime64'):
        return yr.dt.year.astype(int)
    return yr.astype(int)

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

def plot_feature_importance(imp_series, top_k=20, title="Özellik Önemleri (RF)"):
    imp_top = imp_series.sort_values(ascending=True).tail(top_k)
    plt.figure(figsize=(8, max(5, int(top_k*0.45))))
    imp_top.plot(kind='barh')
    plt.title(title)
    plt.xlabel('Önem Skoru')
    plt.tight_layout()
    plt.show()

# =========================================================
# 2023–2024 sezon tablosu ve top-k kıyas yardımcıları
# =========================================================
def season_tables_from_preds(master_df, X_df, y_true, model, years, name_map=None):
    years_ser = years_series_from_master(master_df)
    mask = years_ser.isin(years)
    df = pd.DataFrame({
        'year'    : years_ser[mask].values,
        'raceId'  : master_df.loc[mask, 'raceId'].values,
        'driverId': master_df.loc[mask, 'driverId'].values,
        'y_true'  : (y_true.loc[mask].values if (y_true is not None and len(y_true)==len(master_df)) else 0.0),
    }, index=X_df.loc[mask].index)
    df['y_pred'] = model.predict(X_df.loc[mask])

    agg_pred = (df.groupby(['year','driverId'])['y_pred'].sum()
                .rename('pred_total').reset_index())
    season = agg_pred

    if (y_true is not None) and len(y_true)==len(master_df):
        agg_true = (df.groupby(['year','driverId'])['y_true'].sum()
                    .rename('true_total').reset_index())
        season = season.merge(agg_true, on=['year','driverId'], how='left')

    if name_map is not None:
        season = season.merge(name_map[['driverId','driver_name']], on='driverId', how='left')

    season['rank_pred'] = season.groupby('year')['pred_total'].rank(ascending=False, method='min')
    if 'true_total' in season.columns:
        season['rank_true'] = season.groupby('year')['true_total'].rank(ascending=False, method='min')
    return season

def topk_overlap_table(season_df, year, k=10):
    sub = season_df[season_df['year']==year].copy()
    if 'rank_true' not in sub.columns:
        raise ValueError("Bu tabloda true_total yok; overlap için gerçek sıralama bulunamadı.")
    pred_top = sub.nsmallest(k, 'rank_pred')
    true_top = sub.nsmallest(k, 'rank_true')
    pred_set = set(pred_top['driverId'])
    true_set = set(true_top['driverId'])
    overlap  = pred_set & true_set
    prec_at_k = len(overlap) / k
    from scipy.stats import spearmanr
    sp = spearmanr(sub['rank_true'], sub['rank_pred']).correlation
    show_pred = pred_top[['driver_name','pred_total','rank_pred']] if 'driver_name' in pred_top.columns else pred_top[['driverId','pred_total','rank_pred']]
    show_true = true_top[['driver_name','true_total','rank_true']] if 'driver_name' in true_top.columns else true_top[['driverId','true_total','rank_true']]
    return {
        'year': year,
        'precision@10': round(prec_at_k, 3),
        'overlap': len(overlap),
        'spearman_season': round(sp, 3)
    }, show_pred, show_true


# =========================================================
# İSİM/KİMLİK EŞLEME SELF-CHECK + EK KARŞILAŞTIRMA METRİKLERİ
# =========================================================

def build_driver_name_map_2025(drivers_df: pd.DataFrame, overrides: dict[int, str] | None = None) -> pd.DataFrame:
    """drivers.csv'den  bir isim haritası üretir: driverId, driver_name.
    İsim yoksa code/driverRef'e düşer; hepsi yoksa UNKNOWN atar. (Overrides destekler)
    """
    import numpy as np
    import pandas as pd
    if isinstance(drivers_df, pd.DataFrame) and not drivers_df.empty and 'driverId' in drivers_df.columns:
        nm = drivers_df.copy()
        for col in ['forename','surname','code','driverRef']:
            if col not in nm.columns:
                nm[col] = ''
        # "F SURNAME" formatı; yoksa CODE/driverRef'e düş
        nm['driver_name'] = (nm['forename'].fillna('').str.strip().str[0].str.upper() + ' ' +
                             nm['surname'].fillna('').str.upper()).str.strip()
        empty_mask = nm['driver_name'].isna() | (nm['driver_name'].str.len() == 0) | (nm['driver_name'] == ' ')
        nm.loc[empty_mask, 'driver_name'] = nm.loc[empty_mask, 'code'].astype(str).str.upper()
        empty_mask2 = nm['driver_name'].isna() | (nm['driver_name'].str.len() == 0) | (nm['driver_name'] == ' ')
        nm.loc[empty_mask2, 'driver_name'] = nm.loc[empty_mask2, 'driverRef'].astype(str).str.upper()
        nm['driver_name'] = nm['driver_name'].replace('', np.nan).fillna('UNKNOWN')

        # Manuel override'lar
        if overrides:
            for k, v in overrides.items():
                try:
                    nm.loc[nm['driverId'] == int(k), 'driver_name'] = str(v)
                except Exception:
                    pass

        return nm[['driverId','driver_name']].drop_duplicates('driverId')
    return pd.DataFrame(columns=['driverId','driver_name'])

def self_check_to_date_vs_results(season_df: pd.DataFrame, clean_2025: dict,
                                  id_col: str = 'driverId', true_col: str = 'true_total') -> None:
    """Sezon tablosundaki true_total(ya da true_to_date) ile results.csv toplamlarını karşılaştırır.Ergast APIdeki güncel
    yarış bilgileri eksikliği sebebiyle yaşanan sorunu önlemesi için """
    import pandas as pd
    try:
        res = clean_2025.get('results', pd.DataFrame())
        if not isinstance(res, pd.DataFrame) or res.empty:
            print('[CHECK] results.csv boş — self-check atlandı.')
            return
        # results.csv -> gerçek toplam
        true_from_results = (res.groupby(id_col)['points'].sum()
                               .rename('true_from_results').reset_index())
        # Sezon df'inde ilgili kolon
        if true_col not in season_df.columns:
            print(f"[CHECK] '{true_col}' bulunamadı — self-check atlandı.")
            return
        merged = (season_df[[id_col, true_col]].copy()
                    .merge(true_from_results, on=id_col, how='left'))
        merged['diff'] = merged[true_col].fillna(0) - merged['true_from_results'].fillna(0)
        bad = merged.loc[merged['diff'].abs() > 1e-6]
        if len(bad) > 0:
            print('[CHECK] true değerleri ile results toplamı arasında fark bulunan ilk 10 satır:')
            print(bad.head(10))
        else:
            print('[CHECK] true değerleri results.csv toplamlarıyla uyumlu.')
    except Exception as e:
        print('[CHECK] Self-check hatası:', e)




def debug_missing_driver_names(season_df: pd.DataFrame, clean_2025: dict) -> None:
    """İsim bulunamayan driverId'leri teşhis eder ve drivers.csv'den aday bilgileri yazdırır."""
    import pandas as pd
    try:
        if not isinstance(season_df, pd.DataFrame) or season_df.empty:
            print('[DEBUG] season_df boş — isim debug atlandı.')
            return
        if 'driver_name' not in season_df.columns:
            print('[DEBUG] season_df içinde driver_name sütunu yok.')
            return
        miss = season_df[season_df['driver_name'].isna()]
        if miss.empty:
            print('[CHECK] İsim eşleşmesi sorunu yok.')
            return
        ids = sorted(miss['driverId'].dropna().astype(int).unique().tolist())
        print('[DEBUG] İsim bulunamayan driverId listesi:', ids)

        drv = clean_2025.get('drivers', pd.DataFrame())
        if isinstance(drv, pd.DataFrame) and not drv.empty:
            cols = [c for c in ['driverId','code','driverRef','forename','surname'] if c in drv.columns]
            if cols:
                print('[DEBUG] drivers.csv eşleştirme adayı satırlar:')
                print(drv[drv['driverId'].isin(ids)][cols].drop_duplicates().head(20))

        res = clean_2025.get('results', pd.DataFrame())
        if isinstance(res, pd.DataFrame) and not res.empty and 'driverId' in res.columns:
            print('[DEBUG] results.csv içinde görülen driverId adayları (ilk 20):')
            print(res[res['driverId'].isin(ids)][['driverId']].drop_duplicates().head(20))

        print('> Bu driverId → isim atamalarını, dosyanın başındaki MANUAL_NAME_OVERRIDES_2025 içine ekleyebiliriz.')
    except Exception as e:
        print('[DEBUG] missing name debug hatası:', e)


def year_level_metrics(master_df: pd.DataFrame, X_df: pd.DataFrame, y_ser: pd.Series,
                       model, year: int, name_map: pd.DataFrame | None = None):
    """Belirli bir yıl için: (1) yarış-level RMSE/MAE/R2, (2) sezon precision@10 & Spearman döndür."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    years_ser = years_series_from_master(master_df)
    mask = (years_ser == year)
    y_pred = model.predict(X_df.loc[mask])
    rmse = mean_squared_error(y_ser.loc[mask], y_pred, squared=False)
    mae  = mean_absolute_error(y_ser.loc[mask], y_pred)
    r2   = r2_score(y_ser.loc[mask], y_pred)
    season = season_tables_from_preds(master_df, X_df, y_ser, model, years=[year], name_map=name_map)
    stats, _, _ = topk_overlap_table(season, year, k=10)
    out = dict(year=year, RMSE=rmse, MAE=mae, R2=r2)
    out.update(stats)
    return out


def export_presentation_figures(master: pd.DataFrame, X: pd.DataFrame, y: pd.Series, model,
                                test_years: list[int], season_2025: pd.DataFrame | None,
                                season_2025_proj: pd.DataFrame | None, out_dir: str,
                                grid_compare_metrics: dict | None = None):

    """Sunumu destekleyecek görselleri PNG olarak kaydedecek.
    - Özellik önemleri (Top-20)
    - 2023 ve 2024 için sezon (driver) bazlı true vs pred scatter
    - 2025 Proxy-Grid projeksiyonu Top-10 bar grafiği
    - (İstersek) ileride gridli vs gridsiz karşılaştırma ekleyebiliriz
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    # 1) Özellik önemleri
    if hasattr(model, 'feature_importances_'):
        fi = pd.Series(model.feature_importances_, index=getattr(model, 'feature_names_in_', X.columns))
        top = fi.sort_values(ascending=False).head(20).sort_values()
        plt.figure(figsize=(8, max(5, int(len(top) * 0.45))))
        top.plot(kind='barh')
        plt.title('RF Özellik Önemleri (Top-20)')
        plt.xlabel('Önem Skoru')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'fig_feature_importance_top20.png'), dpi=160)
        plt.close()

    # 2) 2023/2024 sezon bazlı scatter
    for yr in test_years:
        season = season_tables_from_preds(master, X, y, model, years=[yr], name_map=None)
        sub = season[season['year'] == yr].dropna(subset=['true_total']) if 'true_total' in season.columns else season.copy()
        if sub.empty:
            continue
        max_val = float(sub[['pred_total', 'true_total']].max().max()) if 'true_total' in sub.columns else float(sub['pred_total'].max())
        plt.figure(figsize=(6, 6))
        if 'true_total' in sub.columns:
            plt.scatter(sub['true_total'], sub['pred_total'], alpha=0.7)
            plt.plot([0, max_val], [0, max_val])
            plt.xlabel('Gerçek Sezon Toplamı')
        else:
            plt.scatter(range(len(sub)), sub['pred_total'], alpha=0.7)
            plt.xlabel('Sürücüler')
        plt.ylabel('Tahmin Edilen Sezon Toplamı')
        plt.title(f'{yr} — Sezon Tahmini: Gerçek vs Tahmin')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'fig_scatter_season_{yr}.png'), dpi=160)
        plt.close()

    # 3) 2025 Proxy-Grid Top-10 bar
    if isinstance(season_2025_proj, pd.DataFrame) and not season_2025_proj.empty and 'proj_total' in season_2025_proj.columns:
        top10 = season_2025_proj.sort_values('proj_total', ascending=False).head(10).copy()
        labels = top10.get('driver_name', top10.get('driverId')).astype(str).tolist()
        plt.figure(figsize=(9, 5))
        plt.bar(labels, top10['proj_total'].values)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Projeksiyon Toplam Puan')
        plt.title('2025 — Proxy-Grid Projeksiyonu (Top-10)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'fig_2025_proxygrid_top10.png'), dpi=160)
        plt.close()

    # 4) Gridli vs Gridsiz test metrikleri (RMSE & MAE — grup bar)
    if isinstance(grid_compare_metrics, dict):
        try:
            g = grid_compare_metrics.get('grid', {}) if grid_compare_metrics else {}
            ng = grid_compare_metrics.get('gridless', {}) if grid_compare_metrics else {}
            rmse_vals = [float(g.get('RMSE', np.nan)), float(ng.get('RMSE', np.nan))]
            mae_vals  = [float(g.get('MAE', np.nan)),  float(ng.get('MAE',  np.nan))]
            if not (np.isnan(rmse_vals).all() and np.isnan(mae_vals).all()):
                labels = ['Gridli', 'Gridsiz']
                x = np.arange(len(labels))
                width = 0.35
                plt.figure(figsize=(7, 5))
                # RMSE çubukları (sol grup)
                plt.bar(x - width/2, rmse_vals, width, label='RMSE')
                # MAE çubukları (sağ grup)
                plt.bar(x + width/2, mae_vals,  width, label='MAE')
                plt.xticks(x, labels)
                plt.ylabel('Hata (puan)')
                plt.title('Test Metrikleri: Gridli vs Gridsiz')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, 'fig_grid_vs_gridless_metrics.png'), dpi=160)
                plt.close()
        except Exception:
            pass


# ---------------------------------------------------------
# Ek: Post-race tarzı görseller (Effective Qualifying, Pit Stop, Constructor Timeline)
# ---------------------------------------------------------

def export_postrace_charts(clean_2025: dict, out_dir: str, year: int = 2025):
    """

      a) Effective Qualifying (sürücü/constructor ort. sıralama pozisyonu)
      b) Pit Stop ortalama süreleri (constructor bazında)
      c) Constructor Championship Timeline (kümülatif puan)
    Çıktılar outdirde . Boş veri varsa sessizce atlar.
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # Yardımcı: isim haritaları
    cons = clean_2025.get('constructors', pd.DataFrame())
    cons_map = cons.set_index('constructorId')['name'].to_dict() if 'constructorId' in cons.columns else {}
    drv = clean_2025.get('drivers', pd.DataFrame())
    drv_map = drv.set_index('driverId').apply(
        lambda r: f"{str(r.get('forename','')).strip()} {str(r.get('surname','')).strip()}".strip(), axis=1
    ).to_dict() if 'driverId' in drv.columns else {}

    races = clean_2025.get('races', pd.DataFrame())
    if 'year' in races.columns and 'raceId' in races.columns:
        year_race_ids = set(races.loc[races['year'] == year, 'raceId'].astype(int).tolist())
    else:
        year_race_ids = set()

    # a) Effective Qualifying (ortalama grid/qual pozisyonu — düşük daha iyi)
    try:
        q = clean_2025.get('qualifying', pd.DataFrame()).copy()
        res = clean_2025.get('results', pd.DataFrame()).copy()
        df_eq = pd.DataFrame()
        if not q.empty and {'raceId','driverId','position'}.issubset(q.columns):
            if year_race_ids:
                q = q[q['raceId'].isin(year_race_ids)]
            df_eq = (q.groupby('driverId')['position'].mean()
                       .rename('avg_qual_pos').reset_index())
        elif not res.empty and {'raceId','driverId','grid'}.issubset(res.columns):
            if year_race_ids:
                res = res[res['raceId'].isin(year_race_ids)]
            df_eq = (res.groupby('driverId')['grid'].mean()
                       .rename('avg_qual_pos').reset_index())
        if not df_eq.empty:
            # Etiketleri sürücü adına çevir
            df_eq['driver_name'] = df_eq['driverId'].map(drv_map).fillna(df_eq['driverId'].astype(str))
            df_plot = df_eq.sort_values('avg_qual_pos').head(15)  # en iyi 15
            plt.figure(figsize=(10, 6))
            plt.barh(df_plot['driver_name'], df_plot['avg_qual_pos'])
            plt.gca().invert_yaxis()
            plt.xlabel('Ortalama Sıralama Pozisyonu (↓ daha iyi)')
            plt.title(f'{year} Effective Qualifying — En İyi 15 Sürücü')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'fig_{year}_effective_qualifying.png'), dpi=160)
            plt.close()
    except Exception as e:
        print('[WARN] Effective Qualifying grafiği üretilemedi:', e)

    # b) Pit Stop ort. süreleri (constructor)
    try:
        pit = clean_2025.get('pit_stops', pd.DataFrame()).copy()
        res = clean_2025.get('results', pd.DataFrame()).copy()
        if not pit.empty and not res.empty and 'raceId' in pit.columns:
            if year_race_ids:
                pit = pit[pit['raceId'].isin(year_race_ids)]
                res = res[res['raceId'].isin(year_race_ids)]
            # Ergast şeması: duration (string) ve/veya milliseconds olabilir
            if 'milliseconds' in pit.columns and pit['milliseconds'].notna().any():
                pit['ms'] = pd.to_numeric(pit['milliseconds'], errors='coerce')
            else:
                # duration "MM:SS.sss" veya "SS.sss" olabilir → saniyeye çevir
                dur = pit.get('duration') if 'duration' in pit.columns else pit.get('time')
                dur = dur.astype(str).fillna('')
                def _to_ms(s):
                    try:
                        s = s.strip()
                        if not s:
                            return np.nan
                        parts = s.split(':')
                        if len(parts) == 2:
                            m, sec = int(parts[0]), float(parts[1])
                            return int((m*60 + sec)*1000)
                        return int(float(s)*1000)
                    except Exception:
                        return np.nan
                pit['ms'] = dur.map(_to_ms)
            # driverId → constructorId eşle (aynı raceId üzerinden)
            key_cols = ['raceId','driverId']
            if set(key_cols).issubset(res.columns) and 'constructorId' in res.columns:
                pit2 = pit.merge(res[key_cols + ['constructorId']].drop_duplicates(key_cols),
                                 on=key_cols, how='left')
                grp = (pit2.dropna(subset=['ms','constructorId'])
                           .groupby('constructorId')['ms'].mean().rename('avg_ms').reset_index())
                if not grp.empty:
                    grp['constructor_name'] = grp['constructorId'].map(cons_map).fillna(grp['constructorId'].astype(str))
                    grp = grp.sort_values('avg_ms')
                    plt.figure(figsize=(10, 5))
                    plt.bar(grp['constructor_name'], grp['avg_ms']/1000.0)
                    plt.ylabel('Ortalama Pit Süresi (s)')
                    plt.xticks(rotation=45, ha='right')
                    plt.title(f'{year} Pit Stop Ortalama Süreleri (Constructor)')
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, f'fig_{year}_pitstop_constructor.png'), dpi=160)
                    plt.close()
    except Exception as e:
        print('[WARN] Pit Stop grafiği üretilemedi:', e)

    # c) Constructor Championship Timeline (kümülatif)
    try:
        res = clean_2025.get('results', pd.DataFrame()).copy()
        races = clean_2025.get('races', pd.DataFrame()).copy()
        if not res.empty and not races.empty and {'raceId','constructorId','points'}.issubset(res.columns):
            res = res.merge(races[['raceId','year','round']], on='raceId', how='left')
            res = res[res['year'] == year]
            if not res.empty:
                tbl = (res.groupby(['constructorId','round'])['points'].sum()
                         .rename('pts').reset_index())
                # kümülatif
                tbl = tbl.sort_values(['constructorId','round'])
                tbl['cum_pts'] = tbl.groupby('constructorId')['pts'].cumsum()
                # geniş pivot
                wide = tbl.pivot(index='round', columns='constructorId', values='cum_pts').fillna(method='ffill')
                plt.figure(figsize=(10, 6))
                for cid in wide.columns:
                    label = cons_map.get(cid, str(cid))
                    plt.plot(wide.index, wide[cid].values, label=label)
                plt.xlabel('Round')
                plt.ylabel('Kümülatif Puan')
                plt.title(f'{year} Constructor Championship Timeline')
                plt.legend(ncol=2, fontsize=8)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f'fig_{year}_constructor_timeline.png'), dpi=160)
                plt.close()
    except Exception as e:
        print('[WARN] Constructor timeline grafiği üretilemedi:', e)

# ---------------------------------------------------------
# Ek: Feature Engineering odaklı 4 grafik (hist, scatter, heatmap, FI)
#  + FE öncesi/sonrası karşılaştırma metrik grafiği
# ---------------------------------------------------------

def export_feature_engineering_extra_charts(clean_archive: dict, master_df: pd.DataFrame, out_dir: str,
                                            recent_years: tuple[int, int] | None = (2018, 2024)):


    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # --- 1) Qualifying vs Race Delta histogram ---
    try:
        res = clean_archive.get('results', pd.DataFrame()).copy()
        races = clean_archive.get('races', pd.DataFrame()).copy()
        if not res.empty and not races.empty and {'raceId','grid','position'}.issubset(res.columns):
            res = res.merge(races[['raceId','year']], on='raceId', how='left')
            if recent_years is not None and 'year' in res.columns:
                y0, y1 = recent_years
                res = res[(res['year'] >= y0) & (res['year'] <= y1)]
            res['grid'] = pd.to_numeric(res['grid'], errors='coerce')
            res['position'] = pd.to_numeric(res['position'], errors='coerce')
            df = res.dropna(subset=['grid','position']).copy()
            if not df.empty:
                df['delta_finish_minus_grid'] = df['position'] - df['grid']  # negatif = yer kazancı
                vals = df['delta_finish_minus_grid'].clip(-10, 10).values
                plt.figure(figsize=(8,5))
                plt.hist(vals, bins=21)
                plt.axvline(0, linestyle='--')
                plt.xlabel('Finish - Grid (negatif → yer kazandı)')
                plt.ylabel('Yarış Sayısı')
                title_span = f" ({recent_years[0]}–{recent_years[1]})" if recent_years else ""
                plt.title('Qualifying vs Race Delta Dağılımı' + title_span)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, 'fig_fe_delta_grid_hist.png'), dpi=160)
                plt.close()
    except Exception as e:
        print('[WARN] FE delta histogram üretilemedi:', e)

    # --- 2) Pit stop sayısı (sürücü başına ort.) vs ortalama puan ---
    try:
        pit = clean_archive.get('pit_stops', pd.DataFrame()).copy()
        res = clean_archive.get('results', pd.DataFrame()).copy()
        races = clean_archive.get('races', pd.DataFrame()).copy()
        if not pit.empty and not res.empty and {'raceId','driverId'}.issubset(pit.columns):
            pitn = (pit.groupby(['raceId','driverId'])['stop'].count()
                      .rename('pit_count').reset_index())
            res2 = res.copy()
            if not races.empty and 'raceId' in races.columns and 'year' in races.columns and recent_years is not None:
                res2 = res2.merge(races[['raceId','year']], on='raceId', how='left')
                y0, y1 = recent_years
                res2 = res2[(res2['year'] >= y0) & (res2['year'] <= y1)]
                pitn = pitn.merge(races[['raceId','year']], on='raceId', how='left')
                pitn = pitn[(pitn['year'] >= y0) & (pitn['year'] <= y1)]
            merged = (pitn.merge(res2[['raceId','driverId','points']], on=['raceId','driverId'], how='left')
                           .dropna(subset=['pit_count','points']))
            if not merged.empty:
                agg = (merged.groupby('driverId').agg(avg_pit=('pit_count','mean'),
                                                      avg_pts=('points','mean')).reset_index())
                plt.figure(figsize=(7,5))
                plt.scatter(agg['avg_pit'], agg['avg_pts'], alpha=0.6)
                # basit eğilim çizgisi
                if len(agg) >= 3:
                    m, b = np.polyfit(agg['avg_pit'], agg['avg_pts'], 1)
                    xs = np.linspace(float(agg['avg_pit'].min()), float(agg['avg_pit'].max()), 50)
                    plt.plot(xs, m*xs + b)
                plt.xlabel('Ortalama Pit Sayısı / Yarış (sürücü)')
                plt.ylabel('Ortalama Puan / Yarış')
                plt.title('Pit Sayısı vs Puan İlişkisi (sürücü bazında)')
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, 'fig_fe_pit_vs_points.png'), dpi=160)
                plt.close()
    except Exception as e:
        print('[WARN] Pit vs points scatter üretilemedi:', e)

    # --- 3) Korelasyon ısı haritası (seçilmiş FE özellikleri) ---
    try:
        cols = [c for c in [
            'target_points','grid','driver_last5_points','constructor_last5_points','driver_last3_grid',
            'constructor_last3_grid','driver_circuit_avg_points','constructor_circuit_avg_points',
            'driver_consistency_std5','constructor_consistency_std5','driver_age',
            'pre_driver_total_points','pre_constructor_total_points','driver_form_delta',
            'constructor_form_delta','combined_circuit_affinity','driver_last5_minus_last3'
        ] if c in master_df.columns]
        sub = master_df[cols].copy()
        for c in sub.columns:
            sub[c] = pd.to_numeric(sub[c], errors='coerce')
        sub = sub.dropna()
        if not sub.empty and sub.shape[1] >= 3:
            corr = sub.corr(numeric_only=True)
            plt.figure(figsize=(min(12, 1.1*len(cols)), min(10, 1.1*len(cols))))
            im = plt.imshow(corr.values, aspect='auto')
            plt.xticks(range(len(cols)), cols, rotation=45, ha='right')
            plt.yticks(range(len(cols)), cols)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.title('FE Özellikleri – Hedef Korelasyon Isı Haritası')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'fig_fe_corr_heatmap.png'), dpi=160)
            plt.close()
    except Exception as e:
        print('[WARN] Korelasyon ısı haritası üretilemedi:', e)


    try:
        # Eğer export_presentation_figures zaten ürettiyse tekrar etmeyelim
        fi_path = os.path.join(out_dir, 'fig_feature_importance_top20.png')
        if not os.path.exists(fi_path):
            # Burada tekrar üretmek yerine mevcut fonksiyona bırakıyoruz.
            pass
    except Exception:
        pass


def export_fe_vs_baseline_metrics_chart(master_df: pd.DataFrame, X_full: pd.DataFrame, y: pd.Series,
                                         test_years: list[int], out_dir: str,
                                         random_state: int = 42):
    """
    Aynı train/test bölünmesiyle iki model kurar:
      - Baseline (FE'siz/az FE): sadece grid, round ve temel kategori kodları
      - Full FE: mevcut X_full
    Sonra RMSE, MAE, R2 metriklerini yan yana çubuk grafikte gösterir.
    Yani FE öncesi ve sonrası metrikler.
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    os.makedirs(out_dir, exist_ok=True)

    # 1) Baseline kolonları (FE prefikslerini dışla)
    fe_prefixes = (
        'driver_last', 'constructor_last', 'driver_circuit_', 'constructor_circuit_',
        'driver_consistency_', 'constructor_consistency_', 'pre_driver_', 'pre_constructor_',
        'driver_form_delta', 'constructor_form_delta', 'combined_circuit_affinity',
        'driver_first_race', 'constructor_first_race', 'driver_last5_minus_last3'
    )
    base_keep_exact = {'grid', 'round', 'driverRef', 'constructorRef'}
    base_keep_prefix = ('constructorRef_', 'circuit_name_', 'location_', 'country_', 'nationality')

    base_cols = []
    for c in X_full.columns:
        if c in base_keep_exact:
            base_cols.append(c); continue
        if any(c.startswith(p) for p in fe_prefixes):
            continue
        if any(c.startswith(p) for p in base_keep_prefix):
            base_cols.append(c); continue
    if 'grid' in X_full.columns and 'grid' not in base_cols:
        base_cols.append('grid')
    if 'round' in X_full.columns and 'round' not in base_cols:
        base_cols.append('round')

    if len(base_cols) < 2:
        print('[WARN] Baseline kolonları çok az bulundu; FE karşılaştırma grafiği atlandı.')
        return

    X_base = X_full[base_cols].copy()

    # 2) Aynı test yıllarıyla böl
    years = years_series_from_master(master_df)
    test_mask = years.isin(test_years)
    train_mask = ~test_mask

    Xb_tr, Xb_te = X_base.loc[train_mask], X_base.loc[test_mask]
    Xf_tr, Xf_te = X_full.loc[train_mask], X_full.loc[test_mask]
    y_tr, y_te   = y.loc[train_mask],      y.loc[test_mask]

    # 3) Modeller
    rf_base = RandomForestRegressor(n_estimators=500, max_depth=None, random_state=random_state, n_jobs=-1)
    rf_full = RandomForestRegressor(n_estimators=500, max_depth=None, random_state=random_state, n_jobs=-1)
    rf_base.fit(Xb_tr, y_tr)
    rf_full.fit(Xf_tr, y_tr)

    # 4) Metrikler
    pred_b = rf_base.predict(Xb_te)
    pred_f = rf_full.predict(Xf_te)

    def _metrics(y_true, y_pred):
        return dict(
            RMSE=float(mean_squared_error(y_true, y_pred, squared=False)),
            MAE =float(mean_absolute_error(y_true, y_pred)),
            R2  =float(r2_score(y_true, y_pred))
        )

    mb = _metrics(y_te, pred_b)
    mf = _metrics(y_te, pred_f)

    # 5) Grafik: 3 metrik, iki model
    labels = ['RMSE','MAE','R2']
    base_vals = [mb['RMSE'], mb['MAE'], mb['R2']]
    full_vals = [mf['RMSE'], mf['MAE'], mf['R2']]

    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(7,5))
    plt.bar(x - width/2, base_vals, width, label='Baseline (az FE)')
    plt.bar(x + width/2, full_vals, width, label='Full FE')
    plt.xticks(x, labels)
    plt.ylabel('Değer')
    plt.title('Feature Engineering Öncesi / Sonrası Test Metrikleri')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig_fe_before_after_metrics.png'), dpi=160)
    plt.close()

    print('[FE-METRICS] Baseline:', mb)
    print('[FE-METRICS] Full FE :', mf)
# ---------------------------------------------------------
# Yeni: Inference sırasında kolonları eğitim şemasına hizala
# ---------------------------------------------------------
def align_columns_for_inference(X_new: pd.DataFrame, train_columns: list) -> pd.DataFrame:
    """
    Eğitimde kullanılan X kolon listesini referans alarak,
    yeni gelen X'i aynı sırada yeniden indeksler.
    Eksik kolonları 0 ile doldurur, fazlalıkları atar.
    """
    if not isinstance(train_columns, (list, tuple)):
        train_columns = list(train_columns)
    X_aligned = X_new.reindex(columns=train_columns, fill_value=0)
    return X_aligned


def import_f1_2025_kaggle_to_base(kaggle_dir: str, out_dir: str, archive_dir: str):
    """
    Kaggle 2025 csv'lerini (QualifyingResults / RaceResults) alır,
    benim ilk şemama uygun results.csv ve qualifying.csv dosyalarını out_dir'e yazar.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Kaynak dosyalar
    qual_path = os.path.join(kaggle_dir, "F1_2025_QualifyingResults.csv")
    race_path = os.path.join(kaggle_dir, "F1_2025_RaceResults.csv")

    if not (os.path.exists(qual_path) and os.path.exists(race_path)):
        print("[WARN] Kaggle 2025 dosyaları bulunamadı:", qual_path, race_path)
        return

    # Kimlik eşleştirmeleri (ergast arşivinden)
    drv = pd.read_csv(os.path.join(archive_dir, "drivers.csv"))
    con = pd.read_csv(os.path.join(archive_dir, "constructors.csv"))

    drv["fullname"] = (drv["forename"].astype(str).str.strip() + " " + drv["surname"].astype(str).str.strip()).str.replace(r"\s+", " ", regex=True)
    drv_map = drv.set_index("fullname")["driverId"].to_dict()

    # takımlar: isimler tutarsız olabilir; constructorRef + name ikilisini deneriz
    con_names = con.copy()
    con_names["normname"] = con_names["name"].astype(str).str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    con_map = con_names.set_index("normname")["constructorId"].to_dict()

    # 2025 takvim eşleştirmesi: Track → round
    # (Gerektiğinde güncelle. Sıra standart 2025 takvimi)
    track_to_round = {
        "Australia": 1, "China": 2, "Japan": 3, "Bahrain": 4, "Saudi Arabia": 5,
        "Miami": 6, "Emilia Romagna": 7, "Monaco": 8, "Spain": 9, "Canada": 10,
        "Austria": 11, "Great Britain": 12, "Belgium": 13, "Hungary": 14,
        "Netherlands": 15, "Italy": 16, "Azerbaijan": 17, "Singapore": 18,
        "United States": 19, "Mexico": 20, "São Paulo": 21, "Las Vegas": 22,
        "Qatar": 23, "Abu Dhabi": 24
    }

    # races.csv (out_dir) varsa raceId üretiminde kullan
    races_csv = os.path.join(out_dir, "races.csv")
    if os.path.exists(races_csv):
        races_df = pd.read_csv(races_csv)
        races_df["year"] = 2025
        races_df["raceId"] = races_df["year"] * 100 + races_df["round"].astype(int)
        round_to_raceid = races_df.set_index("round")["raceId"].to_dict()
    else:
        # yoksa basit üretim
        round_to_raceid = {rnd: 2025*100 + rnd for rnd in track_to_round.values()}

    # === Qualifying ===
    q = pd.read_csv(qual_path)
    q["driver_full"] = q["Driver"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    q["driverId"] = q["driver_full"].map(drv_map)

    q["team_norm"] = q["Team"].astype(str).str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    q["constructorId"] = q["team_norm"].map(con_map)

    q["round"] = q["Track"].map(track_to_round)
    q["raceId"] = q["round"].map(round_to_raceid)

    # === Qualifying ===
    q_out = pd.DataFrame({
        "qualifyId": range(1, len(q) + 1),
        "raceId": q["raceId"],
        "driverId": q["driverId"],
        "constructorId": q["constructorId"],
        "number": pd.to_numeric(q.get("No"), errors="coerce"),
        "position": pd.to_numeric(q.get("Position"), errors="coerce"),
        "q1": q.get("Q1"),
        "q2": q.get("Q2"),
        "q3": q.get("Q3")
    })

    # Temizle ve sonra yaz
    for col in ["driverId", "constructorId", "raceId"]:
        q_out[col] = pd.to_numeric(q_out[col], errors="coerce")

    q_out = q_out.dropna(subset=["raceId", "driverId", "constructorId"])
    q_out = q_out[(q_out["driverId"] != 0) & (q_out["constructorId"] != 0)]
    q_out = q_out.drop_duplicates()

    # === Results ===
    # ... r, grid_col, laps_col, time_col, pts_col hazırlıkları ...
    # Kaggle RaceResults'u okur ve ortak şemaya eşler
    r = pd.read_csv(race_path)

    # driver/constructor eşleşmeleri (qual tarafıyla aynı eşleme kullan)
    r["Driver"] = r["Driver"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    r["driverId"] = r["Driver"].map(drv_map)

    r["team_norm"] = r["Team"].astype(str).str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    r["constructorId"] = r["team_norm"].map(con_map)

    r["round"] = r["Track"].map(track_to_round)
    r["raceId"] = r["round"].map(round_to_raceid)

    # Esnek kolon tespiti (var olana düş)
    def pick(colnames):
        for c in colnames:
            if c in r.columns:
                return c
        return None

    grid_col = pick(["Grid", "StartingGrid", "StartPosition", "grid"])
    laps_col = pick(["Laps", "CompletedLaps", "laps"])
    time_col = pick(["Time", "RaceTime", "TotalTime", "time"])
    pts_col = pick(["PTS", "Points", "Score", "points"])

    # Güvenlik: id alanları sayı olsun
    for col in ["driverId", "constructorId", "raceId"]:
        r[col] = pd.to_numeric(r[col], errors="coerce")

    results_out = pd.DataFrame({
        "resultId": range(1, len(r) + 1),
        "raceId": r["raceId"],
        "driverId": r["driverId"],
        "constructorId": r["constructorId"],
        "number": pd.to_numeric(r.get("No"), errors="coerce"),
        "grid": pd.to_numeric(r[grid_col], errors="coerce") if grid_col in r.columns else pd.NA,
        "position": pd.to_numeric(r.get("Position"), errors="coerce"),
        "positionText": r.get("Position").astype(str),
        "positionOrder": pd.NA,
        "points": pd.to_numeric(r[pts_col], errors="coerce") if pts_col in r.columns else 0.0,
        "laps": pd.to_numeric(r[laps_col], errors="coerce") if laps_col in r.columns else pd.NA,
        "time": r[time_col] if time_col in r.columns else pd.NA,
        "milliseconds": pd.NA,
        "fastestLap": pd.NA,
        "rank": pd.NA,
        "fastestLapTime": pd.NA,
        "fastestLapSpeed": pd.NA,
        "statusId": pd.NA
    })

    for col in ["driverId", "constructorId", "raceId"]:
        results_out[col] = pd.to_numeric(results_out[col], errors="coerce")

    results_out = results_out.dropna(subset=["raceId", "driverId", "constructorId"])
    results_out = results_out[(results_out["driverId"] != 0) & (results_out["constructorId"] != 0)]
    results_out = results_out.drop_duplicates()

    # En sonda yaz
    q_out.to_csv(os.path.join(out_dir, "qualifying.csv"), index=False)
    results_out.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    print("[OK] Kaggle 2025 → qualifying.csv & results.csv yazıldı →", out_dir)


# =========================================================
# 2025: FastF1 (timing) ile tamamlanan yarışları çeker → CSV
# =========================================================
def fetch_2025_via_fastf1(save_dir: str):
    """
    2025 sezonunda tamamlanmış yarışları FastF1 timing API üzerinden çekip
    minimal CSV çıktıları üretir (races, circuits, drivers, constructors, results).
    """
    try:
        import fastf1
    except ImportError:
        print("fastf1 bulunamadı. Yüklemek için: pip install fastf1")
        os.makedirs(save_dir, exist_ok=True)
        # Boş şema oluşturup çıkar
        for fname, cols in [
            ('races.csv',              ['raceId','year','round','circuitId','name','date','time']),
            ('circuits.csv',           ['circuitId','circuitRef','name','location','country','lat','lng','alt','url']),
            ('drivers.csv',            ['driverId','driverRef','number','code','forename','surname','dob','nationality','url']),
            ('constructors.csv',       ['constructorId','constructorRef','name','nationality','url']),
            ('results.csv',            ['resultId','raceId','driverId','constructorId','number','grid','position','positionText','positionOrder','points','laps','time','milliseconds','fastestLap','rank','fastestLapTime','fastestLapSpeed','statusId']),
            ('driver_standings.csv',   ['driverStandingsId','raceId','driverId','points','position','positionText','wins']),
            ('constructor_standings.csv',['constructorStandingsId','raceId','constructorId','points','position','positionText','wins']),
            ('qualifying.csv',         ['qualifyId','raceId','driverId','constructorId','number','position','q1','q2','q3']),
            ('sprint_results.csv',     ['resultId','raceId','driverId','constructorId','number','grid','position','positionText','positionOrder','points','laps','time','milliseconds','fastestLap','rank','fastestLapTime','fastestLapSpeed','statusId']),
            ('seasons.csv',            ['year','url']),
            ('status.csv',             ['statusId','status']),
            ('lap_times.csv',          ['raceId','driverId','lap','position','time','milliseconds']),
            ('pit_stops.csv',          ['raceId','driverId','stop','lap','time','duration','milliseconds']),
            ('constructor_results.csv',['constructorResultsId','raceId','constructorId','points','status']),
        ]:
            p = os.path.join(save_dir, fname)
            if not os.path.exists(p):
                pd.DataFrame(columns=cols).to_csv(p, index=False)
        return

    import numpy as np
    os.makedirs(save_dir, exist_ok=True)
    try:
        fastf1.Cache.enable_cache(os.path.expanduser("~/Library/Caches/fastf1"))
    except Exception:
        fastf1.Cache.enable_cache(os.path.join(save_dir, "_fastf1_cache"))

    # 1) Takvim
    sched = fastf1.get_event_schedule(2025)
    sched['RoundNumber'] = pd.to_numeric(sched['RoundNumber'], errors='coerce').astype('Int64')
    sched = sched.dropna(subset=['RoundNumber']).copy()

    def slugify(s):
        s = str(s).lower()
        return re.sub(r'[^a-z0-9]+', '_', s).strip('_')

    circ_tab = []
    for _, r in sched.iterrows():
        circ_ref = slugify(r.get('Location', r.get('EventName', 'unknown')))
        circ_name = str(r.get('OfficialEventName', r.get('EventName', 'Unknown')))
        circ_tab.append((circ_ref, circ_name, str(r.get('Location', '')), str(r.get('Country', ''))))
    circ_df = pd.DataFrame(circ_tab, columns=['circuitRef','name','location','country']).drop_duplicates('circuitRef')
    circ_df['circuitId'] = pd.factorize(circ_df['circuitRef'])[0] + 1
    circuits_out = circ_df[['circuitId','circuitRef','name','location','country']].copy()
    circuits_out['lat'] = np.nan; circuits_out['lng'] = np.nan; circuits_out['alt'] = np.nan; circuits_out['url'] = np.nan

    # 2) races.csv
    races_rows = []
    for _, r in sched.iterrows():
        rnd = int(r['RoundNumber'])
        race_id = 2025 * 100 + rnd
        circ_ref = slugify(r.get('Location', r.get('EventName', 'unknown')))
        cid = circuits_out.loc[circuits_out['circuitRef'] == circ_ref, 'circuitId']
        cid = int(cid.values[0]) if len(cid) else np.nan
        races_rows.append({
            'raceId': race_id,
            'year': 2025,
            'round': rnd,
            'circuitId': cid,
            'name': str(r.get('OfficialEventName', r.get('EventName', 'Unknown'))),
            'date': pd.to_datetime(r.get('EventDate', pd.NaT), errors='coerce'),
            'time': pd.NaT
        })
    races_out = pd.DataFrame(races_rows)
    if 'date' in races_out.columns:
        races_out['date'] = pd.to_datetime(races_out['date'], errors='coerce').dt.date.astype(str)
    if 'time' in races_out.columns:
        races_out['time'] = races_out['time'].astype(str)
    else:
        races_out['time'] = ""

    # 3) Tamamlanan yarışların sonuçları
    res_list, driver_rows, team_rows = [], [], []
    rounds = sorted(races_out['round'].dropna().astype(int).tolist())
    for rnd in rounds:
        try:
            session = fastf1.get_session(2025, rnd, 'R')  # Race
            session.load(laps=False, telemetry=False)
        except Exception:
            continue
        if session.results is None or len(session.results) == 0:
            continue
        race_id = 2025 * 100 + rnd
        sr = session.results.copy()
        for _, row in sr.iterrows():
            driver_name = str(row.get('BroadcastName', 'Unknown'))
            drv_abbr = str(row.get('Abbreviation', 'UNK'))
            driver_ref = drv_abbr.lower() if drv_abbr and drv_abbr != 'None' else slugify(driver_name)
            driver_rows.append({
                'driverRef': driver_ref,
                'number'  : row.get('DriverNumber', pd.NA),
                'code'    : drv_abbr if drv_abbr and drv_abbr != 'None' else pd.NA,
                'forename': driver_name.split(' ')[0] if ' ' in driver_name else driver_name,
                'surname' : driver_name.split(' ')[-1] if ' ' in driver_name else driver_name,
                'dob'     : pd.NA,
                'nationality': pd.NA,
                'url'        : pd.NA
            })
            team_name = str(row.get('TeamName', 'Unknown'))
            team_rows.append({
                'constructorRef': slugify(team_name),
                'name'          : team_name,
                'nationality'   : pd.NA,
                'url'           : pd.NA
            })
        for _, row in sr.iterrows():
            driver_name = str(row.get('BroadcastName', 'Unknown'))
            drv_abbr = str(row.get('Abbreviation', 'UNK'))
            driver_ref = drv_abbr.lower() if drv_abbr and drv_abbr != 'None' else slugify(driver_name)
            team_name = str(row.get('TeamName', 'Unknown'))
            constructor_ref = slugify(team_name)
            res_list.append({
                'raceId'       : race_id,
                'driverRef'    : driver_ref,
                'constructorRef': constructor_ref,
                'grid'         : row.get('GridPosition', np.nan),
                'position'     : row.get('Position', np.nan),
                'positionText' : str(row.get('Position', '')),
                'points'       : row.get('Points', 0.0),
                'laps'         : row.get('Laps', np.nan),
                'status'       : row.get('Status', None),
                'time'         : None,
                'milliseconds' : None,
                'fastestLap'   : None, 'rank': None,
                'fastestLapTime': None, 'fastestLapSpeed': None,
                'statusId'     : None
            })

    drivers = pd.DataFrame(driver_rows).drop_duplicates('driverRef') if driver_rows else pd.DataFrame(
        columns=['driverRef','number','code','forename','surname','dob','nationality','url'])
    constructors = pd.DataFrame(team_rows).drop_duplicates('constructorRef') if team_rows else pd.DataFrame(
        columns=['constructorRef','name','nationality','url'])

    if not drivers.empty:
        drivers['driverId'] = pd.factorize(drivers['driverRef'])[0] + 1
    if not constructors.empty:
        constructors['constructorId'] = pd.factorize(constructors['constructorRef'])[0] + 1

    drivers_out = drivers[['driverId','driverRef','number','code','forename','surname','dob','nationality','url']] \
        if not drivers.empty else pd.DataFrame(
            columns=['driverId','driverRef','number','code','forename','surname','dob','nationality','url'])
    constructors_out = constructors[['constructorId','constructorRef','name','nationality','url']] \
        if not constructors.empty else pd.DataFrame(
            columns=['constructorId','constructorRef','name','nationality','url'])

    id_map_drv = drivers_out.set_index('driverRef')['driverId'] if not drivers_out.empty else pd.Series(dtype='int')
    id_map_con = constructors_out.set_index('constructorRef')['constructorId'] if not constructors_out.empty else pd.Series(dtype='int')

    if res_list:
        res = pd.DataFrame(res_list)
        res['driverId'] = res['driverRef'].map(id_map_drv)
        res['constructorId'] = res['constructorRef'].map(id_map_con)
        res = res.merge(pd.DataFrame({'raceId': races_out['raceId'], 'round': races_out['round']}), on='raceId', how='left')
        res = res.sort_values(['raceId','position'])
        results_out = pd.DataFrame({
            'resultId'      : range(1, len(res)+1),
            'raceId'        : res['raceId'],
            'driverId'      : res['driverId'],
            'constructorId' : res['constructorId'],
            'number'        : pd.NA,
            'grid'          : pd.to_numeric(res['grid'], errors='coerce'),
            'position'      : pd.to_numeric(res['position'], errors='coerce'),
            'positionText'  : res['positionText'].astype(str),
            'positionOrder' : pd.NA,
            'points'        : pd.to_numeric(res['points'], errors='coerce'),
            'laps'          : pd.to_numeric(res['laps'], errors='coerce'),
            'time'          : res['time'],
            'milliseconds'  : res['milliseconds'],
            'fastestLap'    : res['fastestLap'],
            'rank'          : res['rank'],
            'fastestLapTime': res['fastestLapTime'],
            'fastestLapSpeed': res['fastestLapSpeed'],
            'statusId'      : pd.NA
        })
    else:
        results_out = pd.DataFrame(columns=[
            'resultId','raceId','driverId','constructorId','number','grid','position','positionText','positionOrder',
            'points','laps','time','milliseconds','fastestLap','rank','fastestLapTime','fastestLapSpeed','statusId'
        ])

    driver_standings_out = pd.DataFrame(columns=['driverStandingsId','raceId','driverId','points','position','positionText','wins'])
    constructor_standings_out = pd.DataFrame(columns=['constructorStandingsId','raceId','constructorId','points','position','positionText','wins'])

    races_out.to_csv(os.path.join(save_dir, 'races.csv'), index=False)
    circuits_out.to_csv(os.path.join(save_dir, 'circuits.csv'), index=False)
    drivers_out.to_csv(os.path.join(save_dir, 'drivers.csv'), index=False)
    constructors_out.to_csv(os.path.join(save_dir, 'constructors.csv'), index=False)
    results_out.to_csv(os.path.join(save_dir, 'results.csv'), index=False)
    driver_standings_out.to_csv(os.path.join(save_dir, 'driver_standings.csv'), index=False)
    constructor_standings_out.to_csv(os.path.join(save_dir, 'constructor_standings.csv'), index=False)

    for fname, cols in [
        ('qualifying.csv',        ['qualifyId','raceId','driverId','constructorId','number','position','q1','q2','q3']),
        ('sprint_results.csv',    ['resultId','raceId','driverId','constructorId','number','grid','position','positionText','positionOrder','points','laps','time','milliseconds','fastestLap','rank','fastestLapTime','fastestLapSpeed','statusId']),
        ('seasons.csv',           ['year','url']),
        ('status.csv',            ['statusId','status']),
        ('lap_times.csv',         ['raceId','driverId','lap','position','time','milliseconds']),
        ('pit_stops.csv',         ['raceId','driverId','stop','lap','time','duration','milliseconds']),
        ('constructor_results.csv',['constructorResultsId','raceId','constructorId','points','status'])
    ]:
        p = os.path.join(save_dir, fname)
        if not os.path.exists(p):
            pd.DataFrame(columns=cols).to_csv(p, index=False)

    print(f"[OK] 2025 FastF1 çıktılarına yazıldı → {save_dir}")

# 2025 Proxy-Grid ile Tam Sezon Projeksiyonu
# =========================================================

def _proxy_grid_from_kaggle_or_results(clean_2025: dict) -> pd.DataFrame:
    """
    Sürücü başına 2025 için PROXY grid değeri üretir.
    Öncelik: Kaggle qualifying (son 3 ort) -> 2025 results'taki grid (son 3 ort)
             -> constructor'a göre takım ortalaması -> 12 (varsayılan)
    Dönen tablo: driverId, constructorId (son görülen), proxy_grid
    """
    res = clean_2025['results'].copy()
    qual = clean_2025['qualifying'].copy()
    # --- sanitize driverId: drop NaN / 0 (0 bazen NaN cast'inden geliyor)
    for _name, _df in [('results', res), ('qualifying', qual)]:
        if isinstance(_df, pd.DataFrame) and not _df.empty and 'driverId' in _df.columns:
            _df['driverId'] = pd.to_numeric(_df['driverId'], errors='coerce')
            _df = _df[_df['driverId'].notna()]
            _df = _df[_df['driverId'] != 0]
            if _name == 'results':
                res = _df
            else:
                qual = _df

    # 1) Qualifying son 3 ort (2025)
    qg = qual.sort_values(['driverId','raceId']).groupby('driverId')['position'].apply(
        lambda s: s.tail(3).mean()
    ).rename('proxy_grid_q').reset_index()

    # 2) Results içindeki 'grid' son 3 ort
    rg = res.sort_values(['driverId','raceId']).groupby('driverId')['grid'].apply(
        lambda s: s.tail(3).mean()
    ).rename('proxy_grid_r').reset_index()

    # 3) Sürücünün son görülen constructorId'si
    last_team = (res.sort_values(['driverId','raceId'])
                   .groupby('driverId')['constructorId']
                   .last().rename('constructorId').reset_index())

    # 4) Takım ortalaması (grid)
    team_avg = (res.groupby('constructorId')['grid'].mean()
                  .rename('team_grid_avg').reset_index())

    proxy = last_team.merge(qg, on='driverId', how='left').merge(rg, on='driverId', how='left') \
                     .merge(team_avg, on='constructorId', how='left')

    # karışım: öncelik sırası (q -> r -> team -> 12)
    proxy['proxy_grid'] = proxy['proxy_grid_q']
    proxy.loc[proxy['proxy_grid'].isna(), 'proxy_grid'] = proxy.loc[proxy['proxy_grid'].isna(), 'proxy_grid_r']
    proxy.loc[proxy['proxy_grid'].isna(), 'proxy_grid'] = proxy.loc[proxy['proxy_grid'].isna(), 'team_grid_avg']
    proxy['proxy_grid'] = proxy['proxy_grid'].fillna(12.0)

    # Güvenli aralık
    proxy['proxy_grid'] = proxy['proxy_grid'].clip(lower=1.0, upper=20.0)
    # Canonical column for projector
    proxy['grid_proxy'] = proxy['proxy_grid']
    proxy = proxy[pd.to_numeric(proxy['driverId'], errors='coerce').notna()]
    proxy = proxy[proxy['driverId'] != 0]
    return proxy[['driverId', 'constructorId', 'grid_proxy']]


def _remaining_rounds_2025(clean_2025: dict) -> list:
    """
    2025'te kalan round listesi.
    'races.csv' teki tüm round'lar - results.csv'de yer alan round'lar.
    """
    races = clean_2025['races'].copy()
    results = clean_2025['results'].copy()
    have_rounds = (races[['raceId','round']]
                   .merge(results[['raceId']].drop_duplicates(), on='raceId', how='inner')
                   ['round'].dropna().astype(int).unique().tolist())
    all_rounds = races['round'].dropna().astype(int).unique().tolist()
    remain = sorted([r for r in all_rounds if r not in have_rounds])
    return remain

def _last_feature_row_per_driver(master_2025: pd.DataFrame, X_2025: pd.DataFrame) -> pd.DataFrame:
    """
    Her sürücü için 2025'teki SON yarışına ait X satırını döndürür.
    (Projeksiyonda bu satırı kopyalayıp sadece grid ve round'ı güncelleyeceğim.)
    Dönen: index=driverId, sütunlar = X kolonları + 'round' (varsa)
    """
    if master_2025.empty or X_2025.empty:
        return pd.DataFrame()
    years = years_series_from_master(master_2025)
    mask = (years == 2025)
    m25 = master_2025.loc[mask].copy()
    X25 = X_2025.loc[mask].copy()

    # Son yarış index'i
    m25 = m25[['driverId','round']].copy()
    m25['round'] = m25['round'].astype(int)
    # X25 ile aynı indexte birleşelim
    tmp = m25.join(X25, how='left')
    tmp = tmp.sort_values(['driverId','round'])
    last_idx = tmp.groupby('driverId').tail(1).copy()
    last_idx = last_idx.set_index('driverId')
    return last_idx  # kolonlar: round + X kolonları

def project_season_2025_with_proxy_grid(master_2025, X_2025, clean_2025, model, proxy_grid_df):
    """
    Proxy-grid kullanarak 2025'in KALAN yarışları için puan projeksiyonu üretir
    ve sezon sonu toplamını (şu ana kadarki gerçek + kalan tahmin) döndürür.

    RETURN:
      season_proj: driverId, driver_name, true_to_date, remaining_pred, proj_total (sıralı)
      per_round_pred: driverId, round, pred_points (kalan turlar için satır bazında)
    """

    # Normalize proxy-grid column name for safety
    if ('grid_proxy' not in proxy_grid_df.columns) and ('proxy_grid' in proxy_grid_df.columns):
        proxy_grid_df = proxy_grid_df.rename(columns={'proxy_grid': 'grid_proxy'})


    # --- Güvenli okuma yardımcıları (clean_2025 sözlük değilse bile çalışsın----) ---
    def _safe_get(df_or_dict, key):
        if isinstance(df_or_dict, dict) and key in df_or_dict:
            return df_or_dict[key]
        return None

    def _read_csv_if_exists(path):
        try:
            if path and os.path.exists(path):
                return pd.read_csv(path)
        except Exception:
            pass
        return pd.DataFrame()

    # Global LIVE2025_DIR varsa yedek olarak kullan
    live_dir = globals().get('LIVE2025_DIR', None)

    races_df   = _safe_get(clean_2025, 'races')
    results_df = _safe_get(clean_2025, 'results')
    if isinstance(results_df, pd.DataFrame) and not results_df.empty and 'driverId' in results_df.columns:
        results_df['driverId'] = pd.to_numeric(results_df['driverId'], errors='coerce')
        results_df = results_df[results_df['driverId'].notna()]
        results_df = results_df[results_df['driverId'] != 0]
    drivers_df = _safe_get(clean_2025, 'drivers')

    # Fallback: CSV'den çekmeye çalış
    if races_df is None or races_df.empty:
        races_df = _read_csv_if_exists(os.path.join(live_dir, 'races.csv') if live_dir else None)
    if results_df is None or (isinstance(results_df, pd.DataFrame) and results_df.empty):
        results_df = _read_csv_if_exists(os.path.join(live_dir, 'results.csv') if live_dir else None)
    if drivers_df is None or (isinstance(drivers_df, pd.DataFrame) and drivers_df.empty):
        drivers_df = _read_csv_if_exists(os.path.join(live_dir, 'drivers.csv') if live_dir else None)

    if races_df is None or races_df.empty:
        raise RuntimeError("Proxy-grid projeksiyonu için 'races' verisi bulunamadı.")

    # --- Kolon hizalama yardımcı ---
    def _align_like(df, cols):
        out = df.copy()
        for c in cols:
            if c not in out.columns:
                out[c] = 0
        extra = [c for c in out.columns if c not in cols]
        if extra:
            out = out.drop(columns=extra)
        return out[cols]

    # --- 1) Kalan turları bul (takvim - oynananlar) ---
    races_25 = races_df.query('year == 2025').copy()
    races_25['round'] = races_25['round'].astype(int)

    played_rounds = pd.Series(dtype=int)
    if isinstance(results_df, pd.DataFrame) and not results_df.empty:
        rr = results_df.dropna(subset=['raceId']).copy()
        if 'raceId' in rr.columns:
            rr['round'] = rr['raceId'].astype(str).str[-2:].astype(int)
            played_rounds = rr['round'].drop_duplicates().sort_values()

    remaining = races_25[~races_25['round'].isin(played_rounds)][['round']].drop_duplicates().sort_values('round')

    # --- 2) Sürücü isim haritası ---
    # 2) Sürücü isim haritası (sağlam)
    name_map = build_driver_name_map_2025(drivers_df if isinstance(drivers_df, pd.DataFrame) else pd.DataFrame())

    # Kalan yarış yoksa: sadece gerçekleşenlerin toplamını dön
    if remaining.empty:
        print("[INFO] 2025 remaining rounds: 0 — sezon projeksiyonu yapmadan 'şu ana kadar' toplamını döndürüyorum.")
        true_to_date = (results_df.groupby('driverId')['points'].sum()
                        .rename('true_to_date').reset_index()) if isinstance(results_df, pd.DataFrame) and not results_df.empty \
                        else pd.DataFrame(columns=['driverId','true_to_date'])
        season = true_to_date.merge(name_map[['driverId','driver_name']], on='driverId', how='left').fillna({'true_to_date': 0})
        season['remaining_pred'] = 0.0
        season['proj_total'] = season['true_to_date']
        season = season.sort_values('proj_total', ascending=False).reset_index(drop=True)
        return season, pd.DataFrame(columns=['driverId','round','pred_points'])

    # --- 3) Sürücü başına 2025'teki SON feature satırını al ---
    last_idx = master_2025.groupby('driverId').tail(1).index
    lastX = X_2025.loc[last_idx].copy()
    # Önce driverId’yi koy, SONRA temizle
    lastX['driverId'] = master_2025.loc[last_idx, 'driverId'].values
    lastX['driverId'] = pd.to_numeric(lastX['driverId'], errors='coerce')
    lastX = lastX[lastX['driverId'].notna()]
    lastX = lastX[lastX['driverId'] != 0]

    # round çakışmasını önlemek için çıkar
    lastX_no_round = lastX.drop(columns=['round'], errors='ignore')

    # --- 4) Sürücüler x kalan turlar kartesyen çarpımı ---
    drivers = lastX[['driverId']].drop_duplicates().reset_index(drop=True)
    rem = remaining.rename(columns={'round': 'round_sched'}).copy()
    drivers['key'] = 1
    rem['key'] = 1
    cart = drivers.merge(rem, on='key').drop(columns='key')  # driverId, round_sched

    # --- 5) Feature'ları sürücüye göre bindir, turu takvimden ata ---
    proj = cart.merge(lastX_no_round, on='driverId', how='left')
    proj['round'] = proj['round_sched'].astype(int)
    proj = proj.drop(columns=['round_sched'])

    # --- 6) Proxy-grid'i bindir ---
    proxy_cols = [c for c in ['driverId', 'grid_proxy'] if c in proxy_grid_df.columns]
    proj = proj.merge(proxy_grid_df[proxy_cols], on='driverId', how='left')
    proj['grid'] = proj['grid_proxy']
    proj = proj.drop(columns=['grid_proxy'], errors='ignore')

    # --- 7) Modele göre kolonları hizala & tahmin et ---
    model_cols = list(getattr(model, 'feature_names_in_', X_2025.columns))
    Xa = _align_like(proj.drop(columns=['driverId'], errors='ignore'), model_cols)
    proj['pred_points'] = model.predict(Xa)

    # --- 8) Sezon toplamı (şu ana kadar gerçek + kalan tahmin) ---
    true_to_date = (results_df.groupby('driverId')['points'].sum()
                    .rename('true_to_date').reset_index()) if isinstance(results_df, pd.DataFrame) and not results_df.empty \
                    else pd.DataFrame(columns=['driverId','true_to_date'])

    season = proj.groupby('driverId', as_index=False)['pred_points'].sum().rename(columns={'pred_points':'remaining_pred'})
    season = season.merge(true_to_date, on='driverId', how='left').fillna({'true_to_date': 0})
    season['proj_total'] = season['remaining_pred'] + season['true_to_date']

    # İsim ekle
    if not name_map.empty and 'driver_name' in name_map.columns:
        season = season.merge(name_map[['driverId','driver_name']], on='driverId', how='left')

    # Final güvenlik: hayalet id ve isimsiz satırları at
    if 'driverId' in season.columns:
        season['driverId'] = pd.to_numeric(season['driverId'], errors='coerce')
        season = season[season['driverId'].notna()]
        season = season[season['driverId'] != 0]
    if 'driver_name' in season.columns:
        season = season[season['driver_name'].notna()]

    season = season.sort_values('proj_total', ascending=False).reset_index(drop=True)
    per_round_pred = proj[['driverId','round','pred_points']].sort_values(['round','pred_points'], ascending=[True, False]).reset_index(drop=True)
    # per-round için de phantom id temizliği
    if isinstance(per_round_pred, pd.DataFrame) and not per_round_pred.empty and 'driverId' in per_round_pred.columns:
        per_round_pred['driverId'] = pd.to_numeric(per_round_pred['driverId'], errors='coerce')
        per_round_pred = per_round_pred[per_round_pred['driverId'].notna()]
        per_round_pred = per_round_pred[per_round_pred['driverId'] != 0]
    return season, per_round_pred

    # -- Finish-line görselleri için CSV kaynakları --


# Monte Carlo
# =========================================================
def monte_carlo_season(season_proj_df: pd.DataFrame, n_sims=5000, sigma=None, seed=42):
    """
    remaining_pred üzerine gürültü ekleyip proj_total dağılımı üretir.
    sigma: None ise son validasyon/test RMSE'sinden birini kullanmak uygundur.
    Döner: summary (expected, p05, p95, win_prob), full_matrix (n_sims x drivers)
    """
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    if not isinstance(season_proj_df, pd.DataFrame) or season_proj_df.empty:
        return pd.DataFrame(columns=['driverId','driver_name','exp_total','p05','p95','win_prob']), None

    df = season_proj_df[['driverId','driver_name','true_to_date','remaining_pred']].copy()
    df['true_to_date']   = pd.to_numeric(df['true_to_date'], errors='coerce').fillna(0.0)
    df['remaining_pred'] = pd.to_numeric(df['remaining_pred'], errors='coerce').fillna(0.0)

    # Varsayılan belirsizlik: 4.5 puan ( test RMSEye uyumlu)
    if sigma is None:
        sigma = 4.5

    # (n_sims, n_drivers) rasgele gürültü
    mu  = df['remaining_pred'].values[None, :]
    eps = rng.normal(loc=0.0, scale=sigma, size=(n_sims, mu.shape[1]))
    rem_sim = np.clip(mu + eps, 0, None)  # negatif puan yok

    total_sim = rem_sim + df['true_to_date'].values[None, :]

    # Özetler
    exp = total_sim.mean(axis=0)
    p05 = np.percentile(total_sim, 5, axis=0)
    p95 = np.percentile(total_sim, 95, axis=0)
    # Şampiyonluk olasılığı: her simde en yüksek toplamı alan sürücüyü say
    max_idx = total_sim.argmax(axis=1)
    win_prob = np.bincount(max_idx, minlength=mu.shape[1]) / n_sims

    summary = pd.DataFrame({
        'driverId': df['driverId'].values,
        'driver_name': df['driver_name'].values,
        'exp_total': exp,
        'p05': p05,
        'p95': p95,
        'win_prob': win_prob
    }).sort_values('exp_total', ascending=False).reset_index(drop=True)

    return summary, total_sim




# =========================================================
# ÇALIŞTIRMA (tek akış)
# =========================================================

def run_all():
    # 0) Yol sabitleri dosyanın üstünde tanımlı: ARCHIVE_DIR, LIVE2025_DIR, (varsa) KAGGLE_2025_DIR
    import os
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import matplotlib.pyplot as plt

    # 1) ARŞİVİ OKU → FEATURE
    all_clean = load_raw_and_clean(ARCHIVE_DIR)
    master, X, y, meta = create_features(
        results=all_clean['results'],
        races=all_clean['races'],
        drivers=all_clean['drivers'],
        constructors=all_clean['constructors'],
        status=all_clean['status'],
        circuits=all_clean['circuits'],
        driver_standings=all_clean['driver_standings'],
        constructor_standings=all_clean['constructor_standings'],
        max_cardinality=40,
        rare_min_count=None,
        add_extra_features=True
    )
    print("master / X / y şekil:", master.shape, X.shape, y.shape)

    # 2) HIZLI ÖNEM KONTROLÜ
    imp, _ = rf_feature_importance(X, y, top_k=20)
    print("\nEn önemli 20 özellik:\n", imp)

    # 3) ZAMAN-BİLİNÇLİ TRAIN/TEST ve MODEL
    X_train, X_test, y_train, y_test, test_years = split_train_test_by_last_n_years(master, X, y, holdout_years=2)
    print(f"\nTest yılları: {test_years}")

    try:
        best_rf = joblib.load('rf_points_model_bestcv.joblib')
        print("best_rf yüklendi.")
    except Exception:
        print("best_rf bulunamadı → yeniden eğitiliyor (800 ağaç, max_features=0.5, min_leaf=5)...")
        best_rf = RandomForestRegressor(
            n_estimators=800, max_depth=None, random_state=42, n_jobs=-1,
            max_features=0.5, min_samples_leaf=5, min_samples_split=2
        )
        best_rf.fit(X_train, y_train)
        joblib.dump(best_rf, 'rf_points_model_bestcv.joblib')

    pred_test = best_rf.predict(X_test)
    print(f"BestRF Test RMSE: {mean_squared_error(y_test, pred_test, squared=False):.3f} | "
          f"MAE: {mean_absolute_error(y_test, pred_test):.3f} | "
          f"R2: {r2_score(y_test, pred_test):.3f}")
    rmse_grid = mean_squared_error(y_test, pred_test, squared=False)
    mae_grid  = mean_absolute_error(y_test, pred_test)

    # İsim haritası (ARŞİV 2023/24 değerlendirmesi için)
    name_map = None
    if isinstance(meta.get('id_to_driver'), pd.DataFrame):
        name_map = meta['id_to_driver'].copy()
        name_map['driver_name'] = (
            name_map['forename'].astype(str) + ' ' + name_map['surname'].astype(str)
        )


    # 4) 2023–2024 SEZON TABLOLARI & TOP-10

    season_test = season_tables_from_preds(master, X, y, best_rf, years=test_years, name_map=name_map)
    if 'true_total' in season_test.columns:
        for yr in test_years:
            stats, pred_top, true_top = topk_overlap_table(season_test, yr, k=10)
            print(f"\n==== {yr} Sezonu Top-10 ====")
            print("Metrics:", stats)
            print("\nTahmin Top-10:\n", pred_top.reset_index(drop=True))
            print("\nGerçek Top-10:\n", true_top.reset_index(drop=True))

    #Ek: Yıl bazında genel metrik özeti (yarış-level RMSE/MAE/R2 + sezon Precision@10/Spearman)
    for yr in test_years:
        try:
            summ = year_level_metrics(master, X, y, best_rf, yr, name_map=name_map)
            print(f"\n[YIL ÖZET {yr}] RMSE: {summ['RMSE']:.3f} | MAE: {summ['MAE']:.3f} | R2: {summ['R2']:.3f} | "
                  f"precision@10: {summ['precision@10']} | spearman: {summ['spearman_season']}")
        except Exception as e:
            print(f"[UYARI] {yr} yıl özeti hesaplanamadı:", e)

    # 4.1) Grid'siz kısa karşılaştırma — ! (Artık X,X_train vb. kapsamda)
    X_train_gridless = X_train.drop(columns=['grid'], errors='ignore')
    X_test_gridless  = X_test.drop(columns=['grid'], errors='ignore')
    rf_gridless = RandomForestRegressor(
        n_estimators=800, max_depth=None, random_state=42, n_jobs=-1,
        max_features=0.5, min_samples_leaf=5, min_samples_split=2
    )
    rf_gridless.fit(X_train_gridless, y_train)
    pred_gridless = rf_gridless.predict(X_test_gridless)
    print(f"\n[Grid'siz] Test RMSE: {mean_squared_error(y_test, pred_gridless, squared=False):.3f} | "
          f"MAE: {mean_absolute_error(y_test, pred_gridless):.3f} | "
          f"R2: {r2_score(y_test, pred_gridless):.3f}")
    rmse_nogrid = mean_squared_error(y_test, pred_gridless, squared=False)
    mae_nogrid  = mean_absolute_error(y_test, pred_gridless)
    grid_compare_metrics = {
        'grid':     {'RMSE': rmse_grid,    'MAE': mae_grid},
        'gridless': {'RMSE': rmse_nogrid,  'MAE': mae_nogrid}
    }
    # === Gridli vs Gridsiz: Top-20 Özellik Önemleri ===
    try:
        # 1) Gridli modelin importanceları
        fi_grid = pd.Series(best_rf.feature_importances_,
                            index=getattr(best_rf, 'feature_names_in_', X.columns)).sort_values(ascending=False).head(
            20)

        # 2) Gridsiz modeli aynı hiperparametrelerle tekrar eğit (sadece grid kolonunu düşür)
        X_ng = X.copy()
        if 'grid' in X_ng.columns:
            X_ng = X_ng.drop(columns=['grid'])
        from sklearn.ensemble import RandomForestRegressor
        rf_ng = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
        rf_ng.fit(X_ng.loc[X_train.index], y_train)  #  train bölünmesi nasıl ise  öyle kullandm
        fi_ng = pd.Series(rf_ng.feature_importances_,
                          index=getattr(rf_ng, 'feature_names_in_', X_ng.columns)).sort_values(ascending=False).head(20)

        # 3) İki ayrı görsel kaydet
        out_dir = os.path.join(LIVE2025_DIR, "figs")
        os.makedirs(out_dir, exist_ok=True)

        plt.figure(figsize=(8, 7))
        fi_grid.sort_values().plot(kind='barh')
        plt.title('Önemler (Gridli) – Top-20')
        plt.xlabel('Önem skoru');
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'fi_top20_gridli.png'), dpi=160);
        plt.close()

        plt.figure(figsize=(8, 7))
        fi_ng.sort_values().plot(kind='barh')
        plt.title('Önemler (Gridsiz) – Top-20')
        plt.xlabel('Önem skoru');
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'fi_top20_gridsiz.png'), dpi=160);
        plt.close()

        #  kıyas grafiği (yanyana)
        cmp = (fi_grid.rename('gridli')
               .to_frame().merge(fi_ng.rename('gridsiz').to_frame(),
                                 left_index=True, right_index=True, how='outer')
               .fillna(0).sort_values('gridli', ascending=True).tail(20))
        plt.figure(figsize=(9, 7))
        cmp.plot(kind='barh');
        plt.title('Önem Kıyas – Gridli vs Gridsiz (Top-20)')
        plt.xlabel('Önem skoru');
        plt.legend();
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'fi_top20_kıyas.png'), dpi=160);
        plt.close()
    except Exception as e:
        print('[WARN] Önem kıyas grafikleri üretilemedi:', e)

    # 5) 2025 VERİSİ (Kaggle varsa içe aktar, yoksa atla)
    try:
        import_f1_2025_kaggle_to_base(KAGGLE_2025_DIR, LIVE2025_DIR, ARCHIVE_DIR)
    except NameError:
        print("[INFO] Kaggle içe aktarma fonksiyonu tanımlı değil; bu adım atlandı.")
    except FileNotFoundError as e:
        print(f"[WARN] Kaggle 2025 dosyaları bulunamadı: {e}")

    # 5.1) 2025 FastF1 → CSV
    fetch_2025_via_fastf1(LIVE2025_DIR)

    # 5.2) 2025’i yükle → feature → sezon tahmini
    clean_2025 = load_raw_and_clean(base_dir=LIVE2025_DIR)
    if ('results' not in clean_2025) or clean_2025['results'].empty:
        print("[UYARI] 2025 'results.csv' boş görünüyor veya hiç veri yok — 2025 projeksiyonu atlanıyor.")
        return


    try:
        export_presentation_figures(


            master=master, X=X, y=y, model=best_rf,
            test_years=test_years,
            season_2025=locals().get('season_2025', None),
            season_2025_proj=locals().get('season_proj', None),
            out_dir=os.path.join(LIVE2025_DIR, "figs"),
            grid_compare_metrics=grid_compare_metrics
        )

        # Ek post-race tarzı grafikler (Effective Qualifying, Pit, Constructor timeline)
        # --- FE odaklı ek görseller ---
        try:
            export_feature_engineering_extra_charts(all_clean, master, out_dir=os.path.join(LIVE2025_DIR, 'figs'))
        except Exception as _e:
            print('[WARN] FE ek grafikler atlandı:', _e)

        # --- FE öncesi / sonrası metrik karşılaştırma grafiği ---
        try:
            export_fe_vs_baseline_metrics_chart(master, X, y, test_years, out_dir=os.path.join(LIVE2025_DIR, 'figs'))
        except Exception as _e:
            print('[WARN] FE metrik karşılaştırma grafiği atlandı:', _e)
        print(f"[OK] Görseller kaydedildi → {os.path.join(LIVE2025_DIR, 'figs')}")
    except Exception as e:
        print('[WARN] Sunum görselleri üretilemedi:', e)
        master_2025, X_2025, y_2025, meta_2025 = (pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float), {})
    else:
        master_2025, X_2025, y_2025, meta_2025 = create_features(
            results=clean_2025['results'],
            races=clean_2025['races'],
            drivers=clean_2025['drivers'],
            constructors=clean_2025['constructors'],
            status=clean_2025['status'],
            circuits=clean_2025['circuits'],
            driver_standings=clean_2025['driver_standings'],
            constructor_standings=clean_2025['constructor_standings'],
            max_cardinality=40,
            rare_min_count=None,
            add_extra_features=True
        )
    print("2025 master/X/y:", master_2025.shape, X_2025.shape, y_2025.shape)
    # Yeni:
    name_map_2025 = build_driver_name_map_2025(
        clean_2025.get('drivers', pd.DataFrame()),
        overrides=MANUAL_NAME_OVERRIDES_2025
    )
    # Sadece drivers.csv boşsa meta'dan doldur
    if isinstance(meta_2025.get('id_to_driver'), pd.DataFrame) and name_map_2025.empty:
        name_map_2025 = meta_2025['id_to_driver'].copy()
        name_map_2025['driver_name'] = (
                name_map_2025['forename'].astype(str) + ' ' + name_map_2025['surname'].astype(str)
        )

    if not X_2025.empty:
        # Eğitimdeki kolon şemasını referans al
        cols_train = meta.get('columns_X', X.columns.tolist())
        # 2025 X'ini eğitim şemasına hizala (eksikler: 0, fazlalar: drop)
        missing = [c for c in cols_train if c not in X_2025.columns]
        extra   = [c for c in X_2025.columns if c not in cols_train]
        if missing or extra:
            print(f"[INFO] 2025 align: +{len(missing)} eksik kolonu 0 ile doldurduk, -{len(extra)} fazlalığı attık.")
        X_2025_aligned = align_columns_for_inference(X_2025, cols_train)

        season_2025 = season_tables_from_preds(master_2025, X_2025_aligned, y_2025, best_rf, years=[2025], name_map=name_map_2025)
        print("\n=== 2025 Sezonu (şu ana kadar) – Tahmin Top-10 ===")
        print(season_2025[season_2025['year']==2025].nsmallest(10, 'rank_pred')
              [['driver_name','pred_total','rank_pred']].reset_index(drop=True))
        if 'true_total' in season_2025.columns:
            print("\n=== 2025 Sezonu (şu ana kadar) – Gerçek Top-10 ===")
            print(season_2025[season_2025['year']==2025].nsmallest(10, 'rank_true')
                  [['driver_name','true_total','rank_true']].reset_index(drop=True))
            # Self-check: true_total vs results.csv toplamı
            try:
                sub_2025 = season_2025[season_2025['year']==2025][['driverId','true_total']].copy()
                self_check_to_date_vs_results(sub_2025, clean_2025, id_col='driverId', true_col='true_total')
            except Exception as e:
                print('[CHECK] 2025 to-date self-check hata:', e)
            # Eksik isim varsa teşhis et (log'a yaz)
            try:
                if isinstance(season_2025, pd.DataFrame) \
                   and ('driver_name' in season_2025.columns) \
                   and season_2025['driver_name'].isna().any():
                    debug_missing_driver_names(season_2025, clean_2025)
            except Exception as _e:
                print('[DEBUG] isim debug çağrısı hatası:', _e)

            # 5.3) PROXY-GRID ile TAM SEZON PROJEKSİYONU
            try:
                # Proxy-grid üret (tamamlanan yarışlardan gerçek grid, geleceklere sezonsal grid medyanı)
                proxy_grid_df = _proxy_grid_from_kaggle_or_results(clean_2025)

                # PROJEKSİYON: keyword argümanlarla çağır → sıra karışmasın
                season_proj, per_round_pred = project_season_2025_with_proxy_grid(
                    master_2025=master_2025,
                    X_2025=X_2025_aligned,
                    clean_2025=clean_2025,
                    model=best_rf,  # eğittiği/yüklediğim model
                    proxy_grid_df=proxy_grid_df  # biraz önce ürettiğimiz proxy grid DF
                )

                # Sıralama kolonlarını ekle (projeksiyon toplamına göre)
                season_proj = season_proj.copy()
                if 'proj_total' in season_proj.columns:
                    season_proj['rank_proj'] = season_proj['proj_total'].rank(ascending=False, method='min')
                if 'true_to_date' in season_proj.columns and 'rank_true' not in season_proj.columns:
                    season_proj['rank_true'] = season_proj['true_to_date'].rank(ascending=False, method='min')

                # Raporla
                print("\n=== 2025 Tam Sezon – Proxy-Grid Projeksiyonu (İlk 15) ===")
                cols_show = ['rank_proj', 'driver_name', 'true_to_date', 'remaining_pred', 'proj_total']
                cols_show = [c for c in cols_show if c in season_proj.columns]
                print(season_proj[cols_show].head(15).reset_index(drop=True))
                # Kaydet: finish-line animasyonu için girdiler
                import os as _os
                _csv_root = LIVE2025_DIR if 'LIVE2025_DIR' in globals() else _os.getcwd()
                season_proj.to_csv(_os.path.join(_csv_root, "season_2025_proj.csv"), index=False)
                per_round_pred.to_csv(_os.path.join(_csv_root, "per_round_pred.csv"), index=False)
                print(f"[OK] Finish-line girdileri yazıldı → {_csv_root}/season_2025_proj.csv & per_round_pred.csv")


                # --- Monte Carlo: sezon belirsizliği (beklenen + %90 bandı + şampiyonluk olasılığı) ---
                try:
                    if isinstance(season_proj, pd.DataFrame) and not season_proj.empty:
                        mc_summary, _ = monte_carlo_season(season_proj, n_sims=5000, sigma=4.5)
                        print("\n=== 2025 Monte Carlo Özet (ilk 10) ===")
                        print(mc_summary.head(10))

                        # Görsel: Top-10 beklenen toplam ve %90 güven bandı
                        figs_root = globals().get('LIVE2025_DIR', os.getcwd())
                        out_dir = os.path.join(figs_root, 'figs')
                        os.makedirs(out_dir, exist_ok=True)
                        top10 = mc_summary.head(10).copy()
                        labels = top10.get('driver_name', top10['driverId']).astype(str).tolist()

                        import matplotlib.pyplot as plt
                        import numpy as np
                        plt.figure(figsize=(9, 5))
                        plt.bar(range(len(top10)), top10['exp_total'].values)
                        # Error bar: p05–p95
                        yerr_lower = top10['exp_total'].values - top10['p05'].values
                        yerr_upper = top10['p95'].values - top10['exp_total'].values
                        plt.errorbar(range(len(top10)), top10['exp_total'].values,
                                     yerr=[yerr_lower, yerr_upper], fmt='none', capsize=4)
                        plt.xticks(range(len(top10)), labels, rotation=45, ha='right')
                        plt.ylabel('Beklenen Toplam (±90% CI)')
                        plt.title('2025 Monte Carlo Projeksiyonu – Top-10')
                        plt.tight_layout()
                        plt.savefig(os.path.join(out_dir, 'fig_2025_mc_top10.png'), dpi=160)
                        plt.close()
                except Exception as e:
                    print('[WARN] Monte Carlo üretilemedi:', e)

                # Self-check: true_to_date column consistency
                try:
                    self_check_to_date_vs_results(season_proj[['driverId','true_to_date']], clean_2025,
                                                  id_col='driverId', true_col='true_to_date')
                except Exception as e:
                    print('[CHECK] projeksiyon self-check hata:', e)

                # Sunum için figürleri dışa aktar------çoğu olmadı
                try:
                    figs_dir = os.path.join(LIVE2025_DIR, 'figs') if 'LIVE2025_DIR' in globals() else 'figs'
                    export_presentation_figures(master, X, y, best_rf, test_years,
                                                season_2025=season_2025, season_2025_proj=season_proj,
                                                out_dir=figs_dir,
                                                grid_compare_metrics=grid_compare_metrics)
                    try:
                        export_postrace_charts(clean_2025, figs_dir, year=2025)
                    except Exception as e:
                        print('[UYARI] Post-race görseller üretilemedi:', e)
                    print(f"[OK] Görseller kaydedildi → {figs_dir}")
                except Exception as e:
                    print('[UYARI] Görseller üretilemedi:', e)

                # Bugüne kadarki kısım için metrik (proxy sonrasında da değişmez — geleceğin gerçeği yok)
                try:
                    season_to_date = season_tables_from_preds(
                        master_2025, X_2025_aligned, y_2025, best_rf, years=[2025], name_map=name_map_2025
                    )
                    stats_2025, _, _ = topk_overlap_table(season_to_date, 2025, k=10)
                    print("\n[METRİK | 2025 bugüne kadar] precision@10:", stats_2025['precision@10'],
                          "| spearman_season:", stats_2025['spearman_season'])
                except Exception as e:
                    print("[UYARI] 2025 bugüne kadarki metrikler hesaplanamadı:", e)
            except Exception as e:
                print("[UYARI] Proxy-grid projeksiyonu başarısız:", e)
    else:
        print("[BİLGİ] 2025 feature tablosu boş; sezon çıktısı üretilemedi.")

if __name__ == "__main__":
    run_all()