from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from pandas.api.types import (
    is_categorical_dtype,
    is_object_dtype,
    is_numeric_dtype
)

USE_GRID_FEATURE = True
from sklearn.impute import KNNImputer, SimpleImputer


def season_tables_from_preds(master_df, X_df, y_true, model, years, name_map=None):
    # --- Robust year filtering: support datetime64, strings, and ints ---
    if 'year' not in master_df.columns:
        raise KeyError("season_tables_from_preds: 'year' column missing in master_df")

    year_series = master_df['year']

    # Convert the master_df['year'] column to integer years
    if str(year_series.dtype).startswith('datetime64'):
        year_int = year_series.dt.year
    else:
        try:
            year_int = year_series.astype('int64')
        except Exception:
            # Fallback for string/object mixed types
            year_int = pd.to_datetime(year_series, errors='coerce').dt.year.astype('Int64')

    # Normalize the incoming `years` argument to a list of plain ints
    def _to_int_year_list(ys):
        out = []
        for y in (ys if isinstance(ys, (list, tuple, pd.Series)) else [ys]):
            if hasattr(y, 'year'):
                out.append(int(y.year))
            else:
                try:
                    out.append(int(y))
                except Exception:
                    y_dt = pd.to_datetime(y, errors='coerce')
                    if pd.notna(y_dt):
                        out.append(int(y_dt.year))
        return [int(x) for x in out if pd.notna(x)]

    years_int = _to_int_year_list(years)
    mask = year_int.isin(years_int)

    # Defensive fallback to avoid empty selections during prediction
    if getattr(mask, "sum", lambda: 0)() == 0:
        # pick the latest available year in the data
        last_available = int(pd.Series(year_int).dropna().max())
        print(f"[WARN] No rows for requested years={years} (normalized={years_int}). "
              f"Falling back to last available year: {last_available}")
        mask = (year_int == last_available)

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
    return df.groupby(by, sort=False, observed=False)[col].transform(
        lambda s: s.shift().rolling(window, min_periods=min_periods).mean()
    )

def safe_group_shift_std(df, by, col, window=5, min_periods=2):
    return df.groupby(by, sort=False, observed=False)[col].transform(
        lambda s: s.shift().rolling(window, min_periods=min_periods).std()
    )

def safe_group_shift_expanding_mean(df, by, col):
    return df.groupby(by, sort=False, observed=False)[col].transform(
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
        master['combined_circuit_affinity'] = (
            pd.concat([
                master.get('driver_circuit_avg_points'),
                master.get('constructor_circuit_avg_points')
            ], axis=1)
            .mean(axis=1, skipna=True)
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


def align_columns_for_inference(X_new, train_columns):
    """
    Eğitimdeki kolon şemasını referans alarak, yeni X'i hizalar.
    - Eksik kolonları 0 ile ekler
    - Fazla kolonları düşürür
    - Kolon sırasını eğitim sırasına göre ayarlar
    """
    import pandas as _pd

    if X_new is None or len(getattr(X_new, "columns", [])) == 0:
        return X_new

    Xc = X_new.copy()
    # Eksikleri ekle
    missing = [c for c in train_columns if c not in Xc.columns]
    for c in missing:
        Xc[c] = 0
    # Fazlaları at
    extra = [c for c in Xc.columns if c not in train_columns]
    if extra:
        Xc = Xc.drop(columns=extra, errors="ignore")
    # Sıra
    Xc = Xc.reindex(columns=train_columns)
    # NaN güvenliği
    Xc = Xc.fillna(0)
    return Xc