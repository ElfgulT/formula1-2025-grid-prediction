import pandas as pd
import numpy as np
import os
from pathlib import Path

def build_driver_name_map_2025(drivers_df: pd.DataFrame, overrides: dict | None = None) -> pd.DataFrame:
    """2025 için basit driverId -> driver_name haritası üretir."""
    overrides = overrides or {}
    if not isinstance(drivers_df, pd.DataFrame) or drivers_df.empty:
        return pd.DataFrame(columns=['driverId', 'driver_name'])

    df = drivers_df.copy()
    cols = df.columns

    # İsim oluşturma (Ergast/Kaggle ortak kolonlarına göre)
    if {'forename', 'surname'}.issubset(cols):
        name = df['forename'].astype(str) + ' ' + df['surname'].astype(str)
    elif {'givenName', 'familyName'}.issubset(cols):
        name = df['givenName'].astype(str) + ' ' + df['familyName'].astype(str)
    elif 'name' in cols:
        name = df['name'].astype(str)
    else:
        name = df.get('driverRef', df.get('code', df.index.astype(str))).astype(str)

    # driverId yoksa index’i kullanırız
    if 'driverId' in cols:
        driver_id = df['driverId']
    else:
        driver_id = pd.Series(range(len(df)), index=df.index)

    out = pd.DataFrame({'driverId': driver_id, 'driver_name': name})

    # Manuel override’lar (örn: {'max_verstappen': 'Max Verstappen'})
    for key, val in overrides.items():
        mask = (
            (df.get('driverRef', pd.Series(False, index=df.index)) == key) |
            (df.get('code', pd.Series(False, index=df.index)) == key) |
            (out['driverId'].astype(str) == str(key))
        )
        out.loc[mask, 'driver_name'] = val

    return out


def import_f1_2025_kaggle_to_base(kaggle_dir: str | Path, live_dir: str | Path, archive_dir: str | Path) -> None:
    """Kaggle 2025 CSV’lerini varsa live klasörüne kopyalar; yoksa sessizce geçer."""
    kaggle_dir = Path(kaggle_dir)
    live_dir = Path(live_dir)
    live_dir.mkdir(parents=True, exist_ok=True)

    wanted = [
        'results.csv', 'races.csv', 'drivers.csv', 'constructors.csv',
        'status.csv', 'circuits.csv', 'driver_standings.csv', 'constructor_standings.csv'
    ]
    copied = 0
    for fn in wanted:
        src = kaggle_dir / fn
        if src.exists():
            (live_dir / fn).write_bytes(src.read_bytes())
            copied += 1
    print(f"[KAGGLE] {copied} dosya kopyalandı (var olanlar).")


def fetch_2025_via_fastf1(out_dir: str | Path) -> None:
    """FastF1 çekimi için şimdilik no-op; klasörü hazırlar ve bilgi basar."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[FastF1] (stub) Veri çekimi atlandı.")

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


def clean_dataset(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    df = df.copy()
    df = df.replace('\\N', np.nan)
    name = dataset_name.lower().strip()

    # --- GENEL: sayı olması muhtemel bazı kolonları numeriğe zorla (NaN'a güvenle düşsün) ---
    for col in ['duration_seconds', 'milliseconds', 'millis', 'ms']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

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
        # Kategorik ID'ler
        for col in ['raceId','driverId']:
            if col in df.columns:
                df[col] = df[col].astype('category')
        # Tip güvenliği
        for col in ['stop','lap','milliseconds']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # duration_seconds üret (duration sayıysa direkt al, değilse parser ile saniyeye çevir)
        if 'duration' in df.columns:
            # önce sayıya zorla
            duration_num = pd.to_numeric(df['duration'], errors='coerce')
            # sayı olmayanları ham string olarak bırakıp parser'a ver
            duration_sec_parsed = df['duration'].where(duration_num.notna(), df['duration']).apply(parse_duration_to_seconds)
            df['duration_seconds'] = np.where(duration_num.notna(), duration_num, duration_sec_parsed)
        else:
            if 'duration_seconds' not in df.columns:
                df['duration_seconds'] = np.nan
            # varsa bile numeriğe zorla (üstteki genel blok da yapıyor ama güvence)
            df['duration_seconds'] = pd.to_numeric(df['duration_seconds'], errors='coerce')

        # milliseconds <-> duration_seconds karşılıklı doldurma
        if 'milliseconds' in df.columns:
            # duration_seconds boş ama milliseconds var → seconds türet
            mask_dur = df['duration_seconds'].isna() & df['milliseconds'].notna()
            if mask_dur.any():
                df.loc[mask_dur, 'duration_seconds'] = pd.to_numeric(df.loc[mask_dur, 'milliseconds'], errors='coerce') / 1000.0

            # milliseconds boş ama duration_seconds var → ms türet
            mask_ms = df['milliseconds'].isna() & df['duration_seconds'].notna()
            if mask_ms.any():
                # Burada dtype kesin numeric; sonra Int64 (nullable) yapıyoruz
                secs = pd.to_numeric(df.loc[mask_ms, 'duration_seconds'], errors='coerce')
                df.loc[mask_ms, 'milliseconds'] = (secs * 1000.0).round()

            # Son tip güvenliği
            df['milliseconds'] = pd.to_numeric(df['milliseconds'], errors='coerce').astype('Int64')

        # stop/lap da güvenli tiplere çekilsin
        for c in ['lap', 'stop']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int64')

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

def self_check_to_date_vs_results(season_sub: pd.DataFrame, clean_2025: dict,
                                  id_col: str = 'driverId', true_col: str = 'true_to_date'):
    """Lightweight consistency check placeholder."""
    try:
        if isinstance(season_sub, pd.DataFrame):
            _ = season_sub[[id_col, true_col]]  # ensure columns exist
        print("[CHECK] self_check_to_date_vs_results: OK (stub)")
    except Exception as e:
        print("[CHECK] self_check_to_date_vs_results: WARN (stub)", e)


def debug_missing_driver_names(season_df: pd.DataFrame, clean_2025: dict):
    """Placeholder debug routine; prints rows with missing names."""
    if isinstance(season_df, pd.DataFrame) and 'driver_name' in season_df.columns:
        miss = season_df['driver_name'].isna().sum()
        if miss:
            print(f"[DEBUG] {miss} missing driver_name rows (stub)")



