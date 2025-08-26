import pandas as pd
import os
from f1proj.data_io import build_driver_name_map_2025

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
