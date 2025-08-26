# 2023–2024 sezon tablosu ve top-k kıyas yardımcıları
# =========================================================


import pandas as pd

def years_series_from_master(master_df: pd.DataFrame) -> pd.Series:
    """master tablosundan yıl serisini (Int64) döndürür; datetime/str/int hepsini destekler."""
    if 'year' in master_df.columns:
        ser = master_df['year']
    elif 'date' in master_df.columns:
        ser = pd.to_datetime(master_df['date'], errors='coerce').dt.year
        return ser.astype('Int64')

    # year kolonu varsa:
    if str(ser.dtype).startswith('datetime64'):
        return ser.dt.year.astype('Int64')

    # önce datetime'a zorla (string/object olabilir)
    ser_dt = pd.to_datetime(ser, errors='coerce').dt.year
    if ser_dt.notna().any():
        return ser_dt.astype('Int64')

    # son çare: direkt Int64'e çevir
    return ser.astype('Int64')
def season_tables_from_preds(master_df, X_df, y_true, model, years, name_map=None):
    """
    Per-race tahminlerden sezon toplamını üretir ve
    'pred_total' (tahmini sezon puanı) ile 'rank_pred' (sıra) kolonlarını verir.

    Dönüş kolonları en az şunları içerir:
    ['year','driverId','pred_total','rank_pred', 'true_total'(ops.), 'rank_true'(ops.), 'driver_name'(ops.)]
    """
    import pandas as pd
    import numpy as np

    if 'year' not in master_df.columns:
        raise KeyError("season_tables_from_preds: 'year' column missing in master_df")

    # --- master 'year' -> int
    year_series = master_df['year']
    if str(year_series.dtype).startswith('datetime64'):
        year_int = year_series.dt.year.astype('Int64')
    else:
        try:
            year_int = year_series.astype('Int64')
        except Exception:
            year_int = pd.to_datetime(year_series, errors='coerce').dt.year.astype('Int64')

    # --- years arg -> [int]
    def _to_int_year_list(ys):
        out = []
        seq = ys if isinstance(ys, (list, tuple, pd.Series)) else [ys]
        for y in seq:
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
    if not years_int:
        avail = pd.Series(year_int).dropna().astype(int)
        years_int = sorted(avail.unique())[-2:] if not avail.empty else []

    # --- satırları topla
    rows = []
    for yy in years_int:
        # Year mask → positional boolean (index mis-match'lerine dayanıklı)
        mask = (year_int == yy)
        mask_bool = mask.fillna(False).to_numpy()

        n = int(mask_bool.sum())
        if n == 0:
            print(f"[UYARI] {yy} yılı için satır bulunamadı, atlanıyor.")
            continue

        X_year = X_df.iloc[mask_bool, :]
        if X_year.shape[0] == 0:
            print(f"[UYARI] {yy} yılı için özellik matrisi boş (index hizasızlığı olabilir), atlanıyor.")
            continue

        try:
            y_pred = model.predict(X_year)
        except Exception as e:
            print(f"[UYARI] {yy} yılı tahmini yapılamadı: {e}")
            continue

        tmp = master_df.iloc[mask_bool, :][['driverId']].copy()
        tmp['year'] = int(yy)
        tmp['y_pred'] = pd.to_numeric(y_pred, errors='coerce')
        if y_true is not None and len(y_true) == len(master_df):
            tmp['y_true'] = pd.to_numeric(y_true.iloc[mask_bool], errors='coerce')
        rows.append(tmp)

    if not rows:
        # beklenen kolonlarla boş dön
        return pd.DataFrame(columns=['year','driverId','pred_total','rank_pred','true_total','rank_true','driver_name'])

    per_race = pd.concat(rows, ignore_index=True)

    # --- (year, driver) seviyesinde topla
    grp = per_race.groupby(['year','driverId'], sort=False, observed=False)
    season = grp['y_pred'].sum(min_count=1).rename('pred_total').to_frame()

    if 'y_true' in per_race.columns:
        season['true_total'] = grp['y_true'].sum(min_count=1)

    # --- sıralamalar (büyük puan daha iyi)
    season['rank_pred'] = season.groupby('year')['pred_total'].rank(method='dense', ascending=False)
    if 'true_total' in season.columns:
        season['rank_true'] = season.groupby('year')['true_total'].rank(method='dense', ascending=False)

    season = season.reset_index()

    # --- isimler (opsiyonel)
    if name_map is not None and isinstance(name_map, (pd.DataFrame,)):
        cols = {c.lower(): c for c in name_map.columns}
        id_col = cols.get('driverid', 'driverId')
        name_col = cols.get('driver_name', None) or cols.get('name', None)
        if name_col is not None and id_col in name_map.columns:
            season = season.merge(
                name_map[[id_col, name_col]].rename(columns={id_col: 'driverId', name_col: 'driver_name'}),
                on='driverId', how='left')

    # int karşılaştırmaları için
    season['year'] = season['year'].astype(int)

    return season

# --- Helper: robust year to int (for Timestamp/str/int)
def _to_int_year_value(y):
    """Tek bir yıl değerini güvenle int'e çevirir (Timestamp/str/int destekler)."""
    if hasattr(y, 'year'):
        return int(y.year)
    try:
        return int(y)
    except Exception:
        y_dt = pd.to_datetime(y, errors='coerce')
        return int(y_dt.year) if pd.notna(y_dt) else None

def topk_overlap_table(season_df, year, k=10):
    yval = _to_int_year_value(year)
    sub = season_df[season_df['year'] == yval].copy()

    import numpy as np
    if sub.empty:
        return (
            {'year': yval, 'precision@10': 0.0, 'overlap': 0, 'spearman_season': float('nan')},
            pd.DataFrame(columns=['driver_name','pred_total','rank_pred']),
            pd.DataFrame(columns=['driver_name','true_total','rank_true'])
        )

    pred_top = sub.nsmallest(k, 'rank_pred') if 'rank_pred' in sub.columns else pd.DataFrame(columns=['driverId','pred_total','rank_pred'])

    if 'rank_true' not in sub.columns:
        return (
            {'year': yval, 'precision@10': 0.0, 'overlap': 0, 'spearman_season': float('nan')},
            pred_top[['driver_name','pred_total','rank_pred']] if 'driver_name' in pred_top.columns else pred_top,
            pd.DataFrame(columns=['driver_name','true_total','rank_true'])
        )

    true_top = sub.nsmallest(k, 'rank_true')

    pred_set = set(pred_top.get('driverId', pd.Series(dtype=object)))
    true_set = set(true_top.get('driverId', pd.Series(dtype=object)))
    overlap  = pred_set & true_set
    prec_at_k = len(overlap) / max(k, 1)

    from scipy.stats import spearmanr
    sp = spearmanr(sub['rank_true'], sub['rank_pred']).correlation if {'rank_true','rank_pred'}.issubset(sub.columns) else float('nan')

    show_pred = pred_top[['driver_name','pred_total','rank_pred']] if 'driver_name' in pred_top.columns else pred_top[['driverId','pred_total','rank_pred']]
    show_true = true_top[['driver_name','true_total','rank_true']] if 'driver_name' in true_top.columns else true_top[['driverId','true_total','rank_true']]

    return {
        'year': yval,
        'precision@10': round(prec_at_k, 3),
        'overlap': len(overlap),
        'spearman_season': round(float(sp), 3) if sp == sp else float('nan')
    }, show_pred, show_true

def year_level_metrics(master_df: pd.DataFrame, X_df: pd.DataFrame, y_ser: pd.Series,
                       model, year: int, name_map: pd.DataFrame | None = None):
    """Belirli bir yıl için: (1) yarış-level RMSE/MAE/R2, (2) sezon precision@10 & Spearman döndür."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    yval = _to_int_year_value(year)
    years_ser = years_series_from_master(master_df)  # Int64 seri
    mask_bool = (years_ser == yval).fillna(False).to_numpy()

    default_stats = {'year': yval, 'RMSE': float('nan'), 'MAE': float('nan'), 'R2': float('nan')}

    if mask_bool.sum() == 0:
        print(f"[UYARI] {year} yılı için satır bulunamadı; yarış-level metrikler atlandı.")
        season = season_tables_from_preds(master_df, X_df, y_ser, model, years=[yval], name_map=name_map)
        stats, _, _ = topk_overlap_table(season, yval, k=10)
        out = dict(default_stats)
        out.update(stats)
        return out

    # Tahmin denemesi (hata olursa yıl özetini yine döndür)
    try:
        X_year = X_df.iloc[mask_bool, :]
        y_year = y_ser.iloc[mask_bool]
        y_pred = model.predict(X_year)
    except Exception as e:
        print(f"[UYARI] {year} yılı tahmin sırasında hata: {e}")
        season = season_tables_from_preds(master_df, X_df, y_ser, model, years=[yval], name_map=name_map)
        stats, _, _ = topk_overlap_table(season, yval, k=10)
        out = dict(default_stats)
        out.update(stats)
        return out

    # --- Metrikler (eski sklearn uyumluluğu için squared=False fallback)
    try:
        rmse = mean_squared_error(y_year, y_pred, squared=False)
    except TypeError:
        rmse = mean_squared_error(y_year, y_pred) ** 0.5

    mae = mean_absolute_error(y_year, y_pred)
    r2  = r2_score(y_year, y_pred)

    # Sezon top-k karşılaştırmaları (precision@10, spearman)
    season = season_tables_from_preds(master_df, X_df, y_ser, model, years=[yval], name_map=name_map)
    stats, _, _ = topk_overlap_table(season, yval, k=10)

    out = dict(year=yval, RMSE=float(rmse), MAE=float(mae), R2=float(r2))
    out.update(stats)
    return out

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


def self_check_to_date_vs_results(season_sub: pd.DataFrame, clean_2025: dict,
                                  id_col: str = 'driverId', true_col: str = 'true_to_date'):
    try:
        _ = season_sub[[id_col, true_col]]
        print("[CHECK] self_check_to_date_vs_results: OK (stub)")
    except Exception as e:
        print("[CHECK] self_check_to_date_vs_results: WARN (stub)", e)

def debug_missing_driver_names(season_df: pd.DataFrame, clean_2025: dict):
    if isinstance(season_df, pd.DataFrame) and 'driver_name' in season_df.columns:
        miss = season_df['driver_name'].isna().sum()
        if miss:
            print(f"[DEBUG] {miss} missing driver_name rows (stub)")