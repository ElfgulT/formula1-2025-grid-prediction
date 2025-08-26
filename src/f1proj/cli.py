import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

from f1proj.config import CFG
from f1proj.data_io import (
    load_raw_and_clean,
    build_driver_name_map_2025,
    import_f1_2025_kaggle_to_base,
    fetch_2025_via_fastf1,
)
from f1proj.fe import create_features, align_columns_for_inference
from f1proj.modeling import rf_feature_importance, split_train_test_by_last_n_years
from f1proj.eval import (
    season_tables_from_preds,
    topk_overlap_table,
    year_level_metrics,
    self_check_to_date_vs_results,
    debug_missing_driver_names,
)
from f1proj.viz import (
    export_presentation_figures,
    export_feature_engineering_extra_charts,
    export_fe_vs_baseline_metrics_chart,
    export_postrace_charts,
)
from f1proj.simulate import (
    _proxy_grid_from_kaggle_or_results,
    project_season_2025_with_proxy_grid,
    monte_carlo_season,
)
def run_all(skip_kaggle=False, skip_fastf1=False, holdout_years=2, random_state=42):
    """
    Uçtan-uca: arşivi oku -> FE -> RF eğit -> 2023–2024 test et -> gridsiz ablation ->
    2025'i içe aktar (Kaggle/FastF1) -> 2025 FE -> 2025 'to-date' metrik/tablolar ->
    Proxy-Grid ile kalan sezon projeksiyonu -> Monte Carlo -> figürler/tablolar.
    """
    ARCHIVE_DIR = CFG.ARCHIVE_DIR
    LIVE2025_DIR = CFG.LIVE2025_DIR
    KAGGLE_2025_DIR = getattr(CFG, "KAGGLE_2025_DIR", None)

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

    # 2) Hızlı önem kontrolü
    imp, _ = rf_feature_importance(X, y, top_k=20)
    print("\nEn önemli 20 özellik:\n", imp)

    # 3) Zaman-duyarlı train/test ve model
    X_train, X_test, y_train, y_test, test_years = split_train_test_by_last_n_years(
        master, X, y, holdout_years=holdout_years
    )
    print(f"\nTest yılları: {test_years}")

    try:
        best_rf = joblib.load('rf_points_model_bestcv.joblib')
        print("best_rf yüklendi.")
    except Exception:
        print("best_rf bulunamadı → yeniden eğitiliyor (800 ağaç, max_features=0.5, min_leaf=5)...")
        best_rf = RandomForestRegressor(
            n_estimators=800, max_depth=None, random_state=random_state, n_jobs=-1,
            max_features=0.5, min_samples_leaf=5, min_samples_split=2
        )
        best_rf.fit(X_train, y_train)
        joblib.dump(best_rf, 'rf_points_model_bestcv.joblib')

    pred_test = best_rf.predict(X_test)

    def _rmse(y_true, y_pred):
        try:
            return mean_squared_error(y_true, y_pred, squared=False)
        except TypeError:
            return np.sqrt(mean_squared_error(y_true, y_pred))

    print(
        f"BestRF Test RMSE: {_rmse(y_test, pred_test):.3f} | "
        f"MAE: {mean_absolute_error(y_test, pred_test):.3f} | "
        f"R2: {r2_score(y_test, pred_test):.3f}"
    )
    rmse_grid = _rmse(y_test, pred_test)
    mae_grid  = mean_absolute_error(y_test, pred_test)

    # İsim haritası (2023/24 değerlendirmesi için)
    name_map = None
    if isinstance(meta.get('id_to_driver'), pd.DataFrame):
        name_map = meta['id_to_driver'].copy()
        name_map['driver_name'] = (
            name_map['forename'].astype(str) + ' ' + name_map['surname'].astype(str)
        )

    # 4) 2023–2024 sezon tabloları & Top-10
    season_test = season_tables_from_preds(master, X, y, best_rf, years=test_years, name_map=name_map)
    if 'true_total' in season_test.columns:
        for yr in test_years:
            stats, pred_top, true_top = topk_overlap_table(season_test, yr, k=10)
            print(f"\n==== {yr} Sezonu Top-10 ====")
            print("Metrics:", stats)
            print("\nTahmin Top-10:\n", pred_top.reset_index(drop=True))
            print("\nGerçek Top-10:\n", true_top.reset_index(drop=True))

    # Yıl bazında özet metrikler
    for yr in test_years:
        try:
            summ = year_level_metrics(master, X, y, best_rf, yr, name_map=name_map)
            print(f"\n[YIL ÖZET {yr}] RMSE: {summ['RMSE']:.3f} | MAE: {summ['MAE']:.3f} | R2: {summ['R2']:.3f} | "
                  f"precision@10: {summ['precision@10']} | spearman: {summ['spearman_season']}")
        except Exception as e:
            print(f"[UYARI] {yr} yıl özeti hesaplanamadı:", e)

    # 4.1) Grid ablation (gridsiz kıyas)
    X_train_gridless = X_train.drop(columns=['grid'], errors='ignore')
    X_test_gridless  = X_test.drop(columns=['grid'], errors='ignore')
    rf_gridless = RandomForestRegressor(
        n_estimators=800, max_depth=None, random_state=random_state, n_jobs=-1,
        max_features=0.5, min_samples_leaf=5, min_samples_split=2
    )
    rf_gridless.fit(X_train_gridless, y_train)
    pred_gridless = rf_gridless.predict(X_test_gridless)
    print(
        f"\n[Grid'siz] Test RMSE: {_rmse(y_test, pred_gridless):.3f} | "
        f"MAE: {mean_absolute_error(y_test, pred_gridless):.3f} | "
        f"R2: {r2_score(y_test, pred_gridless):.3f}"
    )
    rmse_nogrid = _rmse(y_test, pred_gridless)
    mae_nogrid  = mean_absolute_error(y_test, pred_gridless)
    grid_compare_metrics = {'grid': {'RMSE': rmse_grid, 'MAE': mae_grid},
                            'gridless': {'RMSE': rmse_nogrid, 'MAE': mae_nogrid}}

    # Önem kıyas figürleri (gridli vs gridsiz)
    try:
        fi_grid = pd.Series(best_rf.feature_importances_,
                            index=getattr(best_rf, 'feature_names_in_', X.columns)
                            ).sort_values(ascending=False).head(20)

        X_ng = X.drop(columns=['grid'], errors='ignore').copy()
        rf_ng = RandomForestRegressor(n_estimators=500, random_state=random_state, n_jobs=-1)
        rf_ng.fit(X_ng.loc[X_train.index], y_train)
        fi_ng = pd.Series(rf_ng.feature_importances_,
                          index=getattr(rf_ng, 'feature_names_in_', X_ng.columns)
                          ).sort_values(ascending=False).head(20)

        out_dir = os.path.join(LIVE2025_DIR, "figs"); os.makedirs(out_dir, exist_ok=True)
        plt.figure(figsize=(8, 7)); fi_grid.sort_values().plot(kind='barh')
        plt.title('Önemler (Gridli) – Top-20'); plt.xlabel('Önem skoru'); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'fi_top20_gridli.png'), dpi=160); plt.close()

        plt.figure(figsize=(8, 7)); fi_ng.sort_values().plot(kind='barh')
        plt.title('Önemler (Gridsiz) – Top-20'); plt.xlabel('Önem skoru'); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'fi_top20_gridsiz.png'), dpi=160); plt.close()

        cmp = (fi_grid.rename('gridli').to_frame()
               .merge(fi_ng.rename('gridsiz').to_frame(), left_index=True, right_index=True, how='outer')
               .fillna(0).sort_values('gridli', ascending=True).tail(20))
        plt.figure(figsize=(9, 7)); cmp.plot(kind='barh'); plt.title('Önem Kıyas – Gridli vs Gridsiz (Top-20)')
        plt.xlabel('Önem skoru'); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'fi_top20_kıyas.png'), dpi=160); plt.close()
    except Exception as e:
        print('[WARN] Önem kıyas grafikleri üretilemedi:', e)

    # 5) 2025 verisi (Kaggle/FastF1)
    if not skip_kaggle and KAGGLE_2025_DIR:
        try:
            import_f1_2025_kaggle_to_base(KAGGLE_2025_DIR, LIVE2025_DIR, ARCHIVE_DIR)
        except FileNotFoundError as e:
            print(f"[WARN] Kaggle 2025 dosyaları bulunamadı: {e}")
    if not skip_fastf1:
        fetch_2025_via_fastf1(LIVE2025_DIR)

    # 5.2) 2025’i yükle → feature → sezon tahmini (to-date)
    clean_2025 = load_raw_and_clean(base_dir=LIVE2025_DIR)
    if ('results' not in clean_2025) or clean_2025['results'].empty:
        print("[UYARI] 2025 'results.csv' boş görünüyor veya hiç veri yok — 2025 projeksiyonu atlanıyor.")
        return

    # Sunum figürleri (varsa)
    try:
        export_presentation_figures(
            master=master, X=X, y=y, model=best_rf, test_years=test_years,
            season_2025=None, season_2025_proj=None,
            out_dir=os.path.join(LIVE2025_DIR, "figs"),
            grid_compare_metrics=grid_compare_metrics
        )
        try:
            export_feature_engineering_extra_charts(all_clean, master, out_dir=os.path.join(LIVE2025_DIR, 'figs'))
        except Exception as _e:
            print('[WARN] FE ek grafikler atlandı:', _e)
        try:
            export_fe_vs_baseline_metrics_chart(master, X, y, test_years, out_dir=os.path.join(LIVE2025_DIR, 'figs'))
        except Exception as _e:
            print('[WARN] FE metrik karşılaştırma grafiği atlandı:', _e)
        print(f"[OK] Görseller kaydedildi → {os.path.join(LIVE2025_DIR, 'figs')}")
    except Exception as e:
        print('[WARN] Sunum görselleri üretilemedi:', e)

    # 2025 FE
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

    # İsim haritası 2025
    name_map_2025 = build_driver_name_map_2025(
        clean_2025.get('drivers', pd.DataFrame()),
        overrides=getattr(CFG, "MANUAL_NAME_OVERRIDES_2025", {})
    )
    if isinstance(meta_2025.get('id_to_driver'), pd.DataFrame) and name_map_2025.empty:
        name_map_2025 = meta_2025['id_to_driver'].copy()
        name_map_2025['driver_name'] = (
            name_map_2025['forename'].astype(str) + ' ' + name_map_2025['surname'].astype(str)
        )

    if X_2025.empty:
        print("[BİLGİ] 2025 feature tablosu boş; sezon çıktısı üretilemedi.")
        return

    # Eğitimdeki kolon şemasına hizala
    cols_train = meta.get('columns_X', X.columns.tolist())
    missing = [c for c in cols_train if c not in X_2025.columns]
    extra   = [c for c in X_2025.columns if c not in cols_train]
    if missing or extra:
        print(f"[INFO] 2025 align: +{len(missing)} eksik kolonu 0 ile doldurduk, -{len(extra)} fazlalığı attık.")
    X_2025_aligned = align_columns_for_inference(X_2025, cols_train)

    # 2025 bugüne kadarki sezon tablosu
    season_2025 = season_tables_from_preds(master_2025, X_2025_aligned, y_2025, best_rf, years=[2025], name_map=name_map_2025)
    print("\n=== 2025 Sezonu (şu ana kadar) – Tahmin Top-10 ===")
    print(season_2025[season_2025['year']==2025].nsmallest(10, 'rank_pred')
          [['driver_name','pred_total','rank_pred']].reset_index(drop=True))
    if 'true_total' in season_2025.columns:
        print("\n=== 2025 Sezonu (şu ana kadar) – Gerçek Top-10 ===")
        print(season_2025[season_2025['year']==2025].nsmallest(10, 'rank_true')
              [['driver_name','true_total','rank_true']].reset_index(drop=True))
        # self-check: results toplamı ile
        try:
            sub_2025 = season_2025[season_2025['year']==2025][['driverId','true_total']].copy()
            self_check_to_date_vs_results(sub_2025, clean_2025, id_col='driverId', true_col='true_total')
        except Exception as e:
            print('[CHECK] 2025 to-date self-check hata:', e)
        # isim debug
        try:
            if season_2025['driver_name'].isna().any():
                debug_missing_driver_names(season_2025, clean_2025)
        except Exception as _e:
            print('[DEBUG] isim debug çağrısı hatası:', _e)

        # 5.3) PROXY-GRID ile tam sezon projeksiyonu
        try:
            proxy_grid_df = _proxy_grid_from_kaggle_or_results(clean_2025)
            season_proj, per_round_pred = project_season_2025_with_proxy_grid(
                master_2025=master_2025,
                X_2025=X_2025_aligned,
                clean_2025=clean_2025,
                model=best_rf,
                proxy_grid_df=proxy_grid_df
            )
            # sıralama kolonları
            if 'proj_total' in season_proj.columns:
                season_proj['rank_proj'] = season_proj['proj_total'].rank(ascending=False, method='min')
            if 'true_to_date' in season_proj.columns and 'rank_true' not in season_proj.columns:
                season_proj['rank_true'] = season_proj['true_to_date'].rank(ascending=False, method='min')

            print("\n=== 2025 Tam Sezon – Proxy-Grid Projeksiyonu (İlk 15) ===")
            cols_show = ['rank_proj', 'driver_name', 'true_to_date', 'remaining_pred', 'proj_total']
            cols_show = [c for c in cols_show if c in season_proj.columns]
            print(season_proj[cols_show].head(15).reset_index(drop=True))

            # CSV kayıtları
            _csv_root = LIVE2025_DIR if hasattr(CFG, 'LIVE2025_DIR') else os.getcwd()
            season_proj.to_csv(os.path.join(_csv_root, "season_2025_proj.csv"), index=False)
            per_round_pred.to_csv(os.path.join(_csv_root, "per_round_pred.csv"), index=False)
            print(f"[OK] Finish-line girdileri yazıldı → {_csv_root}/season_2025_proj.csv & per_round_pred.csv")

            # Monte Carlo (şampiyonluk belirsizliği)
            try:
                if not season_proj.empty:
                    mc_summary, _ = monte_carlo_season(season_proj, n_sims=5000, sigma=4.5)
                    print("\n=== 2025 Monte Carlo Özet (ilk 10) ===")
                    print(mc_summary.head(10))
                    # figür
                    figs_root = LIVE2025_DIR
                    out_dir = os.path.join(figs_root, 'figs'); os.makedirs(out_dir, exist_ok=True)
                    top10 = mc_summary.head(10).copy()
                    labels = top10.get('driver_name', top10['driverId']).astype(str).tolist()
                    plt.figure(figsize=(9, 5))
                    plt.bar(range(len(top10)), top10['exp_total'].values)
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

            # self-check (projeksiyon tablosu)
            try:
                self_check_to_date_vs_results(season_proj[['driverId','true_to_date']], clean_2025,
                                              id_col='driverId', true_col='true_to_date')
            except Exception as e:
                print('[CHECK] projeksiyon self-check hata:', e)

            # Ek sunum figürleri (opsiyonel)
            try:
                figs_dir = os.path.join(LIVE2025_DIR, 'figs')
                export_presentation_figures(master, X, y, best_rf, test_years,
                                            season_2025=season_2025, season_2025_proj=season_proj,
                                            out_dir=figs_dir, grid_compare_metrics=grid_compare_metrics)
                try:
                    export_postrace_charts(clean_2025, figs_dir, year=2025)
                except Exception as e:
                    print('[UYARI] Post-race görseller üretilemedi:', e)
                print(f"[OK] Görseller kaydedildi → {figs_dir}")
            except Exception as e:
                print('[UYARI] Görseller üretilemedi:', e)

            # 2025 bugüne kadarki sıralama metrikleri
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


def main():
    parser = argparse.ArgumentParser(description="F1 2025 – uçtan uca puan/şampiyon projeksiyonu")
    parser.add_argument("--mode", default="run-all", choices=["run-all"])
    parser.add_argument("--skip-kaggle", action="store_true", help="Kaggle 2025 içe aktarmayı atla")
    parser.add_argument("--skip-fastf1", action="store_true", help="FastF1 çekimini atla")
    parser.add_argument("--holdout-years", type=int, default=2, help="Zaman duyarlı hold-out yıl sayısı (varsayılan 2)")
    args = parser.parse_args()

    if args.mode == "run-all":
        run_all(skip_kaggle=args.skip_kaggle, skip_fastf1=args.skip_fastf1, holdout_years=args.holdout_years)


if __name__ == "__main__":
    main()