import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from f1proj.eval import season_tables_from_preds

# ---- Placeholder charts (çalışmayı engellemesin diye basit çıktılar) ----
def export_fe_vs_baseline_metrics_chart(master, X, y, years, out_dir):
    """FE vs Baseline karşılaştırma grafiği (placeholder)."""
    import os
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    figpath = os.path.join(out_dir, "fe_vs_baseline_dummy.png")
    plt.figure()
    plt.plot([0, 1], [0, 1])
    plt.title("FE vs Baseline (placeholder)")
    plt.tight_layout()
    plt.savefig(figpath, dpi=160)
    plt.close()
    return figpath


def export_feature_engineering_extra_charts(all_clean, master, out_dir):
    """FE ile ilgili ek grafikler (placeholder)."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    # Buraya ileride gerçek grafikler eklenebilir
    return None


def export_postrace_charts(clean_2025, out_dir, year=2025):
    """Post-race tarzı grafikler (placeholder)."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    return None

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

