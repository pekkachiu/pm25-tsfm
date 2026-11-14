# 時間序列基礎模型能否追上本地模型？台灣 PM2.5 零樣本預測

*Can Time Series Foundation Models Close the Gap? Covariate-Informed Zero-Shot PM2.5 Forecasting in Taiwan*

人工智慧導論期末專題。比較零樣本（Zero-shot）時間序列基礎模型（TSFM，Chronos-Bolt／Chronos-2）與各測站自行訓練的模型（XGBoost、LSTM），預測台灣 12 個測站未來 1、6、24 小時的逐時 PM2.5。

**互動式預測瀏覽器：<https://pekkachiu.github.io/pm25-tsfm/>**
選測站、測試年與提前時間，比較各模型的預測曲線、80% 預測區間與誤差。

## 研究問題

- **RQ1**：零樣本 TSFM 能否追上本地訓練的基準模型？微調（Fine-tuning）之後呢？
- **RQ2**：Chronos-2 以情境學習（In-context Learning）加入風場共變數（Covariates），能否降低誤差？當所有模型都拿到同樣的數值天氣預報（NWP）風場時呢？
- **RQ3**：高污染事件與季節轉換（概念漂移，Concept Drift）時，模型表現如何？

## 主要結果

12 站合併、提前 24 小時的 MAE（μg/m³，越低越好）。測試年 2025 以 2021–2024 訓練；2024 以 2021–2023 訓練，用來檢查結論是否穩健。

| 模型 | 類型 | 2025 | 2024 |
|---|---|---:|---:|
| 持續性（Persistence） | 基準 | 6.86 | 7.07 |
| XGBoost | 各站訓練 | 5.70 | 5.95 |
| LSTM | 各站訓練 | 5.81 | 5.98 |
| XGBoost＋NWP | 各站訓練 | 5.41 | 5.63 |
| Chronos-Bolt Small | 零樣本 | 6.02 | 6.39 |
| Chronos-2（只看 PM2.5） | 零樣本 | 6.09 | 6.46 |
| Chronos-2＋過去共變數 | 零樣本 | 5.88 | 6.25 |
| Chronos-2＋NWP 風場 | 零樣本 | 5.70 | 5.92 |
| **Chronos-2 微調＋NWP** | 微調 | **5.26** | **5.55** |

Diebold-Mariano 檢定（Newey-West 48 小時，12 站合併）：

1. **零樣本還追不上本地訓練模型。** Chronos-2 比持續性模型好 11.1%（2024：8.5%），但比 XGBoost 差 6.8%（2024：8.6%），12 站全輸，p < 0.001。加入過去共變數後差距縮小到 3.1%（p = 0.002；2024：5.0%）。
2. **NWP 預報風場對所有模型都有幫助，但不改變排名。** XGBoost −5.1%、Chronos-2 −3.0%（2024：皆約 −5.2%）。同樣資訊下 Chronos-2＋NWP 仍落後 XGBoost＋NWP 5.4%（2024：5.0%），兩年皆 p < 0.001。
3. **微調後翻盤，但幅度不穩健。** 微調 Chronos-2＋NWP 比零樣本版本好 7.8%（2024：6.2%）。2025 年勝 XGBoost＋NWP 2.8%（p = 0.008，12/12 站）；2024 年只勝 1.5%，且不顯著（p = 0.16，10/12 站）。
4. **高污染事件期間，微調反而較差。** 秋冬事件（PM2.5 ≥ 30.5）期間，微調版 MAE 14.35，高於零樣本＋NWP 的 13.61 與 XGBoost＋NWP 的 13.43；2024 年結果相同。
5. **預測區間偏窄。** 原始 80% 區間的實際覆蓋率約 0.73–0.77；以自適應保形推論（ACI）校正後達 0.79–0.80。
6. **鄰站 PM2.5 對短時距幫助大。** 1 小時與 6 小時約改善 2–3%，24 小時影響小。

## 資料與實驗設計

- **資料**：環境部環境資料開放平臺 AQX_P_13 逐時測值，2021–2025 年，含 PM2.5、PM10、NO2、O3、風速、風向。
- **測站**：北部、中部、雲嘉南、高屏空品區各 3 站，共 12 站。
- **前處理**：無效值視為缺值；連續缺值 ≤ 3 小時線性內插；風向編碼為 (sin θ, cos θ)。
- **滾動評估**：每小時一個預測起點，context 512 小時，預測 24 步，取第 1、6、24 步。
- **NWP 風場**：Open-Meteo previous-runs 的 JMA GSM，使用有效時間前 24 小時發布的版本，預測起點時已可取得，沒有資訊洩漏。
- **模型**：
  - Persistence
  - XGBoost（多分位數、直接多步）
  - LSTM（pinball loss）
  - Chronos-Bolt Small／Base
  - Chronos-2：單變量、過去共變數、NWP、觀測未來風（oracle 上界）、鄰站
  - 全域微調 Chronos-2
- **評估**：MAE、RMSE、80% 區間覆蓋率、WQL、區間分數、超標（≥ 30.5 μg/m³，AQI 100）的 P/R/F1；並依空品區、季節、事件分析，以 ADWIN 偵測漂移，以 DM 檢定比較模型，以 ACI 校正區間。

## 重現

需要 Python 3.10–3.12 與 [uv](https://docs.astral.sh/uv/)。macOS 上 XGBoost 另需 `brew install libomp`。

```bash
uv sync --extra models
./scripts/test.sh                                   # Python + 網頁 JS 測試

# 下載資料需要環境部開放資料 API 金鑰：放在 .env 的 MOENV_API_KEY（不要提交）
uv run --env-file .env python -m pm25.data.download
uv run python -m pm25.data.build
uv run python -m pm25.data.wind_forecast --model jma_gsm

# 實驗（加 --test-year 2024 跑第二測試年）
uv run python -m pm25.experiment --family baseline|xgboost|lstm|chronos|nwp_xgboost|nwp_torch|finetuned|spatial_xgboost|spatial_chronos
uv run python -m pm25.evaluation.report
uv run python -m pm25.evaluation.analysis
uv run python -m pm25.evaluation.figures

# 更新網頁資料（docs/data）
uv run python -m pm25.web.export
```

Chronos 與 LSTM 較耗時的實驗在 GPU 上執行（`notebooks/` 內的 Colab notebook，`./scripts/build_colab_zip.sh` 打包專案）。macOS 上 torch 與 xgboost 不能在同一個行程執行，所以兩者分開跑。

## 專案結構

```
src/pm25/
  config.py        # 所有實驗常數（測站、時距、門檻、模型設定）
  data/            # 下載、前處理、NWP 風場、鄰站
  models/          # persistence、xgboost、lstm、chronos、微調
  evaluation/      # 滾動評估、指標、報表、DM 檢定、ACI、圖
  web/export.py    # 網頁資料匯出
  experiment.py    # 依實驗組執行滾動評估
docs/              # GitHub Pages 預測瀏覽器（靜態網頁與資料）
notebooks/         # Colab notebook（GPU 實驗、微調）
tests/             # pytest 與網頁 JS 測試
```

## 資料來源

- 空氣品質：[環境部環境資料開放平臺](https://data.moenv.gov.tw/)（AQX_P_13）
- 預報風場：[Open-Meteo](https://open-meteo.com/)（JMA GSM）
- 地圖：© [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors
