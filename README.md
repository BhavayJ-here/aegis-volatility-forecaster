# Aegis

A self-recalibrating volatility forecasting system. It predicts a range for
BTC/USDT realized volatility instead of a single number, and instead of
trusting that range blindly, it constantly checks whether the range is
actually right, and fixes itself when it's not.

## What it does

1. Trains LightGBM quantile regression models on 2 years of BTC/USDT hourly
   data to forecast realized volatility at three horizons (1h, 6h, 24h).
2. Wraps the raw model output in Conformalized Quantile Regression (CQR) to
   correct systematic undercoverage, the raw model claims a 90% interval but
   doesn't actually hit 90% in practice, CQR fixes that.
3. Adds Adaptive Conformal Inference (ACI) on top, so the correction updates
   itself online as new data comes in, instead of staying fixed after one
   calibration pass. Standard conformal prediction assumes the data is
   exchangeable, time series data isn't, ACI is the fix for that.
4. Serves the calibrated forecaster as a live FastAPI service that tracks
   its own coverage, runs a Page-Hinkley drift test, and recalibrates itself
   when coverage breaks down, no model retraining involved, just a refit of
   the calibration correction.

## Results

Expanding-window walk-forward backtest, 5 folds per horizon, averaged:

| Horizon | Naive Coverage | Static CQR Coverage | Adaptive ACI Coverage | Naive Winkler | Static Winkler | Adaptive Winkler |
|---|---|---|---|---|---|---|
| 1h  | 0.831 | 0.892 | 0.899 | 0.0103 | 0.0100 | 0.0099 |
| 6h  | 0.773 | 0.895 | 0.898 | 0.0220 | 0.0203 | 0.0192 |
| 24h | 0.715 | 0.890 | 0.897 | 0.0423 | 0.0365 | 0.0312 |

Target coverage is 0.90. Naive (uncalibrated) coverage degrades badly as the
horizon grows, adaptive ACI holds steady near target across all three
horizons and beats static CQR on both coverage and Winkler score
(width-and-coverage combined) at every horizon, with the gap widening at
longer horizons.

The live service was also stress-tested with a synthetic regime shift
(replayed test data with a 5x volatility shock injected mid-stream). Rolling
coverage dropped during the shock and recovered afterward as the online
correction adapted, confirming the recalibration mechanism works under
distribution shift, not just in the offline backtest.

## Architecture


aegis_notebook.ipynb   -> data pull, feature engineering, model training,
                           CQR/ACI validation, walk-forward backtest,
                           saves trained models + calibration data
aegis_service/
  main.py               -> FastAPI app: /forecast, /actual, /health, /coverage
  calibration_store.py  -> holds per-horizon calibration state, applies the
                            CQR correction, runs the ACI online update
  drift_detector.py     -> Page-Hinkley test, flags sustained coverage drift
  coverage_tracker.py   -> rolling coverage tracking per horizon


## Running it

**1. Set up the environment**
bash
python -m venv aegis-venv
source aegis-venv/bin/activate      # windows: aegis-venv\Scripts\activate
pip install -r requirements.txt


**2. Run the notebook**

Run aegis_notebook.ipynb top to bottom. This pulls fresh data from
Binance's public API, trains all models, runs the backtest, and saves the
trained models and calibration data (.pkl / .npy / .json files) needed
by the service. These files aren't committed to the repo, you need to
generate them locally first.

Move the generated files into aegis_service/.

**3. Start the service**
bash
cd aegis_service
uvicorn main:app --reload


Visit http://127.0.0.1:8000/docs for an interactive API explorer.

## API

- POST /forecast — takes a horizon (1, 6, or 24) and a feature dict,
  returns a calibrated interval
- POST /actual — reports the real observed value for a past forecast,
  updates coverage tracking and triggers recalibration if needed
- GET /coverage — current rolling coverage, calibration state, and
  recalibration history, per horizon
- GET /health — basic health check

## Known limitations

- The recalibration check runs synchronously when /actual is called,
  not as an independent scheduled background job. Functionally similar,
  architecturally different from a true background monitor.
- No database or persistence layer, calibration state resets if the
  service restarts. No Docker setup currently included.
- The walk-forward backtest uses 5 expanding folds per horizon, not a
  larger sweep, given time constraints. More folds would tighten
  confidence in the averaged results.
- Marginal coverage is what's guaranteed here (correct on average across
  all inputs), not conditional coverage (correct for every specific
  input or regime), which is a much harder, largely open problem.
