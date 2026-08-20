import joblib
import json
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from calibration_store import MultiHorizonCalibrationStore
from coverage_tracker import CoverageTracker
from drift_detector import DriftDetector

app = FastAPI(title="Aegis")

HORIZONS = [1, 6, 24]
ALPHA = 0.10
GAMMA = 0.01

lower_models = {}
upper_models = {}
initial_scores_by_horizon = {}

for h in HORIZONS:
    lower_models[h] = joblib.load(f"lower_model_h{h}.pkl")
    upper_models[h] = joblib.load(f"upper_model_h{h}.pkl")
    initial_scores_by_horizon[h] = np.load(f"calibration_scores_h{h}.npy")

with open("feature_cols.json") as f:
    feature_cols = json.load(f)

cal_store = MultiHorizonCalibrationStore(initial_scores_by_horizon, alpha=ALPHA, window_size=500, gamma=GAMMA)
coverage_trackers = {h: CoverageTracker(window_size=200) for h in HORIZONS}
drift_detectors = {h: DriftDetector(delta=0.005, threshold=5.0) for h in HORIZONS}

waiting_for_actual = {}
drift_event_log = {h: [] for h in HORIZONS}


class ForecastRequest(BaseModel):
    request_id: str
    horizon: int
    features: dict


class ActualValueRequest(BaseModel):
    request_id: str
    actual_value: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/forecast")
def forecast(payload: ForecastRequest):
    if payload.horizon not in HORIZONS:
        return {"error": "horizon must be one of " + str(HORIZONS)}

    h = payload.horizon
    row = [[payload.features[col] for col in feature_cols]]

    raw_lower = lower_models[h].predict(row)[0]
    raw_upper = upper_models[h].predict(row)[0]

    store = cal_store.get_store(h)
    final_lower, final_upper = store.widen_interval(raw_lower, raw_upper)

    waiting_for_actual[payload.request_id] = {
        "horizon": h,
        "raw_lower": raw_lower,
        "raw_upper": raw_upper,
        "final_lower": final_lower,
        "final_upper": final_upper,
    }

    return {
        "request_id": payload.request_id,
        "horizon": h,
        "lower_bound": final_lower,
        "upper_bound": final_upper,
        "nominal_coverage": 1 - ALPHA,
    }


@app.post("/actual")
def report_actual(payload: ActualValueRequest):
    prediction = waiting_for_actual.pop(payload.request_id, None)
    if prediction is None:
        return {"error": "no matching prediction found for this id"}

    h = prediction["horizon"]
    store = cal_store.get_store(h)

    was_covered = bool(prediction["final_lower"] <= payload.actual_value <= prediction["final_upper"])
    coverage_trackers[h].record(was_covered)
    store.update_after_actual(prediction["raw_lower"], prediction["raw_upper"], payload.actual_value, was_covered)

    error_flag = 0 if was_covered else 1
    drift_found = drift_detectors[h].check(error_flag)

    if drift_found:
        drift_event_log[h].append({
            "alpha_t": store.alpha_t,
            "q_hat": store.q_hat,
            "rolling_coverage": coverage_trackers[h].rolling_coverage,
        })

    return {
        "horizon": h,
        "was_covered": was_covered,
        "rolling_coverage": coverage_trackers[h].rolling_coverage,
        "current_alpha_t": store.alpha_t,
        "current_q_hat": store.q_hat,
        "drift_flagged": drift_found,
    }


@app.get("/coverage")
def coverage():
    result = {}
    for h in HORIZONS:
        store = cal_store.get_store(h)
        result[f"horizon_{h}"] = {
            "rolling_coverage": coverage_trackers[h].rolling_coverage,
            "target_coverage": 1 - ALPHA,
            "current_alpha_t": store.alpha_t,
            "current_q_hat": store.q_hat,
            "drift_events_flagged": len(drift_event_log[h]),
            "drift_event_log": drift_event_log[h],
        }
    return result