from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib

ARTIFACT_PATH = "model_birmingham_hgb.joblib"

artifact = joblib.load(ARTIFACT_PATH)
model = artifact["model"]
feature_cols = artifact["feature_cols"]

app = FastAPI(title="Smart Parking ML", version="1.0")

class PredictBatchRequest(BaseModel):
    timestamps: List[str]  # ISO strings: "2026-03-10T14:30:00+02:00"

class PredictBatchResponse(BaseModel):
    occupancy_pct: List[float]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict_batch", response_model=PredictBatchResponse)
def predict_batch(req: PredictBatchRequest):
    ts = pd.to_datetime(pd.Series(req.timestamps), utc=False, errors="coerce")
    if ts.isna().any():
        bad = [req.timestamps[i] for i, v in enumerate(ts.isna()) if v]
        return {"occupancy_pct": [], "error": f"Invalid timestamps: {bad}"}

    df = pd.DataFrame({"timestamp": ts})

    # Calendar features (trebuie să corespundă exact cu training-ul v2)
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["minute_bucket"] = (df["minute"] // 15) * 15
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["timestamp"].dt.month

    # IMPORTANT:
    # Modelul v2 are și lag features. În cloud, pentru predicții “pe viitor” ai două opțiuni:
    # (A) folosești doar modelul calendar-only (v1) pentru intervale de zile/ore
    # (B) trimiți și lag-urile din backend (dacă ai occupancy recentă în DB)
    #
    # Ca să fie simplu și corect pentru “interval de zile/ore”,
    # recomand să deployezi în cloud și modelul v1 calendar-only pentru această grilă.
    # Aici presupunem că folosești varianta calendar-only (v1) SAU ai creat un v2 compatibil.
    #
    # Dacă vrei tot v2, trebuie să modificăm endpoint-ul să primească lag-uri.
    # (îți explic imediat mai jos)

    X = df.reindex(columns=feature_cols, fill_value=0)
    preds = model.predict(X)
    preds = [float(max(0, min(100, p))) for p in preds]
    return {"occupancy_pct": preds}