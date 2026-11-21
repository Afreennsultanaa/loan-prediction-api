import os
import json
import logging
from typing import Optional, Union, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import joblib

# Import custom preprocessing pipelines
from preprocessing import preprocess_stage1, preprocess_stage2

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("loan_api")

# Global model and metadata store
MODEL_STORE: Dict[str, Any] = {
    "stage1_model": None,
    "stage1_metadata": None,
    "stage2_model": None,
    "stage2_metadata": None,
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_artifacts():
    """Load both XGBoost models and their respective metadata JSON files."""
    # 1. Load Stage 1
    m1_path = os.path.join(BASE_DIR, "stage1_xgboost.joblib")
    meta1_path = os.path.join(BASE_DIR, "stage1_metadata.json")

    if os.path.exists(m1_path) and os.path.exists(meta1_path):
        MODEL_STORE["stage1_model"] = joblib.load(m1_path)
        with open(meta1_path, "r", encoding="utf-8") as f:
            MODEL_STORE["stage1_metadata"] = json.load(f)
        logger.info(f"Loaded Stage 1 model and metadata ({len(MODEL_STORE['stage1_metadata']['feature_names'])} features).")
    else:
        logger.warning(f"Stage 1 artifacts not found at {m1_path} or {meta1_path}")

    # 2. Load Stage 2
    m2_path = os.path.join(BASE_DIR, "stage2_xgboost.joblib")
    meta2_path = os.path.join(BASE_DIR, "stage2_metadata.json")

    if os.path.exists(m2_path) and os.path.exists(meta2_path):
        MODEL_STORE["stage2_model"] = joblib.load(m2_path)
        with open(meta2_path, "r", encoding="utf-8") as f:
            MODEL_STORE["stage2_metadata"] = json.load(f)
        logger.info(f"Loaded Stage 2 model and metadata ({len(MODEL_STORE['stage2_metadata']['feature_names'])} features).")
    else:
        logger.warning(f"Stage 2 artifacts not found at {m2_path} or {meta2_path}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure models & metadata are loaded
    load_artifacts()
    yield
    # Shutdown cleanup if needed
    MODEL_STORE.clear()

# Load immediately on module initialization
load_artifacts()


app = FastAPI(
    title="Loan Intelligence API: Two-Stage Sanctioning & Risk Assessment",
    description=(
        "FastAPI service equipped with trained XGBoost models and complete preprocessing metadata.\n\n"
        "• **Stage 1 (Loan Sanction Classifier)**: Predicts loan approval vs rejection.\n"
        "• **Stage 2 (Payment Risk Analyzer)**: Predicts full repayment success vs delinquency/default risk."
    ),
    version="2.0.0",
    lifespan=lifespan
)


# =====================================================================
# Pydantic Schemas
# =====================================================================

class Stage1Application(BaseModel):
    loan_amount: float = Field(..., description="Requested loan amount ($)", examples=[15000.0])
    employment_duration: Optional[Union[str, int, float]] = Field("5 years", description="Employment length (e.g., '10+ years', '< 1 year', '5 years' or integer 0-10)", examples=["5 years"])
    debt_to_income_ratio: Optional[Union[float, str]] = Field(15.0, description="Debt-to-income ratio (numeric or e.g. '18.5%')", examples=[14.5])
    purpose: Optional[str] = Field("debt_consolidation", description="Loan purpose category", examples=["debt_consolidation"])
    state: Optional[str] = Field("CA", description="2-letter US state code", examples=["CA"])


class Stage2Application(BaseModel):
    loan_amnt: float = Field(..., description="Loan amount ($)", examples=[15000.0])
    term: Optional[Union[str, int, float]] = Field("36 months", description="Loan term (e.g., '36 months', '60 months' or integer)", examples=["36 months"])
    int_rate: Optional[Union[float, str]] = Field(10.5, description="Interest rate percentage", examples=[10.5])
    sub_grade: Optional[str] = Field("B2", description="Lending subgrade (A1 through G5)", examples=["B2"])
    emp_length: Optional[Union[str, int, float]] = Field("5 years", description="Employment length", examples=["5 years"])
    home_ownership: Optional[str] = Field("MORTGAGE", description="Home ownership: RENT, OWN, MORTGAGE, OTHER", examples=["MORTGAGE"])
    annual_inc: Optional[float] = Field(75000.0, description="Annual income ($)", examples=[75000.0])
    purpose: Optional[str] = Field("debt_consolidation", description="Loan purpose category", examples=["debt_consolidation"])
    dti: Optional[Union[float, str]] = Field(14.2, description="Debt to income ratio (%)", examples=[14.2])
    fico_score: Optional[float] = Field(710.0, description="FICO credit score (e.g. 710)", examples=[710.0])
    fico_range_low: Optional[float] = Field(None, description="Low end of FICO range if fico_score not directly provided")
    fico_range_high: Optional[float] = Field(None, description="High end of FICO range if fico_score not directly provided")
    revol_util: Optional[Union[float, str]] = Field(30.5, description="Revolving utilization (%)", examples=[30.5])
    mort_acc: Optional[float] = Field(1.0, description="Number of mortgage accounts", examples=[1.0])


class PipelineApplication(BaseModel):
    """Full application suitable for both Stage 1 (Sanctioning) and Stage 2 (Repayment Risk)."""
    loan_amount: float = Field(..., description="Requested loan amount ($)", examples=[15000.0])
    annual_inc: Optional[float] = Field(75000.0, description="Annual income ($)", examples=[75000.0])
    employment_duration: Optional[Union[str, int, float]] = Field("5 years", description="e.g. '10+ years', '< 1 year', '5 years' or integer 0-10", examples=["5 years"])
    debt_to_income_ratio: Optional[Union[float, str]] = Field(14.5, description="Debt-to-income ratio (%)", examples=[14.5])
    purpose: Optional[str] = Field("debt_consolidation", description="Loan purpose", examples=["debt_consolidation"])
    state: Optional[str] = Field("CA", description="2-letter US state code", examples=["CA"])
    term: Optional[Union[str, int, float]] = Field("36 months", description="Loan term (e.g., '36 months' or 36)", examples=["36 months"])
    int_rate: Optional[Union[float, str]] = Field(10.5, description="Expected interest rate (%)", examples=[10.5])
    sub_grade: Optional[str] = Field("B2", description="Lending subgrade e.g. A1-G5", examples=["B2"])
    home_ownership: Optional[str] = Field("MORTGAGE", description="RENT, OWN, MORTGAGE, OTHER", examples=["MORTGAGE"])
    fico_score: Optional[float] = Field(710.0, description="FICO credit score", examples=[710.0])
    revol_util: Optional[Union[float, str]] = Field(30.5, description="Revolving credit utilization (%)", examples=[30.5])
    mort_acc: Optional[float] = Field(1.0, description="Number of mortgage accounts", examples=[1.0])


# =====================================================================
# API Endpoints
# =====================================================================

@app.get("/", summary="Health Check and Metadata Status")
def home():
    return {
        "status": "online",
        "service": "Loan Intelligence API",
        "stage1_loaded": MODEL_STORE["stage1_model"] is not None and MODEL_STORE["stage1_metadata"] is not None,
        "stage2_loaded": MODEL_STORE["stage2_model"] is not None and MODEL_STORE["stage2_metadata"] is not None,
        "stage1_features_count": len(MODEL_STORE["stage1_metadata"]["feature_names"]) if MODEL_STORE["stage1_metadata"] else 0,
        "stage2_features_count": len(MODEL_STORE["stage2_metadata"]["feature_names"]) if MODEL_STORE["stage2_metadata"] else 0,
        "documentation": "/docs"
    }


@app.get("/health", summary="Health Check Endpoint")
def health():
    return {
        "status": "healthy",
        "service": "Loan Intelligence API",
        "stage1_model_loaded": MODEL_STORE["stage1_model"] is not None,
        "stage2_model_loaded": MODEL_STORE["stage2_model"] is not None
    }


@app.get("/metadata/stage1", summary="Get Stage 1 Preprocessing Metadata")
def get_stage1_metadata():
    if not MODEL_STORE["stage1_metadata"]:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Stage 1 metadata not loaded")
    return MODEL_STORE["stage1_metadata"]


@app.get("/metadata/stage2", summary="Get Stage 2 Preprocessing Metadata")
def get_stage2_metadata():
    if not MODEL_STORE["stage2_metadata"]:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Stage 2 metadata not loaded")
    return MODEL_STORE["stage2_metadata"]


@app.post("/predict/stage1", summary="Stage 1: Predict Loan Sanction / Approval")
def predict_stage1(application: Stage1Application):
    if not MODEL_STORE["stage1_model"] or not MODEL_STORE["stage1_metadata"]:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Stage 1 model/metadata not loaded")

    model = MODEL_STORE["stage1_model"]
    metadata = MODEL_STORE["stage1_metadata"]

    # Preprocess raw input according to metadata
    X = preprocess_stage1(application.model_dump(), metadata)

    # Inference
    probabilities = model.predict_proba(X)[0]
    prob_reject = float(probabilities[0])
    prob_approved = float(probabilities[1])

    threshold = metadata.get("default_threshold", 0.50)
    is_approved = prob_approved >= threshold

    return {
        "stage": 1,
        "model_name": metadata["model_name"],
        "loan_sanctioned": int(is_approved),
        "status": "Approved" if is_approved else "Rejected",
        "approval_probability": round(prob_approved, 4),
        "rejection_probability": round(prob_reject, 4),
        "decision_threshold": threshold
    }


@app.post("/predict/stage2", summary="Stage 2: Payment Risk & Repayment Analyzer")
def predict_stage2(application: Stage2Application):
    if not MODEL_STORE["stage2_model"] or not MODEL_STORE["stage2_metadata"]:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Stage 2 model/metadata not loaded")

    model = MODEL_STORE["stage2_model"]
    metadata = MODEL_STORE["stage2_metadata"]

    # Preprocess raw input according to metadata
    X = preprocess_stage2(application.model_dump(), metadata)

    # Inference
    probabilities = model.predict_proba(X)[0]
    prob_default = float(probabilities[0])
    prob_full_pay = float(probabilities[1])

    thresholds = metadata.get("thresholds", {})
    cost_cutoff = thresholds.get("cost_optimized_default_cutoff", 0.17)
    std_cutoff = thresholds.get("standard_threshold", 0.50)

    is_high_risk = prob_default >= cost_cutoff

    return {
        "stage": 2,
        "model_name": metadata["model_name"],
        "full_payment_success": int(prob_full_pay >= std_cutoff),
        "full_payment_probability": round(prob_full_pay, 4),
        "default_risk_probability": round(prob_default, 4),
        "risk_classification": "High Risk (Default Warning)" if is_high_risk else "Low Risk (Approve Repayment)",
        "cost_optimized_default_cutoff": cost_cutoff,
        "recommendation": "Flag for enhanced review or higher interest rate" if is_high_risk else "Low risk profile"
    }


@app.post("/predict/pipeline", summary="End-to-End Two-Stage Loan Evaluation")
def predict_pipeline(application: PipelineApplication):
    """
    Executes the full two-stage loan evaluation:
    1. Evaluates Stage 1 (Loan Sanctioning). If rejected, returns immediately.
    2. If sanctioned, evaluates Stage 2 (Payment Risk & Default Probability).
    """
    if not MODEL_STORE["stage1_model"] or not MODEL_STORE["stage2_model"]:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Models or metadata not loaded")

    app_dict = application.model_dump()

    # --- Stage 1 Evaluation ---
    meta1 = MODEL_STORE["stage1_metadata"]
    X1 = preprocess_stage1(app_dict, meta1)
    prob1 = MODEL_STORE["stage1_model"].predict_proba(X1)[0]
    prob_approved = float(prob1[1])
    is_approved = prob_approved >= meta1.get("default_threshold", 0.50)

    response = {
        "pipeline_status": "Complete",
        "stage1_sanction": {
            "status": "Approved" if is_approved else "Rejected",
            "approval_probability": round(prob_approved, 4),
            "rejection_probability": round(float(prob1[0]), 4),
        }
    }

    if not is_approved:
        response["overall_decision"] = "REJECTED (Stage 1 Failed)"
        response["stage2_risk"] = None
        response["notes"] = "Applicant does not meet minimum credit sanction criteria."
        return response

    # --- Stage 2 Evaluation ---
    # Map fields for Stage 2 if needed
    stage2_dict = app_dict.copy()
    stage2_dict["loan_amnt"] = app_dict.get("loan_amount", 12500.0)
    stage2_dict["emp_length"] = app_dict.get("employment_duration", "6.0")
    stage2_dict["dti"] = app_dict.get("debt_to_income_ratio", 18.41)

    meta2 = MODEL_STORE["stage2_metadata"]
    X2 = preprocess_stage2(stage2_dict, meta2)
    prob2 = MODEL_STORE["stage2_model"].predict_proba(X2)[0]
    prob_default = float(prob2[0])
    prob_full_pay = float(prob2[1])

    cost_cutoff = meta2.get("thresholds", {}).get("cost_optimized_default_cutoff", 0.17)
    is_high_risk = prob_default >= cost_cutoff

    response["stage2_risk"] = {
        "full_payment_probability": round(prob_full_pay, 4),
        "default_risk_probability": round(prob_default, 4),
        "risk_classification": "High Risk (Default Warning)" if is_high_risk else "Low Risk (Favorable)",
        "cost_cutoff": cost_cutoff
    }

    if is_high_risk:
        response["overall_decision"] = "CONDITIONALLY APPROVED (High Repayment Risk - Higher Margin / Guarantor Recommended)"
    else:
        response["overall_decision"] = "APPROVED (Low Repayment Risk)"

    return response


@app.post("/predict", summary="Main Prediction Endpoint")
def predict(application: PipelineApplication):
    """
    Main Prediction Endpoint (matching the project submission specification).
    Runs the complete loan evaluation pipeline.
    """
    return predict_pipeline(application)


@app.post("/predict/batch/stage1", summary="Batch Stage 1 Predictions")
def batch_predict_stage1(applications: List[Stage1Application]):
    if not MODEL_STORE["stage1_model"] or not MODEL_STORE["stage1_metadata"]:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Stage 1 model/metadata not loaded")

    raw_list = [app.model_dump() for app in applications]
    X = preprocess_stage1(raw_list, MODEL_STORE["stage1_metadata"])
    probs = MODEL_STORE["stage1_model"].predict_proba(X)

    results = []
    threshold = MODEL_STORE["stage1_metadata"].get("default_threshold", 0.50)
    for p in probs:
        p_app = float(p[1])
        results.append({
            "status": "Approved" if p_app >= threshold else "Rejected",
            "approval_probability": round(p_app, 4)
        })
    return {"batch_size": len(results), "results": results}


@app.post("/predict/batch/stage2", summary="Batch Stage 2 Predictions")
def batch_predict_stage2(applications: List[Stage2Application]):
    if not MODEL_STORE["stage2_model"] or not MODEL_STORE["stage2_metadata"]:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Stage 2 model/metadata not loaded")

    raw_list = [app.model_dump() for app in applications]
    X = preprocess_stage2(raw_list, MODEL_STORE["stage2_metadata"])
    probs = MODEL_STORE["stage2_model"].predict_proba(X)

    cost_cutoff = MODEL_STORE["stage2_metadata"].get("thresholds", {}).get("cost_optimized_default_cutoff", 0.17)
    results = []
    for p in probs:
        p_def = float(p[0])
        p_pay = float(p[1])
        results.append({
            "full_payment_probability": round(p_pay, 4),
            "default_risk_probability": round(p_def, 4),
            "risk_classification": "High Risk" if p_def >= cost_cutoff else "Low Risk"
        })
    return {"batch_size": len(results), "results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

