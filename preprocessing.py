import re
import json
import numpy as np
import pandas as pd

def clean_percentage(val):
    if val is None or pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val).strip().replace('%', '')
    try:
        return float(val_str)
    except ValueError:
        return np.nan

def clean_employment(val, emp_map, default_val=1.0):
    if val is None or pd.isna(val):
        return default_val
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val).strip().lower()
    # Handle zero employment / unemployed variations
    if any(val_str == x for x in ['< 1 year', '<1 year', '< 1', '<1', '0', '0.0', '0 year', '0 years', 'none', 'unemployed', 'no']):
        return 0.0
    # Check case-insensitive match in emp_map
    for k, v in emp_map.items():
        if k.lower() == val_str:
            return float(v)
    # Extract digits with regex (e.g. '10+ years' -> 10, '3 years' -> 3, '1 years' -> 1)
    digits = re.findall(r'\d+', val_str)
    if digits:
        d = float(digits[0])
        return min(max(d, 0.0), 10.0)
    return default_val

def map_purpose(purpose_str, regex_rules):
    if purpose_str is None or pd.isna(purpose_str):
        clean_str = 'debt consolidation'
    else:
        clean_str = str(purpose_str).lower().replace('_', ' ').strip()
    
    for rule in regex_rules:
        if re.search(rule["pattern"], clean_str):
            return rule["category"]
    return 'Other'

def preprocess_stage1(input_data, metadata):
    """
    Preprocess raw input into the 60 exact features expected by stage1_xgboost.joblib
    """
    if isinstance(input_data, dict):
        df_raw = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        df_raw = pd.DataFrame(input_data)
    else:
        df_raw = input_data.copy()

    medians = metadata.get("medians", {})
    emp_map = metadata.get("mappings", {}).get("employment_duration_map", {})
    purpose_categories = metadata.get("mappings", {}).get("purpose_categories", [])
    state_categories = metadata.get("mappings", {}).get("state_categories", [])
    rules = metadata.get("purpose_regex_rules", [])
    feature_names = metadata.get("feature_names", [])

    df_out = pd.DataFrame(index=df_raw.index)

    # 1. loan_amount
    loan_val = pd.to_numeric(df_raw.get("loan_amount", np.nan), errors="coerce")
    df_out["loan_amount"] = loan_val.fillna(medians.get("loan_amount", 12000.0))

    # 2. employment_duration
    raw_emp = df_raw.get("employment_duration", df_raw.get("emp_length", np.nan))
    df_out["employment_duration"] = [
        clean_employment(v, emp_map, medians.get("employment_duration", 1.0)) for v in raw_emp
    ]

    # 3. debt_to_income_ratio
    raw_dti = df_raw.get("debt_to_income_ratio", df_raw.get("dti", np.nan))
    dti_vals = [clean_percentage(v) for v in raw_dti]
    dti_series = pd.Series(dti_vals, index=df_raw.index)
    dti_series = dti_series.replace([-1, 9999], np.nan)
    dti_series.loc[dti_series > 1000] = np.nan
    df_out["debt_to_income_ratio"] = dti_series.fillna(medians.get("debt_to_income_ratio", 19.1))

    # 4. purpose_group one-hot encoding
    raw_purpose = df_raw.get("purpose", "")
    mapped_purposes = [map_purpose(p, rules) for p in raw_purpose]
    for cat in purpose_categories:
        col_name = f"purpose_group_{cat}"
        df_out[col_name] = [1 if mp == cat else 0 for mp in mapped_purposes]

    # 5. state one-hot encoding
    raw_state = df_raw.get("state", metadata.get("modes", {}).get("state", "CA"))
    clean_states = [str(s).strip().upper() if pd.notna(s) and str(s).strip() else "CA" for s in raw_state]
    for st in state_categories:
        col_name = f"state_{st}"
        df_out[col_name] = [1 if cs == st else 0 for cs in clean_states]

    # Ensure all required features are present in exact order
    for col in feature_names:
        if col not in df_out.columns:
            df_out[col] = 0

    return df_out[feature_names].astype(float)


def preprocess_stage2(input_data, metadata):
    """
    Preprocess raw input into the 22 exact features expected by stage2_xgboost.joblib
    """
    if isinstance(input_data, dict):
        df_raw = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        df_raw = pd.DataFrame(input_data)
    else:
        df_raw = input_data.copy()

    medians = metadata.get("medians", {})
    emp_map = metadata.get("mappings", {}).get("employment_duration_map", {})
    sub_grade_map = metadata.get("mappings", {}).get("sub_grade_map", {})
    purpose_categories = metadata.get("mappings", {}).get("purpose_categories", [])
    home_categories = metadata.get("mappings", {}).get("home_ownership_categories", [])
    rules = metadata.get("purpose_regex_rules", [])
    feature_names = metadata.get("feature_names", [])

    df_out = pd.DataFrame(index=df_raw.index)

    # 1. loan_amnt
    loan_val = pd.to_numeric(df_raw.get("loan_amnt", df_raw.get("loan_amount", np.nan)), errors="coerce")
    df_out["loan_amnt"] = loan_val.fillna(medians.get("loan_amnt", 12500.0))

    # 2. int_rate
    raw_rate = df_raw.get("int_rate", np.nan)
    rates = [clean_percentage(v) for v in raw_rate]
    df_out["int_rate"] = pd.Series(rates, index=df_raw.index).fillna(medians.get("int_rate", 11.99))

    # 3. annual_inc
    inc_val = pd.to_numeric(df_raw.get("annual_inc", np.nan), errors="coerce")
    df_out["annual_inc"] = inc_val.fillna(medians.get("annual_inc", 65000.0))

    # 4. dti
    raw_dti = df_raw.get("dti", df_raw.get("debt_to_income_ratio", np.nan))
    dti_vals = [clean_percentage(v) for v in raw_dti]
    dti_series = pd.Series(dti_vals, index=df_raw.index)
    dti_series = dti_series.replace([-1, 9999], np.nan)
    dti_series.loc[dti_series > 100] = np.nan
    df_out["dti"] = dti_series.fillna(medians.get("dti", 18.41))

    # 5. revol_util
    raw_revol = df_raw.get("revol_util", np.nan)
    revol_vals = [clean_percentage(v) for v in raw_revol]
    df_out["revol_util"] = pd.Series(revol_vals, index=df_raw.index).fillna(medians.get("revol_util", 52.3))

    # 6. mort_acc
    mort_val = pd.to_numeric(df_raw.get("mort_acc", np.nan), errors="coerce")
    df_out["mort_acc"] = mort_val.fillna(medians.get("mort_acc", 1.0))

    # 7. dti_band
    dti_bins = metadata.get("dti_bins", [-float("inf"), 10.0, 20.0, 30.0, 40.0, float("inf")])
    df_out["dti_band"] = pd.cut(df_out["dti"], bins=dti_bins, labels=False).fillna(1).astype(int)

    # 8. employment_duration
    raw_emp = df_raw.get("employment_duration", df_raw.get("emp_length", np.nan))
    df_out["employment_duration"] = [
        clean_employment(v, emp_map, medians.get("employment_duration", 6.0)) for v in raw_emp
    ]

    # 9. fico_score
    if "fico_score" in df_raw and df_raw["fico_score"].notna().any():
        f_val = pd.to_numeric(df_raw["fico_score"], errors="coerce")
        df_out["fico_score"] = f_val.fillna(medians.get("fico_score", 687.0))
    elif "fico_range_low" in df_raw and "fico_range_high" in df_raw:
        low = pd.to_numeric(df_raw["fico_range_low"], errors="coerce")
        high = pd.to_numeric(df_raw["fico_range_high"], errors="coerce")
        df_out["fico_score"] = ((low + high) / 2.0).fillna(medians.get("fico_score", 687.0))
    else:
        df_out["fico_score"] = medians.get("fico_score", 687.0)

    # 10. term_months
    raw_term = df_raw.get("term", df_raw.get("term_months", 36))
    extracted_terms = []
    for t in raw_term:
        if pd.isna(t):
            extracted_terms.append(36.0)
        elif isinstance(t, (int, float)):
            extracted_terms.append(float(t))
        else:
            match = re.search(r'(\d+)', str(t))
            extracted_terms.append(float(match.group(1)) if match else 36.0)
    df_out["term_months"] = extracted_terms

    # 11. purpose_group one-hot encoding
    raw_purpose = df_raw.get("purpose", "")
    mapped_purposes = [map_purpose(p, rules) for p in raw_purpose]
    for cat in purpose_categories:
        col_name = f"purpose_group_{cat}"
        df_out[col_name] = [1 if mp == cat else 0 for mp in mapped_purposes]

    # 12. home_ownership one-hot encoding
    raw_home = df_raw.get("home_ownership", "RENT")
    cleaned_homes = []
    for h in raw_home:
        h_str = str(h).strip().upper() if pd.notna(h) else "RENT"
        if h_str in ["ANY", "NONE"]:
            h_str = "OTHER"
        if h_str not in home_categories:
            h_str = "OTHER"
        cleaned_homes.append(h_str)

    for h_cat in home_categories:
        col_name = f"home_ownership_{h_cat}"
        df_out[col_name] = [1 if ch == h_cat else 0 for ch in cleaned_homes]

    # 13. sub_grade_num
    raw_sg = df_raw.get("sub_grade", "C3")
    mapped_sg = []
    for sg in raw_sg:
        sg_str = str(sg).strip().upper() if pd.notna(sg) else "C3"
        mapped_sg.append(sub_grade_map.get(sg_str, medians.get("sub_grade_num", 15)))
    df_out["sub_grade_num"] = mapped_sg

    # Ensure all required features are present in exact order
    for col in feature_names:
        if col not in df_out.columns:
            df_out[col] = 0

    return df_out[feature_names].astype(float)
