# app.py
import os, json, difflib, requests, re, random
import chainlit as cl
import pandas as pd, joblib, numpy as np
from sentence_transformers import SentenceTransformer, util

# Explainability
import shap
import dice_ml

# =========================
# Paths & artifacts
# =========================
ENC_PATH   = os.getenv("ENC_PATH",   "models/encoder.pkl")
LGBM_PATH  = os.getenv("LGBM_PATH",  "models/lgbm_model.pkl")
COL_PATH   = os.getenv("COL_PATH",   "models/columns.json")  # optional; will sync with encoder if mismatched
BG_PATH    = os.getenv("BG_PATH",    "data/background.csv")  # background data for DiCE

encoder = joblib.load(ENC_PATH)
lgbm    = joblib.load(LGBM_PATH)

# Silence LightGBM duplicate-regularization warnings across versions
try:
    if hasattr(lgbm, "get_params"):
        _p = lgbm.get_params()
        keep_lambda = any(k in _p for k in ("lambda_l1", "lambda_l2"))
        if keep_lambda:
            lgbm.set_params(reg_alpha=None, reg_lambda=None)
        else:
            lgbm.set_params(lambda_l1=None, lambda_l2=None)
except Exception:
    pass

# Load FEATURES (prefer encoder.feature_names_in_)
try:
    FEATURES = json.load(open(COL_PATH, "r", encoding="utf-8"))
except Exception:
    FEATURES = []
MODEL_FEATURES = list(getattr(encoder, "feature_names_in_", [])) or FEATURES[:]
if FEATURES != MODEL_FEATURES:
    print("[WARN] columns.json != encoder.feature_names_in_. Using encoder.feature_names_in_.")
    FEATURES = MODEL_FEATURES[:]

# Categories per feature
try:
    CATS = {col: list(cats) for col, cats in zip(MODEL_FEATURES, encoder.categories_)}
except Exception:
    CATS = {col: [] for col in MODEL_FEATURES}

# =========================
# Friendly parsing & priorities
# =========================
ALIAS = {
    "Gender": {"f": "Female", "female": "Female", "m": "Male", "male": "Male", "other": "Other"},
    "Stress_Levels": {
        "no stress": "No stress", "none": "No stress",
        "mild": "Mild", "moderate": "Moderate", "avg": "Moderate",
        "severe": "Severe", "high": "Severe"
    },
}

PRIORITY_BASE = [
    "Stress_Levels","Sleep_Quality","Daytime_Fatigue","Missed_Classes","Impact_on_Deadlines",
    "Sleep_Hours","Difficulty_Falling_Asleep","Device_Use_Before_Sleep",
    "Physical_Activity","Wakeup_Difficulty","Caffeine_Consumption","Concentration_Problems",
    "Year_of_Study","Gender"
]
PRIORITY = [f for f in PRIORITY_BASE if f in FEATURES] + [f for f in FEATURES if f not in PRIORITY_BASE]

# =========================
# LLM extractor (Ollama)
# =========================
OLLAMA_URL      = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
LLM_MODEL       = os.getenv("LLM_MODEL", "llama3.2:3b")
OLLAMA_TIMEOUT  = int(os.getenv("OLLAMA_TIMEOUT", "180"))
OLLAMA_RETRIES  = int(os.getenv("OLLAMA_RETRIES", "2"))

def ask_ollama(prompt: str, fmt_json=False) -> dict | str:
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "options": {"temperature": 0, "num_ctx": 512, "num_predict": 96},
        "stream": False
    }
    if fmt_json:
        payload["format"] = "json"
    for attempt in range(1, OLLAMA_RETRIES + 1):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            text = (data.get("response") or "").strip()
            if fmt_json:
                try:
                    return json.loads(text)
                except Exception:
                    i, j = text.find("{"), text.rfind("}")
                    return json.loads(text[i:j+1]) if i != -1 and j != -1 else {}
            return text
        except Exception:
            if attempt == OLLAMA_RETRIES:
                return {} if fmt_json else ""

def build_extract_system() -> str:
    keys_csv = ", ".join(FEATURES)
    cats_txt = json.dumps(CATS, ensure_ascii=False)
    examples = [
        ("I am female", {"Gender": "Female"}),
        ("no stress and very poor sleep", {"Stress_Levels":"No stress","Sleep_Quality":"Very poor"}),
        ("Male, year 2, often use phone before sleep", {"Gender":"Male","Year_of_Study":"2","Device_Use_Before_Sleep":"Often"})
    ]
    ex_text = "\n".join(
        f"User: '{u}'\nAssistant(JSON): {json.dumps(a, ensure_ascii=False)}"
        for u, a in examples
    )
    return (
        "Task: Extract a STRICT JSON with exactly these keys:\n"
        f"{keys_csv}.\n\n"
        "Rules:\n"
        f"- Allowed values per key (choose EXACTLY one if known, else null): {cats_txt}\n"
        "- User may write free text/synonyms/any case. Map to the CLOSEST allowed token.\n"
        "- Output JSON ONLY. No extra text.\n\n"
        "Examples:\n"
        f"{ex_text}\n"
    )

EXTRACT_SYSTEM = build_extract_system()

def llm_extract_fields(user_text: str) -> dict:
    resp = ask_ollama(f"System: {EXTRACT_SYSTEM}\nUser: {user_text}\nAssistant:", fmt_json=True)
    if not isinstance(resp, dict) or not resp:
        return {}
    out = {}
    for k in FEATURES:
        v = resp.get(k)
        out[k] = (v if v not in [None, "null", ""] else None)
    return out

# =========================
# Normalization (exact → alias → lowercase → fuzzy → semantic)
# =========================
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", "0.48"))

def apply_alias(col: str, val: str):
    if val is None:
        return None
    amap = ALIAS.get(col, {})
    vlow = str(val).strip().lower()
    return amap.get(vlow, None)

def smart_normalize(col: str, val: str):
    if val is None:
        return None
    choices = CATS.get(col, [])
    if not choices:
        return val
    if val in choices:
        return val
    ali = apply_alias(col, val)
    if ali and ali in choices:
        return ali
    low_map = {str(c).lower(): c for c in choices}
    got = low_map.get(str(val).lower())
    if got:
        return got
    cand = difflib.get_close_matches(str(val), choices, n=1, cutoff=0.6)
    if cand:
        return cand[0]
    try:
        emb1 = embed_model.encode(str(val), convert_to_tensor=True)
        emb2 = embed_model.encode(choices, convert_to_tensor=True)
        sim = util.cos_sim(emb1, emb2)
        best = int(sim.argmax()); score = float(sim.max())
        return choices[best] if score >= SEMANTIC_THRESHOLD else None
    except Exception:
        return None

def merge_and_clean(collected: dict) -> dict:
    return {c: smart_normalize(c, collected.get(c)) if collected.get(c) is not None else None for c in FEATURES}

def missing_fields(collected: dict) -> list:
    return [c for c in PRIORITY if collected.get(c) in [None, ""]]

# =========================
# Labels & helpers
# =========================
DEF_LABELS = {f: f.replace("_"," ").title() for f in FEATURES}
def human_field(f: str) -> str: return DEF_LABELS.get(f, f)
def summarize_updates(old: dict, new: dict):
    return [(k,new[k]) for k in FEATURES if new.get(k) is not None and new.get(k)!=old.get(k)]
def format_missing_prompt(miss: list, limit: int=3) -> str:
    ask = miss[:limit]
    lines = []
    for f in ask:
        lines.append(f"- **{human_field(f)}**: {CATS.get(f, [])}")
    return "Still missing some info, please provide:\n" + "\n".join(lines)

def _nice_pct(p: float) -> str:
    try:
        return f"{int(round(p*100))}%"
    except Exception:
        return f"{p:.0%}"

def _split_contribs_plain(items, top=3):
    inc = [(f, v, s) for f, v, _, s in items if s > 0][:top]
    dec = [(f, v, s) for f, v, _, s in items if s < 0][:top]
    return inc, dec

def make_action(name: str, label: str):
    """Cross-version safe action (payload for newer Chainlit, fallback to value)."""
    try:
        return cl.Action(name=name, label=label, payload={"cmd": name})
    except Exception:
        return cl.Action(name=name, label=label, value=name)

# ---------- Auto tips (no manual per-case authoring) ----------
FREQ_ORDER   = ["Never","Rarely","Sometimes","Often","Always"]
QUAL_ORDER   = ["Very poor","Poor","Average","Good","Excellent"]
STRESS_ORDER = ["No stress","Mild","Moderate","Severe"]

ORDER_MAP_HINTS = {
    "Device_Use_Before_Sleep": FREQ_ORDER,
    "Caffeine_Consumption": FREQ_ORDER,
    "Physical_Activity": FREQ_ORDER,
    "Difficulty_Falling_Asleep": FREQ_ORDER,
    "Wakeup_Difficulty": FREQ_ORDER,
    "Concentration_Problems": FREQ_ORDER,
    "Daytime_Fatigue": FREQ_ORDER,
    "Missed_Classes": FREQ_ORDER,
    "Sleep_Quality": QUAL_ORDER,
    "Impact_on_Deadlines": QUAL_ORDER,
    "Stress_Levels": STRESS_ORDER
}

WORSE_WHEN_HIGHER  = {
    "Device_Use_Before_Sleep","Caffeine_Consumption","Difficulty_Falling_Asleep",
    "Wakeup_Difficulty","Concentration_Problems","Daytime_Fatigue",
    "Missed_Classes","Stress_Levels","Impact_on_Deadlines"
}
BETTER_WHEN_HIGHER = {"Sleep_Quality","Physical_Activity"}

def _order_index(feature: str, value) -> int | None:
    if value is None:
        return None
    order = ORDER_MAP_HINTS.get(feature)
    if not order:
        return None
    s = str(value).strip()
    for i, tok in enumerate(order):
        if s == tok:
            return i
    low_map = {t.lower(): i for i, t in enumerate(order)}
    if s.lower() in low_map:
        return low_map[s.lower()]
    cand = difflib.get_close_matches(s, order, n=1, cutoff=0.6)
    return order.index(cand[0]) if cand else None

def _parse_hours(val) -> float | None:
    if val is None:
        return None
    s = str(val)
    nums = re.findall(r"[0-9]+(?:\.[0-9]+)?", s)
    if nums:
        return float(nums[0])
    if "less" in s.lower():
        return 5.0
    if "more" in s.lower() or "over" in s.lower():
        return 10.0
    return None

def get_tip(feature: str, old, new, row: dict) -> str:
    """Generate one short, practical tip; rule-based first, fallback to LLM."""
    f = human_field(feature)
    if feature == "Sleep_Hours":
        ho, hn = _parse_hours(old), _parse_hours(new)
        if ho is not None and hn is not None:
            if hn < 7:
                return "Go to bed 30–60 minutes earlier and keep a consistent schedule."
            if 7 <= hn <= 9:
                return "Protect a steady 7–9 hour sleep window on weekdays and weekends."
            if hn > 9:
                return "Limit long lie-ins; get daylight within 1 hour of waking."
        return "Aim for a steady 7–9 hours most nights."

    i_old, i_new = _order_index(feature, old), _order_index(feature, new)
    if i_old is not None and i_new is not None:
        if feature in WORSE_WHEN_HIGHER and i_new < i_old:
            return f"Reduce {f} step by step; set a clear cutoff and use reminders."
        if feature in BETTER_WHEN_HIGHER and i_new > i_old:
            return f"Increase {f} gradually; schedule small, regular sessions."

    try:
        prompt = (f"Give one short, practical tip (max 18 words) to help a student move "
                  f"{f} from '{old}' to '{new}' to improve sleep or study.")
        t = ask_ollama(prompt, fmt_json=False)
        t = (t or "").strip()
        if t:
            return t
    except Exception:
        pass
    return "Make a small, consistent change toward the new setting for two weeks, then reassess."
# ---------- end auto tips ----------

# =========================
# Encoding + prediction
# =========================
def _wrap_encoded_with_columns(X_enc: np.ndarray) -> pd.DataFrame:
    train_cols = getattr(lgbm, "feature_names_in_", None)
    if train_cols is not None and len(train_cols) == X_enc.shape[1]:
        return pd.DataFrame(X_enc, columns=list(train_cols))
    return pd.DataFrame(X_enc, columns=[f"f{i}" for i in range(X_enc.shape[1])])

def _encode_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    X_in  = df_raw[MODEL_FEATURES]
    X_enc = encoder.transform(X_in)
    return _wrap_encoded_with_columns(X_enc)

def predict_label(df_raw: pd.DataFrame) -> int:
    X_enc_df = _encode_df(df_raw)
    want  = getattr(lgbm, "n_features_in_", None)
    if want is not None and X_enc_df.shape[1] != want:
        raise ValueError(
            f"Encoded dim={X_enc_df.shape[1]} but model expects {want}. "
            f"Mismatch between encoder.pkl and lgbm_model.pkl. "
            f"MODEL_FEATURES={MODEL_FEATURES}"
        )
    y = lgbm.predict(X_enc_df)
    return int(y[0])

def predict_proba_df(df_raw: pd.DataFrame) -> float:
    X_enc_df = _encode_df(df_raw)
    return float(lgbm.predict_proba(X_enc_df)[0, 1])

# =========================
# SHAP (cross-version safe)
# =========================
TREE_EXPLAINER = None
try:
    TREE_EXPLAINER = shap.TreeExplainer(lgbm)
except Exception:
    TREE_EXPLAINER = None

def compute_shap_values(df_raw: pd.DataFrame) -> tuple[list[tuple], float]:
    """
    Cross-version safe:
    - Supports shap_values as list (multi-class) or ndarray (binary).
    - expected_value may be scalar, array, or list.
    - Returns (items, base_value).
    """
    X_enc_df = _encode_df(df_raw)

    global TREE_EXPLAINER
    if TREE_EXPLAINER is None:
        try:
            TREE_EXPLAINER = shap.TreeExplainer(lgbm)
        except Exception:
            TREE_EXPLAINER = None

    sv_matrix = None
    base_value = 0.0

    if TREE_EXPLAINER is not None:
        shap_vals = TREE_EXPLAINER.shap_values(X_enc_df)
        if isinstance(shap_vals, list):
            sv_matrix = np.array(shap_vals[-1])  # positive class
        else:
            sv_matrix = np.array(shap_vals)
        if sv_matrix.ndim == 1:
            sv_matrix = sv_matrix[None, :]

        ev = TREE_EXPLAINER.expected_value
        if isinstance(ev, (list, tuple, np.ndarray)):
            ev_arr = np.array(ev).reshape(-1)
            base_value = float(ev_arr[-1])
        else:
            base_value = float(ev)
    else:
        bg = X_enc_df.sample(min(50, len(X_enc_df)), replace=True, random_state=42)
        explainer = shap.KernelExplainer(lgbm.predict_proba, bg)
        all_sv = explainer.shap_values(X_enc_df)
        if isinstance(all_sv, list):
            sv_matrix = np.array(all_sv[-1])
        else:
            sv_matrix = np.array(all_sv)
        if sv_matrix.ndim == 1:
            sv_matrix = sv_matrix[None, :]
        base_value = float(np.mean(lgbm.predict_proba(bg)[:, 1]))

    sv = sv_matrix[0]
    enc_cols = list(X_enc_df.columns)
    raw_vals = {c: df_raw.iloc[0][c] for c in df_raw.columns}

    contrib = [(col, None, abs(float(sv[i])), float(sv[i])) for i, col in enumerate(enc_cols)]
    grouped = {}
    for name, _, abs_v, signed_v in contrib:
        root = name.split("__")[0] if "__" in name else name
        g = grouped.setdefault(root, {"abs": 0.0, "signed": 0.0})
        g["abs"] += abs_v
        g["signed"] += signed_v

    items = [(root, raw_vals.get(root, None), d["abs"], d["signed"]) for root, d in grouped.items()]
    items.sort(key=lambda x: x[2], reverse=True)
    return items, base_value

# =========================
# DiCE setup
# =========================
IMMUTABLE = set(os.getenv("IMMUTABLE", "Gender,Year_of_Study").split(","))
IMMUTABLE = {s.strip() for s in IMMUTABLE if s.strip()}
ACTIONABLE = [f for f in FEATURES if f not in IMMUTABLE]

def _normalize_background(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for c in FEATURES:
        if c in df2.columns:
            df2[c] = df2[c].apply(lambda v: smart_normalize(c, v))
    return df2.dropna().reset_index(drop=True)

def load_background_df() -> pd.DataFrame:
    if os.path.exists(BG_PATH):
        try:
            df = pd.read_csv(BG_PATH)
            for c in FEATURES:
                if c not in df.columns:
                    df[c] = None
            df = df[FEATURES].copy()
            return _normalize_background(df)
        except Exception:
            pass
    rows = []
    for _ in range(800):
        row = {}
        for c in FEATURES:
            choices = CATS.get(c, [])
            row[c] = random.choice(choices) if choices else None
        rows.append(row)
    return _normalize_background(pd.DataFrame(rows))

BACKGROUND_DF = load_background_df()

class EncodedModelWrapper:
    def __init__(self, encoder, model, feature_order):
        self.encoder = encoder
        self.model = model
        self.feature_order = feature_order
        self.feature_names_in_ = feature_order

    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_order)
        return _encode_df(X[self.feature_order])

    def predict(self, X):
        X_enc = self._prep(pd.DataFrame(X, columns=self.feature_order))
        return self.model.predict(X_enc)

    def predict_proba(self, X):
        X_enc = self._prep(pd.DataFrame(X, columns=self.feature_order))
        return self.model.predict_proba(X_enc)

def build_dice_data(background_df: pd.DataFrame) -> dice_ml.Data:
    df = background_df.dropna().copy()
    if df.empty:
        df = pd.DataFrame([{c: None for c in FEATURES}])

    probs = [predict_proba_df(pd.DataFrame([df.iloc[i].to_dict()])) for i in range(len(df))]
    df["risk"] = (np.array(probs) >= 0.5).astype(int)

    continuous = []  # declare numeric features here if any
    return dice_ml.Data(
        dataframe=df[FEATURES + ["risk"]],
        continuous_features=continuous,
        outcome_name="risk"
    )

DICE_DATA  = build_dice_data(BACKGROUND_DF)
DICE_MODEL = dice_ml.Model(model=EncodedModelWrapper(encoder, lgbm, MODEL_FEATURES), backend="sklearn")

def generate_counterfactuals(factual_row: dict, total_CFs=5, desired_class="opposite"):
    feats = [f for f in ACTIONABLE if f in FEATURES]
    query_instance = pd.DataFrame([factual_row])[FEATURES]
    attempts = [
        {"method": "genetic", "kwargs": dict(proximity_weight=0.5, diversity_weight=0.1)},
        {"method": "genetic", "kwargs": dict(proximity_weight=0.2, diversity_weight=0.3)},
        {"method": "random",  "kwargs": dict()},
    ]
    for attempt in attempts:
        try:
            dice = dice_ml.Dice(DICE_DATA, DICE_MODEL, method=attempt["method"])
            cf = dice.generate_counterfactuals(
                query_instance,
                total_CFs=total_CFs,
                desired_class=desired_class,
                features_to_vary=feats,
                **attempt["kwargs"],
            )
            if cf and getattr(cf, "cf_examples_list", None):
                valid = [
                    ex for ex in cf.cf_examples_list
                    if getattr(ex, "final_cfs_df", None) is not None and not ex.final_cfs_df.empty
                ]
                if valid:
                    return cf
        except Exception as e:
            print(f"DiCE attempt failed ({attempt['method']}):", e)
            continue
    return None

# =========================
# Chainlit helpers
# =========================
def get_state():
    state = cl.user_session.get("collected")
    if state is None:
        state = {k: None for k in FEATURES}
        cl.user_session.set("collected", state)
    return state

def parse_kv_pairs(s: str) -> dict:
    if " " in s:
        s = s.split(" ", 1)[1]
    parts = re.split(r"[,\s]+", s.strip())
    out = {}
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k.strip()] = v.strip()
    return out

# =========================
# Action callbacks (Explain / Advice / No thanks)
# =========================
def _plain_explanation(col: dict) -> str:
    df_tmp = pd.DataFrame([col])[FEATURES]
    items, _ = compute_shap_values(df_tmp)
    prob = predict_proba_df(df_tmp)
    inc, dec = _split_contribs_plain(items, top=3)

    def line(f, v, s):
        arrow = "raises risk" if s > 0 else "lowers risk"
        return f"- {human_field(f)} — you answered `{v}`. This {arrow}."

    parts = []
    parts.append("**Why you got this result**")
    parts.append(f"- Current chance of risk: **{_nice_pct(prob)}**")
    if inc:
        parts.append("\n**Biggest things pushing your risk up:**")
        parts += [line(f, v, s) for f, v, s in inc]
    if dec:
        parts.append("\n**Things helping you right now:**")
        parts += [line(f, v, s) for f, v, s in dec]
    parts.append("\n*Meaning:* these items had the most influence based on similar people in the data.")
    return "\n".join(parts)

def _plain_advice(col: dict) -> str:
    factual = {k: col.get(k) for k in FEATURES}
    df_fact = pd.DataFrame([factual])[FEATURES]
    p0 = predict_proba_df(df_fact)
    cf_res = generate_counterfactuals(factual, total_CFs=5, desired_class="opposite")
    if cf_res is None or not getattr(cf_res, "cf_examples_list", []):
        # Fallback: propose SHAP-driven 1-step shifts along good direction
        items, _ = compute_shap_values(pd.DataFrame([factual])[FEATURES])
        inc = [(f, v, s) for f, v, _, s in items if s > 0][:2]
        if not inc:
            return ("I could not find good, realistic changes yet.\n"
                    "- Make sure `data/background.csv` looks like real student data.\n"
                    "- Consider relaxing IMMUTABLE (e.g., keep only `Gender`) and try again.")
        lines = []
        for f, v, _ in inc:
            order = ORDER_MAP_HINTS.get(f)
            if order:
                i = _order_index(f, v)
                if i is not None:
                    if f in WORSE_WHEN_HIGHER and i > 0:
                        nv = order[i-1]
                    elif f in BETTER_WHEN_HIGHER and i < len(order)-1:
                        nv = order[i+1]
                    else:
                        nv = v
                else:
                    nv = v
            else:
                nv = v
            tip = get_tip(f, v, nv, factual)
            lines.append(f"- Try shifting **{human_field(f)}** from `{v}` to `{nv}`. Tip: {tip}")
        return ("**Simple advice you can try**\n"
                "These are based on the biggest drivers of your risk right now:\n\n" +
                "\n".join(lines) +
                "\n\nTry a change for 1–2 weeks, then reassess.")

    blocks, kept = [], 0
    for i, ex in enumerate(cf_res.cf_examples_list):
        if ex.final_cfs_df is None or ex.final_cfs_df.empty:
            continue
        row = ex.final_cfs_df.iloc[0].to_dict()
        p_new = predict_proba_df(pd.DataFrame([row])[FEATURES])
        changes = [(k, factual.get(k), row.get(k)) for k in FEATURES
                   if factual.get(k) != row.get(k) and k not in ("Gender","Year_of_Study")]
        if not changes:
            continue
        kept += 1
        changes = changes[:4]
        lines = []
        for k, ov, nv in changes:
            tip = get_tip(k, ov, nv, factual)
            msg = f"- Change **{human_field(k)}** from `{ov}` to `{nv}`. Tip: {tip}"
            lines.append(msg)
        blocks.append(
            f"**Option {kept}** — risk may move from **{_nice_pct(p0)}** to **{_nice_pct(p_new)}**:\n" +
            "\n".join(lines)
        )
        if kept == 3:
            break
    header = ("**Simple advice you can try**\n"
              "Pick one option below and try it for 1–2 weeks. Small, consistent changes work best.")
    footer = ("\n\nTo test your own idea, type: `whatif Feature=Value` "
              "(e.g., `whatif Device_Use_Before_Sleep=Never`).")
    return header + "\n\n" + ("\n\n".join(blocks) if blocks else "No suitable options yet.") + footer

@cl.action_callback("explain")
async def on_explain(action):
    col = cl.user_session.get("collected") or {}
    miss = [c for c in FEATURES if col.get(c) in [None, ""]]
    if miss:
        await cl.Message(content="I still need a bit more info:\n" + "\n".join(f"- {human_field(m)}" for m in miss)).send()
        return
    try:
        await cl.Message(content=_plain_explanation(col)).send()
    except Exception as e:
        await cl.Message(content=f"Sorry, I could not build a simple explanation: {e}").send()

@cl.action_callback("advice")
async def on_advice(action):
    col = cl.user_session.get("collected") or {}
    miss = [c for c in FEATURES if col.get(c) in [None, ""]]
    if miss:
        await cl.Message(content="I need a bit more info first:\n" + "\n".join(f"- {human_field(m)}" for m in miss)).send()
        return
    try:
        await cl.Message(content=_plain_advice(col)).send()
    except Exception as e:
        await cl.Message(content=f"Sorry, I could not build clear suggestions: {e}").send()

@cl.action_callback("noop")
async def on_no_thanks(action):
    await cl.Message(content="Okay. If you change your mind, type `shap` for an explanation or `cf` for advice.").send()

# =========================
# Chainlit entry points
# =========================
@cl.on_chat_start
async def start():
    print("encoder.feature_names_in_ (MODEL_FEATURES) ->", MODEL_FEATURES)
    print("lgbm.n_features_in_                        ->", getattr(lgbm, "n_features_in_", None))
    print("lgbm.feature_names_in_                     ->", getattr(lgbm, "feature_names_in_", None))
    print("FEATURES (effective)                       ->", FEATURES)

    cl.user_session.set("collected", {k: None for k in FEATURES})
    await cl.Message(
        content=(
            "Hello! Please describe your study/sleep habits in free-form English. "
            "I will extract the necessary fields and ask for anything missing.\n\n"
            f"You can also send JSON with these keys: {', '.join(FEATURES)}.\n\n"
            "Commands:\n"
            "- `state` → show current state\n"
            "- `proba` → risk probability\n"
            "- `shap` or `explain` → plain-language explanation\n"
            "- `whatif Feature=Value ...` → simulate changes\n"
            "- `cf` or `counterfactual` → simple advice (DiCE)"
        )
    ).send()

@cl.on_message
async def main(msg: cl.Message):
    prev = get_state()
    col  = prev.copy()
    txt  = (msg.content or "").strip()
    low  = txt.lower().strip()

    # Commands
    if low == "state":
        pretty = {k: col.get(k) for k in FEATURES}
        await cl.Message(content=f"**Current state (all features):**\n```\n{json.dumps(pretty, ensure_ascii=False, indent=2)}\n```").send()
        return

    if low == "proba":
        miss = missing_fields(col)
        if miss:
            await cl.Message(content="I still need these to compute probability:\n" + "\n".join(f"- {human_field(m)}" for m in miss)).send()
            return
        try:
            df_tmp = pd.DataFrame([col])[FEATURES]
            prob = predict_proba_df(df_tmp)
            await cl.Message(content=f"**Risk probability:** `{_nice_pct(prob)}`").send()
        except Exception as e:
            await cl.Message(content=f"Prediction error (proba): {e}").send()
        return

    if low in ("shap", "explain"):
        miss = missing_fields(col)
        if miss:
            await cl.Message(content="I still need a bit more info before explaining:\n" + "\n".join(f"- {human_field(m)}" for m in miss)).send()
            return
        try:
            await cl.Message(content=_plain_explanation(col)).send()
        except Exception as e:
            await cl.Message(content=f"Sorry, I could not build a simple explanation: {e}").send()
        return

    if low.startswith("whatif"):
        changes = parse_kv_pairs(txt)
        if not changes:
            await cl.Message(content="Syntax: `whatif Feature=Value [Feature2=Value2 ...]`").send()
            return
        new_col = col.copy()
        for k, v in changes.items():
            if k in FEATURES:
                new_col[k] = v
        new_col = merge_and_clean(new_col)

        miss = missing_fields(new_col)
        if miss:
            await cl.Message(content="After changes, I still need:\n" + "\n".join(f"- {human_field(m)}" for m in miss)).send()
            return

        try:
            df_old = pd.DataFrame([col])[FEATURES]
            df_new = pd.DataFrame([new_col])[FEATURES]
            p_old = predict_proba_df(df_old)
            p_new = predict_proba_df(df_new)

            upd = [(k, col.get(k), new_col.get(k)) for k in FEATURES if new_col.get(k) != col.get(k)]
            lines = [f"- **{human_field(k)}**: `{ov}` → `{nv}`" for k, ov, nv in upd]
            await cl.Message(
                content=("**What-if**\n" + "\n".join(lines) +
                         f"\n\nChance of risk: `{_nice_pct(p_old)}` → **`{_nice_pct(p_new)}`**\n"
                         "If that looks good, try it for a week and check again.")
            ).send()
        except Exception as e:
            await cl.Message(content=f"What-if error: {e}").send()
        return

    if low in ("cf", "counterfactual"):
        miss = missing_fields(col)
        if miss:
            await cl.Message(content="I need a bit more info before suggesting changes:\n" + "\n".join(f"- {human_field(m)}" for m in miss)).send()
            return
        try:
            await cl.Message(content=_plain_advice(col)).send()
        except Exception as e:
            await cl.Message(content=f"Counterfactual error: {e}").send()
        return

    # Free text (or JSON) → extract → normalize → persist
    parsed = {}
    try:
        maybe = json.loads(msg.content)
        if isinstance(maybe, dict):
            parsed = {k: maybe.get(k) for k in FEATURES}
    except Exception:
        parsed = llm_extract_fields(msg.content)

    for k, v in (parsed or {}).items():
        if v is not None:
            col[k] = v
    col = merge_and_clean(col)
    cl.user_session.set("collected", col)

    # Report updates and prompt for missing
    updates = summarize_updates(prev, col)
    parts = []
    if updates:
        ok_lines = [f"**{human_field(f)}** → `{v}`" for f, v in updates]
        parts.append("Updated fields:\n" + "\n".join(ok_lines))

    miss = missing_fields(col)
    if miss:
        parts.append(format_missing_prompt(miss))
        await cl.Message(content="\n\n".join(parts)).send()
        return

    # Predict and ask next step
    df = pd.DataFrame([col])[FEATURES]
    try:
        y = predict_label(df)
        verdict = "No risk" if y == 0 else "At risk"
        await cl.Message(content=(("\n\n".join(parts) + "\n\n") if parts else "") + f"**Result:** {verdict}").send()
        await cl.Message(
            content="Would you like me to explain the result or give you advice?",
            actions=[make_action("explain", "Explain"),
                     make_action("advice",  "Give advice"),
                     make_action("noop",    "No thanks")],
        ).send()
    except Exception as e:
        await cl.Message(content=f"Prediction error: {e}").send()
