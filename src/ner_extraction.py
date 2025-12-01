"""
ner_extraction.py
Named Entity Recognition for drugs and ADR symptoms from nursing records
- Creates prompts for OpenAI API
- Extracts drug entities and ADR symptoms
"""

import os
import re
import json
import time
import pandas as pd
from collections import Counter
from tqdm import tqdm
from openai import OpenAI

# ==========================================
# Configuration
# ==========================================
INPUT_FILE = 'nursing_records_preprocessed.csv'
OUTPUT_DIR = 'ner_results'
MODEL_NAME = 'gpt-4.1'
API_KEY = 'your-api-key'

# Column names
WINDOW_COL = 'window_id_12h'
PTNO_COL = 'ptno'
RECORD_COL = 'record_no'
DT_COL = 'dt'
TEXT_COL = 'text'
TIME_COL = 'time'
SEQ_COL = 'sequence'
DATE_COL = 'date'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY)

# ==========================================
# Helper Functions
# ==========================================
def safe_filename(s: str) -> str:
    s = str(s)
    s = re.sub(r'[^0-9A-Za-z_\-\.]+', '_', s)
    return s.strip('_') or "NA"

def pick_majority(series) -> str:
    vals = [str(v) for v in series if pd.notna(v)]
    if not vals: return None
    return Counter(vals).most_common(1)[0][0]

def to_abs_time(s) -> str:
    if pd.isna(s): return None
    try:
        d = pd.to_datetime(s)
        return d.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return None

def to_hhmm_from_cols(row) -> str:
    t = getattr(row, TIME_COL, None) if hasattr(row, TIME_COL) else None
    if t and str(t).strip(): return str(t)
    dt_val = getattr(row, DT_COL, None) if hasattr(row, DT_COL) else None
    if dt_val:
        try:
            return pd.to_datetime(dt_val).strftime("%H:%M")
        except:
            return None
    return None

def compute_window_bounds(df_win: pd.DataFrame):
    if DT_COL in df_win.columns:
        dts = pd.to_datetime(df_win[DT_COL], errors="coerce")
        s = dts.min(); e = dts.max()
        s_str = s.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(s) else None
        e_str = e.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(e) else None
        return s_str, e_str
    return None, None

def build_timeline_block(df_win: pd.DataFrame):
    sort_cols = []
    for c in [DATE_COL, DT_COL, TIME_COL, SEQ_COL]:
        if c in df_win.columns: sort_cols.append(c)
    df_order = df_win.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    lines_txt, source_index = [], []
    for i, row in enumerate(df_order.itertuples(index=False), start=1):
        lid = f"L{i}"
        time_str = to_hhmm_from_cols(row) or ""
        text_str = str(getattr(row, TEXT_COL))
        dt_str = to_abs_time(getattr(row, DT_COL, None)) if hasattr(row, DT_COL) else None
        lines_txt.append(f"{lid} {time_str} | {text_str}")
        source_index.append({"line_id": lid, "time": time_str, "dt": dt_str})
    return "\n".join(lines_txt), source_index

# ==========================================
# Prompt Generation
# ==========================================
def ner_system_prompt() -> str:
    return (
        "You are a clinical information extraction specialist. Your task is to extract two entity types from nursing record timelines "
        "and return strict JSON: (1) DRUG mentions and (2) ADR-related SYMPTOMS (observed or suspected symptoms related to predefined categories), "
        "each with its own timestamp. Do NOT output any other entity types. Do NOT invent facts. "
        "If a symptom is explicitly negated (e.g., '없음', 'no ...', '(-)'), DO NOT output that symptom at all."
    )

def ner_user_prompt(record_no, ptno, window_id, window_start, window_end, timeline_str) -> str:
    # Minimal schema — entity-level time only (no line/time index, no extra sections)
    output_schema = """\
{
  "window_meta": {
    "record_no": "string",
    "ptno": "string",
    "window_id": "string"
  },
  "drugs": [
    {
      "id": "string",
      "line_id": "L#",
      "surface": "string",              // raw phrase containing the drug
      "name": "string",                 // brand or generic as written (e.g., 'Norpin','vasopressine','Insulin','KCL','lasix')
      "ingredient_norm": "string|null", // normalize only if obvious; else null
      "action": "start|stop|increase|decrease|restart|ongoing|bolus|unknown",
      "dose_text": "string|null",       // keep as free text if parsing is uncertain
      "rate_text": "string|null",
      "mixture": ["string", "..."],     // carriers like TPN/NS/DW/5%DW/Plasma soln mentioned in same phrase
      "abs_time": "YYYY-MM-DD HH:MM:SS|null",  // entity-level time; derive from that line's dt/time
      "span": "string"
    }
  ],
  "symptoms": [
    {
      "id": "string",
      "line_id": "L#",
      "code": "FEVER|HYPOTENSION|LFT_ABNORMAL|LEUKOPENIA|THROMBOCYTOPENIA|AKI|HYPOGLYCEMIA|RHABDOMYOLYSIS|MUCOSAL_CUTANEOUS|RESPIRATORY",
      "surface": "string",              // minimal excerpt proving the symptom
      "values": { "...": "..." },       // include numeric values if present (e.g., BT, SBP/MAP, PLT, CK)
      "abs_time": "YYYY-MM-DD HH:MM:SS|null",  // entity-level time; derive from that line's dt/time
      "span": "string"
    }
  ]
}"""

    return f"""[WINDOW_META]
record_no: {record_no}
ptno: {ptno}
window_id: {window_id}

[TIMELINE]
{timeline_str}
[/TIMELINE]

[WHAT TO EXTRACT — EXACTLY]
Extract ONLY:
A) DRUGS: explicit drug mentions (brand or generic) with entity-level abs_time.
   - Examples: Norpin (norepinephrine), vasopressine, Insulin, KCL, lasix, denogan, adenocor …
   - Ignore carriers as drugs (NS, DW/5%DW, TPN, Plasma soln, 0.9% NaCl); if they appear in the same phrase, keep them in mixture[].
   - ACTION cues (Korean/English):
     * start: "투여함", "시행함", "start", "IV", "투여 시작"
     * stop: "중단", "cut", "off", "stop"
     * increase: "증량", "↑", "up", "증가"
     * decrease: "감량", "↓", "down", "감소"
     * restart: "재시작", "restart", "re-"
     * ongoing: "투여중", "유지", "mix", "혼합된 수액 투여중"
     * bolus: "bolus", "loading"
   - If action is unclear, set "unknown".
   - ingredient_norm: only if obvious from the surface; else null.
   - abs_time: derive from that line's absolute datetime (dt). If unavailable, derive from the line time within the window date; if uncertain, null.

B) SYMPTOMS (whitelist + numeric thresholds; entity-level abs_time):
   - FEVER: BT ≥ 38.0℃
   - HYPOTENSION: SBP < 90 OR MAP < 65
   - LFT_ABNORMAL: explicit abnormal LFT (ALT/AST ↑); include numbers if present
   - LEUKOPENIA: WBC < 4000 OR ANC < 1500
   - THROMBOCYTOPENIA: PLT < 150000
   - AKI: ΔSCr ≥ 0.3 mg/dL within 48h OR urine output < 0.5 mL/kg/h
   - HYPOGLYCEMIA: Glucose < 70 mg/dL
   - RHABDOMYOLYSIS: CK ≥ 5×ULN (or CK ≥ 1000 IU/L if ULN unknown)
   - MUCOSAL_CUTANEOUS: rash/urticaria/angioedema/mucosal lesions
   - RESPIRATORY: dyspnea/wheezing/stridor/SpO₂ drop documented

[STRICT NEGATION RULE — DO NOT OUTPUT NEGATED SYMPTOMS]
- If a symptom is explicitly negated (e.g., "없음", "(-)", "no …", "부작용 없음", "no fever sign"), DO NOT output that symptom at all.
- Example: "KCL … 부작용 없음" → extract the DRUG (likely 'ongoing'), but output NO SYMPTOM.

[OUTPUT SCHEMA — JSON ONLY]
{output_schema}

[RESTRICTIONS]
- Output ONLY the fields in the schema. No source_index, no time_events, no procedures, no measurements, no negation lists.
- Use only the information present in [TIMELINE]; do not guess.
- Output valid JSON only (no explanations or extra text).
"""

# ==========================================
# API Call
# ==========================================
def call_openai_json(system_msg: str, user_msg: str, retries=3):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.1
            )
            text = resp.choices[0].message.content
            m = re.search(r"\{[\s\S]*\}\s*$", text)
            return json.loads(m.group(0)) if m else {}
        except Exception as e:
            if attempt >= retries - 1: raise
            time.sleep(2 ** attempt)
    return {}

# ==========================================
# Main Pipeline
# ==========================================
def main():
    # Load data
    df = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')
    
    # Convert columns
    for c in [WINDOW_COL, PTNO_COL, RECORD_COL]:
        if c in df.columns: df[c] = df[c].astype(str)
    if DT_COL in df.columns:
        df[DT_COL] = pd.to_datetime(df[DT_COL], errors="coerce")
    
    # Get unique windows
    windows = df[WINDOW_COL].dropna().unique().tolist()
    
    success_count = 0
    error_count = 0
    
    # Process each window
    for window_id in tqdm(windows, desc="Processing NER"):
        df_win = df[df[WINDOW_COL] == window_id].copy()
        
        # Get metadata
        ptno = pick_majority(df_win[PTNO_COL]) if PTNO_COL in df_win.columns else "NA"
        record_no = pick_majority(df_win[RECORD_COL]) if RECORD_COL in df_win.columns else "NA"
        
        # Build timeline
        timeline_str, source_index = build_timeline_block(df_win)
        win_start, win_end = compute_window_bounds(df_win)
        
        # Create prompts
        system_msg = ner_system_prompt()
        user_msg = ner_user_prompt(record_no, ptno, window_id, win_start, win_end, timeline_str)
        
        try:
            # Call API
            model_output = call_openai_json(system_msg, user_msg)
            
            # Save result
            result = {
                "meta": {
                    "ptno": ptno,
                    "record_no": record_no,
                    "window_id": window_id,
                    "window_start": win_start,
                    "window_end": win_end
                },
                "source_index": source_index,
                "model_output": model_output
            }
            
            filename = f"ptno_{safe_filename(ptno)}__record_{safe_filename(record_no)}__win_{safe_filename(window_id)}.json"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            success_count += 1
            
        except Exception as e:
            print(f"Error: {window_id} - {e}")
            error_count += 1
        
        time.sleep(1.0)  # Rate limiting
    
    print(f"\nCompleted: {success_count} success, {error_count} errors")
    print(f"Results saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()