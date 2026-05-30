#!/usr/bin/env python3
"""
GPT extracts 24 features from each diagram, then a deterministic rule-based
classifier assigns UML, Format, Formality, and Viewpoint labels.
Reuses existing AI features from features_from_ai.csv when available.
"""

import os
import sys
import base64
import json
import time

import pandas as pd
from openai import OpenAI


GPT_MODEL      = "gpt-5-mini"
OPENAI_API_KEY = "open ai key"

# 0 = all
MAX_ANALYZE = 0

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_FOLDER     = os.path.join(PROJECT_DIR, "data")
DIAGRAMS_FOLDER = os.path.join(PROJECT_DIR, "diagrams")

MANUAL_FEATURES_CSV      = os.path.join(DATA_FOLDER, "manual_features.csv")
AI_FEATURES_CSV          = os.path.join(DATA_FOLDER, "features_from_ai.csv")
RULES_CLASSIFICATION_CSV = os.path.join(DATA_FOLDER, "rules_based_classification.csv")

DIAGRAM_SUBFOLDERS = [
    os.path.join(DIAGRAMS_FOLDER, "Lindholmen subset"),
    os.path.join(DIAGRAMS_FOLDER, "Mined dataset"),
]


ID_COLUMNS = ["File Name", "URL"]

FEATURE_COLUMNS = [
    "feat_has_classes",
    "feat_has_attributes",
    "feat_has_methods",
    "feat_has_typed_relationships",
    "feat_has_directional_flow",
    "feat_has_sequence_flow",
    "feat_has_lifelines",
    "feat_has_actors",
    "feat_has_use_case_elements",
    "feat_has_states",
    "feat_has_transitions",
    "feat_has_activities",
    "feat_has_decision_nodes",
    "feat_has_components",
    "feat_has_interfaces",
    "feat_has_deployment_nodes",
    "feat_has_artifacts",
    "feat_has_informal_structure",
    "interp_describes_static_structure",
    "interp_describes_runtime_behavior",
    "interp_describes_interactions",
    "interp_describes_deployment",
    "interp_describes_process_flow",
    "interp_architectural_relevance",
]

BOOLEAN_FEATURES = [c for c in FEATURE_COLUMNS if c != "interp_architectural_relevance"]

CLASSIFICATION_COLUMNS = ["UML", "Format", "Formality", "Viewpoint"]


FEATURE_EXTRACTION_PROMPT = """
PASTE MY FEATURE EXTRACTION PROMPT HERE.
"""


def _bool_field():
    return {"type": "boolean"}


FEATURE_EXTRACTION_SCHEMA = {
    "type": "json_schema",
    "name": "diagram_feature_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "feat_has_classes":                  _bool_field(),
            "feat_has_attributes":               _bool_field(),
            "feat_has_methods":                  _bool_field(),
            "feat_has_typed_relationships":      _bool_field(),
            "feat_has_directional_flow":         _bool_field(),
            "feat_has_sequence_flow":            _bool_field(),
            "feat_has_lifelines":                _bool_field(),
            "feat_has_actors":                   _bool_field(),
            "feat_has_use_case_elements":        _bool_field(),
            "feat_has_states":                   _bool_field(),
            "feat_has_transitions":              _bool_field(),
            "feat_has_activities":               _bool_field(),
            "feat_has_decision_nodes":           _bool_field(),
            "feat_has_components":               _bool_field(),
            "feat_has_interfaces":               _bool_field(),
            "feat_has_deployment_nodes":         _bool_field(),
            "feat_has_artifacts":                _bool_field(),
            "feat_has_informal_structure":       _bool_field(),
            "interp_describes_static_structure": _bool_field(),
            "interp_describes_runtime_behavior": _bool_field(),
            "interp_describes_interactions":     _bool_field(),
            "interp_describes_deployment":       _bool_field(),
            "interp_describes_process_flow":     _bool_field(),
            "interp_architectural_relevance": {
                "type": "string",
                "enum": ["none", "low", "medium", "high"],
            },
        },
        "required": [
            "feat_has_classes", "feat_has_attributes", "feat_has_methods",
            "feat_has_typed_relationships", "feat_has_directional_flow",
            "feat_has_sequence_flow", "feat_has_lifelines", "feat_has_actors",
            "feat_has_use_case_elements", "feat_has_states", "feat_has_transitions",
            "feat_has_activities", "feat_has_decision_nodes", "feat_has_components",
            "feat_has_interfaces", "feat_has_deployment_nodes", "feat_has_artifacts",
            "feat_has_informal_structure", "interp_describes_static_structure",
            "interp_describes_runtime_behavior", "interp_describes_interactions",
            "interp_describes_deployment", "interp_describes_process_flow",
            "interp_architectural_relevance",
        ],
        "additionalProperties": False,
    },
}


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}

TEXT_EXTENSIONS = {
    ".puml", ".plantuml", ".wsd",
    ".mmd", ".mermaid",
    ".drawio", ".bpmn",
    ".dot", ".gv",
    ".c4", ".dsl",
    ".uml", ".xmi",
    ".archimate",
}


def is_image_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in IMAGE_EXTENSIONS

def get_mime_type(filename: str) -> str:
    ext = os.path.splitext(filename.lower())[1]
    mime_map = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".svg": "image/svg+xml", ".webp": "image/webp",
    }
    return mime_map.get(ext, "image/png")

def find_diagram_file(filename: str) -> str | None:
    for folder in DIAGRAM_SUBFOLDERS:
        path = os.path.join(folder, filename)
        if os.path.isfile(path):
            return path
    return None


def to_bool(value) -> bool:
    """Best-effort bool conversion - handles strings, ints, actual bools."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value) and not pd.isna(value)
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return False


def load_csv_safe(filepath: str, columns: list[str]) -> pd.DataFrame:
    """Load CSV or return an empty DataFrame with the expected columns."""
    if os.path.isfile(filepath):
        df = pd.read_csv(filepath, encoding="utf-8-sig")
        for col in columns:
            if col not in df.columns:
                df[col] = ""
        return df
    return pd.DataFrame(columns=columns)

def get_key(row) -> str:
    """URL is preferred as the unique key; fall back to File Name."""
    url = str(row.get("URL", "")).strip()
    if url and url.lower() not in ("", "nan", "none"):
        return url
    return str(row.get("File Name", "")).strip()

def build_key_set(df: pd.DataFrame) -> set:
    keys = set()
    for _, row in df.iterrows():
        keys.add(get_key(row))
    return keys


def extract_features_gpt(client: OpenAI, filepath: str, filename: str) -> dict | None:
    """Call GPT-5-mini to extract the 24 features from a diagram."""
    user_content = []

    if is_image_file(filename):
        with open(filepath, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        mime = get_mime_type(filename)
        user_content.append({
            "type": "input_text",
            "text": "Please extract the diagnostic features from this diagram.",
        })
        user_content.append({
            "type": "input_image",
            "image_url": f"data:{mime};base64,{image_data}",
            "detail": "high",
        })
    else:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            text_content = f.read()
        user_content.append({
            "type": "input_text",
            "text": (
                "Please extract the diagnostic features from this text-based "
                "diagram source.\n\n"
                f"```\n{text_content}\n```"
            ),
        })

    response = client.responses.create(
        model=GPT_MODEL,
        instructions=FEATURE_EXTRACTION_PROMPT,
        input=[{"role": "user", "content": user_content}],
        text={"format": FEATURE_EXTRACTION_SCHEMA},
    )
    return json.loads(response.output_text)


# UML-specific features used in scoring (excludes generic flow signals)
_UML_SPECIFIC_FEATURES = [
    "feat_has_classes", "feat_has_attributes", "feat_has_methods",
    "feat_has_typed_relationships", "feat_has_sequence_flow",
    "feat_has_lifelines", "feat_has_actors", "feat_has_use_case_elements",
    "feat_has_states", "feat_has_transitions", "feat_has_activities",
    "feat_has_decision_nodes", "feat_has_components", "feat_has_interfaces",
    "feat_has_deployment_nodes", "feat_has_artifacts",
]


def classify_diagram(row: dict) -> dict:
    """Deterministic rule-based classification from AI-extracted features."""

    b = {col: to_bool(row.get(col)) for col in BOOLEAN_FEATURES}
    architectural_relevance = str(row.get("interp_architectural_relevance", "none")).strip().lower()
    if architectural_relevance not in ("none", "low", "medium", "high"):
        architectural_relevance = "none"

    # 1. UML detection
    uml_score = 0

    if b["feat_has_classes"]:              uml_score += 2
    if b["feat_has_attributes"]:           uml_score += 1
    if b["feat_has_methods"]:              uml_score += 1
    if b["feat_has_typed_relationships"]:  uml_score += 2
    if b["feat_has_sequence_flow"]:        uml_score += 2
    if b["feat_has_lifelines"]:            uml_score += 2
    if b["feat_has_actors"]:               uml_score += 1
    if b["feat_has_use_case_elements"]:    uml_score += 2
    if b["feat_has_states"]:               uml_score += 2
    if b["feat_has_transitions"]:          uml_score += 1
    if b["feat_has_activities"]:           uml_score += 2
    if b["feat_has_decision_nodes"]:       uml_score += 1
    if b["feat_has_components"]:           uml_score += 1
    if b["feat_has_interfaces"]:           uml_score += 1
    if b["feat_has_deployment_nodes"]:     uml_score += 1
    if b["feat_has_artifacts"]:            uml_score += 1

    if b["feat_has_informal_structure"]:   uml_score -= 2

    has_any_uml_specific   = any(b[f] for f in _UML_SPECIFIC_FEATURES)
    generic_box_arrow_signal = b["feat_has_directional_flow"] and not has_any_uml_specific
    if generic_box_arrow_signal:
        uml_score -= 3

    if uml_score >= 2:
        uml = "Yes"
    elif uml_score <= -1:
        uml = "No"
    else:
        uml = "Uncertain"

    # 2. UML type / Format
    if uml != "Yes":
        fmt = "Not UML"
    elif b["feat_has_lifelines"] or b["feat_has_sequence_flow"]:
        fmt = "Sequence Diagram"
    elif b["feat_has_actors"] or b["feat_has_use_case_elements"]:
        fmt = "Use Case Diagram"
    elif b["feat_has_states"] and b["feat_has_transitions"]:
        fmt = "State Machine Diagram"
    elif b["feat_has_activities"] or b["feat_has_decision_nodes"]:
        fmt = "Activity Diagram"
    elif b["feat_has_deployment_nodes"] or b["feat_has_artifacts"]:
        fmt = "Deployment Diagram"
    elif b["feat_has_components"] or b["feat_has_interfaces"]:
        fmt = "Component Diagram"
    elif b["feat_has_classes"]:
        fmt = "Class Diagram"
    else:
        fmt = "Other UML"

    # 3. Formality
    if b["feat_has_informal_structure"]:
        formality = "Informal"
    elif uml == "Yes":
        formality = "Formal"
    elif b["feat_has_typed_relationships"]:
        formality = "Formal"
    else:
        formality = "Informal"

    # 4. Architectural viewpoint
    architecture_score = 0

    if b["interp_describes_deployment"]:       architecture_score += 3
    if b["interp_describes_runtime_behavior"]: architecture_score += 2
    if b["interp_describes_interactions"]:     architecture_score += 2
    if b["interp_describes_static_structure"]: architecture_score += 1
    if b["interp_describes_process_flow"]:     architecture_score += 1

    relevance_map = {"high": 3, "medium": 2, "low": 1, "none": 0}
    architecture_score += relevance_map.get(architectural_relevance, 0)

    if b["feat_has_deployment_nodes"] or b["feat_has_artifacts"]:
        architecture_score += 2
    if b["feat_has_components"] or b["feat_has_interfaces"]:
        architecture_score += 1
    if b["feat_has_classes"] and b["interp_describes_static_structure"]:
        architecture_score += 1

    if generic_box_arrow_signal:
        architecture_score -= 2
    if b["feat_has_informal_structure"]:
        architecture_score -= 1

    architecture_score = max(architecture_score, 0)

    if architecture_score < 4:
        viewpoint = "Not_Architecture"
    elif b["interp_describes_deployment"] or b["feat_has_deployment_nodes"] or b["feat_has_artifacts"]:
        viewpoint = "Allocation"
    elif (b["interp_describes_runtime_behavior"]
          or b["interp_describes_interactions"]
          or b["feat_has_sequence_flow"]
          or b["feat_has_lifelines"]):
        viewpoint = "Component-and-Connector"
    elif b["interp_describes_process_flow"] and not b["interp_describes_static_structure"]:
        viewpoint = "Component-and-Connector"
    elif b["interp_describes_static_structure"]:
        viewpoint = "Module"
    elif b["feat_has_components"] or b["feat_has_interfaces"]:
        viewpoint = "Module"
    else:
        viewpoint = "Not_Architecture"

    return {"UML": uml, "Format": fmt, "Formality": formality, "Viewpoint": viewpoint}


def main():
    print("Rule-Based AI Classification Pipeline\n")

    if not os.path.isfile(MANUAL_FEATURES_CSV):
        print(f"  Input CSV not found: {MANUAL_FEATURES_CSV}")
        sys.exit(1)

    df_input = pd.read_csv(MANUAL_FEATURES_CSV, encoding="utf-8-sig")[["File Name", "URL"]]
    print(f"  Loaded {len(df_input)} diagrams from {os.path.basename(MANUAL_FEATURES_CSV)}")

    df_ai         = load_csv_safe(AI_FEATURES_CSV, ID_COLUMNS + FEATURE_COLUMNS)
    df_classified = load_csv_safe(RULES_CLASSIFICATION_CSV, ID_COLUMNS + CLASSIFICATION_COLUMNS)

    classified_keys = build_key_set(df_classified)
    ai_keys         = build_key_set(df_ai)

    diagrams_to_process = []
    for _, row in df_input.iterrows():
        fn  = str(row["File Name"]).strip()
        url = str(row["URL"]).strip()
        key = url if url and url.lower() not in ("", "nan", "none") else fn
        if key not in classified_keys:
            diagrams_to_process.append((fn, url, key))

    count_skipped          = len(df_input) - len(diagrams_to_process)
    count_reused           = 0
    count_new_gpt          = 0
    count_new_classifications = 0

    print(f"  {count_skipped} already classified, {len(diagrams_to_process)} to process\n")

    if not diagrams_to_process:
        print("  All diagrams already classified. Nothing to do.")
        _print_summary(count_skipped, count_reused, count_new_gpt, count_new_classifications)
        return

    if MAX_ANALYZE > 0 and len(diagrams_to_process) > MAX_ANALYZE:
        print(f"  Limiting to {MAX_ANALYZE} diagram(s)")
        diagrams_to_process = diagrams_to_process[:MAX_ANALYZE]

    needs_gpt = any(key not in ai_keys for _, _, key in diagrams_to_process)

    client = None
    if needs_gpt:
        if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_KEY_HERE":
            print("  OPENAI_API_KEY is not set.")
            sys.exit(1)
        client = OpenAI(api_key=OPENAI_API_KEY)

    new_ai_rows              = []
    new_classification_rows  = []

    total = len(diagrams_to_process)
    for seq, (filename, url, key) in enumerate(diagrams_to_process, 1):
        print(f"  [{seq}/{total}] {filename}")

        if key in ai_keys:
            match = df_ai[df_ai.apply(lambda r: get_key(r) == key, axis=1)]
            if match.empty:
                print(f"    Key found in set but no matching row - skipping")
                continue
            feature_row = match.iloc[0].to_dict()
            count_reused += 1
            print(f"    Reusing existing AI features")
        else:
            filepath = find_diagram_file(filename)
            if not filepath:
                print(f"    File not found: {filename} - skipping")
                continue

            print(f"    {'Image' if is_image_file(filename) else 'Text'} - sending to GPT-5-mini...")

            try:
                features = extract_features_gpt(client, filepath, filename)
                if not features:
                    print(f"    Empty GPT response - skipping")
                    continue
            except Exception as e:
                print(f"    GPT error: {e}")
                continue

            feature_row = {"File Name": filename, "URL": url}
            feature_row.update(features)

            new_ai_rows.append(feature_row)
            ai_keys.add(key)
            count_new_gpt += 1
            print(f"    Features extracted")

            time.sleep(1)  # rate-limit courtesy

        classification = classify_diagram(feature_row)
        classification["File Name"] = filename
        classification["URL"]       = url
        new_classification_rows.append(classification)
        classified_keys.add(key)
        count_new_classifications += 1

    if new_ai_rows:
        df_new_ai   = pd.DataFrame(new_ai_rows)
        df_ai       = pd.concat([df_ai, df_new_ai], ignore_index=True)
        ordered_cols = ID_COLUMNS + FEATURE_COLUMNS
        df_ai       = df_ai[[c for c in ordered_cols if c in df_ai.columns]]
        df_ai.to_csv(AI_FEATURES_CSV, index=False, encoding="utf-8-sig")
        print(f"\n  Saved {len(df_ai)} total rows to {os.path.basename(AI_FEATURES_CSV)}")

    if new_classification_rows:
        df_new_class  = pd.DataFrame(new_classification_rows)
        df_classified = pd.concat([df_classified, df_new_class], ignore_index=True)
        ordered_cols  = ID_COLUMNS + CLASSIFICATION_COLUMNS
        df_classified = df_classified[[c for c in ordered_cols if c in df_classified.columns]]
        df_classified.to_csv(RULES_CLASSIFICATION_CSV, index=False, encoding="utf-8-sig")
        print(f"  Saved {len(df_classified)} total rows to {os.path.basename(RULES_CLASSIFICATION_CSV)}")

    _print_summary(count_skipped, count_reused, count_new_gpt, count_new_classifications)


def _print_summary(skipped: int, reused: int, new_gpt: int, new_classifications: int):
    print("\nSummary:")
    print(f"  Skipped (already classified):   {skipped}")
    print(f"  AI features reused:             {reused}")
    print(f"  Newly analyzed by GPT:          {new_gpt}")
    print(f"  New classifications:            {new_classifications}")
    print(f"\n  AI features:      {AI_FEATURES_CSV}")
    print(f"  Classifications:  {RULES_CLASSIFICATION_CSV}\n")


if __name__ == "__main__":
    main()
