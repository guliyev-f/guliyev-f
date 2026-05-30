#!/usr/bin/env python3
"""
Trains decision trees on human-coded features, then uses GPT to extract the same
features from new diagrams and predicts UML/Format/Formality/Viewpoint labels.

Run order: load manual data -> train trees -> GPT extraction -> predict.
"""

import csv
import os
import sys
import base64
import json
import time

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from openai import OpenAI


MAX_DEPTH = 4
MIN_SAMPLES_LEAF = 3

GPT_MODEL = "gpt-5-mini"
OPENAI_API_KEY = "open ai key"

# 0 = all
MAX_ANALYZE = 0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_FOLDER = os.path.join(PROJECT_DIR, "data")
DIAGRAMS_FOLDER = os.path.join(PROJECT_DIR, "diagrams")

MANUAL_FEATURES_CSV = os.path.join(DATA_FOLDER, "manual_features.csv")
AI_FEATURES_CSV     = os.path.join(DATA_FOLDER, "features_from_ai.csv")
RESULT_CSV          = os.path.join(DATA_FOLDER, "result_decision_tree.csv")

DIAGRAM_SUBFOLDERS = [
    os.path.join(DIAGRAMS_FOLDER, "Lindholmen subset"),
    os.path.join(DIAGRAMS_FOLDER, "Mined dataset"),
]


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

BOOLEAN_FEATURES     = [c for c in FEATURE_COLUMNS if c != "interp_architectural_relevance"]
CATEGORICAL_FEATURES = ["interp_architectural_relevance"]

TARGET_COLUMNS = ["UML", "Format", "Formality", "Viewpoint"]
ID_COLUMNS     = ["File Name", "URL"]


FEATURE_EXTRACTION_PROMPT = """
You are extracting training features from a software diagram.

Use the JSON schema to return the required fields only.

Core rule:
For every boolean feature, answer true only when there is clear evidence in the diagram itself.
If the evidence is weak, ambiguous, inferred from the filename or only guessed from context, answer false.

Use visible diagram content first and what is explicitly represented.

Feature rules:

feat_has_classes:
True if the diagram contains class-like or entity-like boxes representing software/data structures. Strong evidence includes class names, compartments, attributes, methods, entities, models, objects, or domain classes. Do not mark true for generic process boxes, screens, services, or cloud nodes.

feat_has_attributes:
True only if class/entity boxes explicitly list fields, properties, columns, or attributes. Generic labels inside boxes do not count.

feat_has_methods:
True only if class/entity boxes explicitly list operations, methods, functions, or callable behavior, usually with parentheses or operation-like names.

feat_has_typed_relationships:
True only if relationships have explicit UML-style or semantically typed meaning, such as inheritance/generalization, realization, dependency, association, aggregation, composition, implements, extends, uses, owns, contains, or clearly labelled relationship types.
Do not mark true for plain arrows, unlabeled lines, simple data flow, or generic connections.

feat_has_directional_flow:
True if arrows clearly indicate direction from one element to another. This includes control flow, data flow, message flow, dependency direction, request/response, or process sequence.

feat_has_sequence_flow:
True if the diagram shows ordered communication/messages between participants, objects, services, or components. Strong evidence includes horizontal messages, request/response chains, numbered steps, or sequence-diagram style interaction order. Lifelines help, but are not required.

feat_has_lifelines:
True only if vertical lifelines or sequence-diagram participant timelines are visible.

feat_has_actors:
True if external users/systems are shown as stick figures, actor icons, labelled roles, or external participants interacting with the system.

feat_has_use_case_elements:
True if use-case ellipses, system boundary boxes, include/extend relations, or use-case style actor-to-use-case interactions are visible.

feat_has_states:
True if the diagram contains states/statuses/modes of an object or system, such as "Pending", "Active", "Closed", "Authenticated", etc., represented as state nodes.

feat_has_transitions:
True if arrows connect states and represent state changes, especially with triggers, events, guards, or conditions. Do not mark true for generic process arrows unless they clearly connect states.

feat_has_activities:
True if the diagram shows actions, tasks, activity nodes, workflow steps, or operational process steps.

feat_has_decision_nodes:
True if diamond-shaped decision/merge nodes are visible, or if the diagram explicitly shows branching conditions such as yes/no, true/false, if/else.

feat_has_components:
True if the diagram shows software components, modules, services, subsystems, packages, applications, APIs, controllers, repositories, databases, or architectural building blocks. Do not mark true for ordinary class boxes unless they are clearly used as architectural components.

feat_has_interfaces:
True if explicit interfaces are shown, such as lollipop/socket notation, ports, provided/required interfaces, API endpoints, interface labels, or named contracts between components.

feat_has_deployment_nodes:
True if the diagram shows physical or runtime infrastructure: servers, devices, containers, pods, VMs, cloud services, databases as deployed infrastructure, browsers, mobile devices, execution environments, or 3D deployment nodes.

feat_has_artifacts:
True if deployable files or produced artifacts are shown, such as JAR/WAR files, Docker images, executables, packages, config files, databases schemas, documents, binaries, or stored artifacts.

feat_has_informal_structure:
True if the diagram mainly uses ad-hoc boxes, arrows, icons, or labels without clear standard UML/BPMN/architecture notation. Also true when the diagram is understandable but notation is loose or custom.
False if the diagram mostly follows a recognizable formal notation.

Interpretation rules:

interp_describes_static_structure:
True if the diagram describes stable structure: classes, entities, modules, packages, components, layers, data models, dependencies, or code organization.

interp_describes_runtime_behavior:
True if the diagram describes what happens during execution: service calls, object interactions, events, requests, responses, runtime flows, state changes, or dynamic behavior.

interp_describes_interactions:
True if the diagram shows communication between elements, such as messages, API calls, dependencies, events, data exchange, actor-system interaction, or component-to-component communication.

interp_describes_deployment:
True if the diagram shows where software runs or is allocated: cloud, servers, containers, devices, networks, databases, infrastructure, environments, or deployment topology.

interp_describes_process_flow:
True if the diagram shows a workflow, business process, activity sequence, control flow, decision flow, or step-by-step procedure.

interp_architectural_relevance:
Use:
- high: clearly shows software architecture, major components, deployment, system structure, important runtime interactions, or architectural responsibilities.
- medium: partially architectural, but mixed with detailed design, local workflow, or limited subsystem scope.
- low: mostly implementation-level detail, isolated class/data design, small algorithm, or local logic.
- none: not software-related, not architectural, or not enough information.

Consistency rules:
- If attributes or methods are true, classes should usually also be true.
- If lifelines are true, sequence_flow is usually true only when messages or ordered interactions are also shown.
- If transitions are true, states should usually also be true.
- Directional arrows alone do not automatically mean sequence_flow, interactions, or architectural relevance.
- Generic boxes are not automatically classes, components, or deployment nodes. Choose the most specific meaning supported by the diagram.
- Prefer false over true when unsure.
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


def load_manual_features() -> pd.DataFrame:
    df = pd.read_csv(MANUAL_FEATURES_CSV, encoding="utf-8-sig")
    print(f"  Loaded {len(df)} rows from {os.path.basename(MANUAL_FEATURES_CSV)}")

    # normalize booleans to 0/1 for sklearn
    for col in BOOLEAN_FEATURES:
        if col in df.columns:
            df[col] = df[col].map(
                {"True": 1, "False": 0, True: 1, False: 0,
                 "true": 1, "false": 0, "1": 1, "0": 0}
            ).fillna(0).astype(int)

    return df


def build_preprocessor(feature_cols_present: list[str]) -> ColumnTransformer:
    bool_cols = [c for c in BOOLEAN_FEATURES if c in feature_cols_present]
    cat_cols  = [c for c in CATEGORICAL_FEATURES if c in feature_cols_present]

    transformers = []
    if bool_cols:
        transformers.append((
            "bool",
            SimpleImputer(strategy="constant", fill_value=False),
            bool_cols,
        ))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="none")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            cat_cols,
        ))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def train_all_models(df: pd.DataFrame) -> dict:
    """Train 4 targets x 2 balancing modes = 8 trees."""
    feature_cols_present = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[feature_cols_present]

    models = {}

    for target in TARGET_COLUMNS:
        y = df[target]

        mask = y.notna() & (y.astype(str).str.strip() != "")
        X_t = X[mask]
        y_t = y[mask]

        if len(y_t) == 0:
            print(f"  Target '{target}' has no valid labels - skipping")
            continue

        for balance_mode, class_weight in [("unbalanced", None), ("balanced", "balanced")]:
            preprocessor = build_preprocessor(feature_cols_present)
            pipe = Pipeline([
                ("preprocess", preprocessor),
                ("tree", DecisionTreeClassifier(
                    max_depth=MAX_DEPTH,
                    min_samples_leaf=MIN_SAMPLES_LEAF,
                    class_weight=class_weight,
                    random_state=42,
                )),
            ])
            pipe.fit(X_t, y_t)
            models[(target, balance_mode)] = pipe

            n_classes = len(y_t.unique())
            print(f"    {target} ({balance_mode}): {len(y_t)} samples, {n_classes} classes")

    return models


def extract_features_gpt(client: OpenAI, filepath: str, filename: str) -> dict | None:
    """Call GPT-5 mini to extract the 24 features from a diagram."""
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
                f"Please extract the diagnostic features from this text-based diagram source.\n\n"
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


def run_feature_extraction(df_manual: pd.DataFrame) -> pd.DataFrame:
    """Extract features for all diagrams. Resumes from existing CSV if present."""
    if os.path.isfile(AI_FEATURES_CSV):
        df_ai = pd.read_csv(AI_FEATURES_CSV, encoding="utf-8-sig")
        print(f"  Loaded {len(df_ai)} existing rows from {os.path.basename(AI_FEATURES_CSV)}")
    else:
        df_ai = pd.DataFrame(columns=ID_COLUMNS + FEATURE_COLUMNS)

    already_done  = set(df_ai["File Name"].astype(str).str.strip()) if len(df_ai) > 0 else set()
    all_filenames = df_manual["File Name"].astype(str).str.strip().tolist()
    all_urls      = df_manual["URL"].astype(str).str.strip().tolist()

    to_process = [(fn, url) for fn, url in zip(all_filenames, all_urls) if fn not in already_done]

    skipped = len(all_filenames) - len(to_process)
    print(f"  {skipped} already extracted, {len(to_process)} to process")

    if not to_process:
        print("  All diagrams already have AI-extracted features.")
        return df_ai

    if MAX_ANALYZE > 0 and len(to_process) > MAX_ANALYZE:
        print(f"  Limiting to {MAX_ANALYZE} diagram(s)")
        to_process = to_process[:MAX_ANALYZE]

    client = OpenAI(api_key=OPENAI_API_KEY)
    new_rows = []

    for seq, (filename, url) in enumerate(to_process, 1):
        filepath = find_diagram_file(filename)
        if not filepath:
            print(f"  [{seq}/{len(to_process)}] File not found: {filename} - skipping")
            continue

        print(f"  [{seq}/{len(to_process)}] {'Image' if is_image_file(filename) else 'Text'} {filename}")

        try:
            features = extract_features_gpt(client, filepath, filename)
            if features:
                row = {"File Name": filename, "URL": url}
                row.update(features)
                new_rows.append(row)
                print(f"    Extracted features")
            else:
                print(f"    Empty response - skipping")
        except Exception as e:
            print(f"    Error: {e}")

        time.sleep(1)

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df_ai  = pd.concat([df_ai, df_new], ignore_index=True)
        df_ai.to_csv(AI_FEATURES_CSV, index=False, encoding="utf-8-sig")
        print(f"  Saved {len(df_ai)} total rows to {os.path.basename(AI_FEATURES_CSV)}")
    else:
        print(f"  No new rows to add.")

    return df_ai


# build prediction column names upfront
PRED_COLUMNS = []
for t in TARGET_COLUMNS:
    PRED_COLUMNS.append(f"{t}_without_balancing")
    PRED_COLUMNS.append(f"{t}_with_balancing")


def run_predictions(models: dict, df_ai: pd.DataFrame) -> pd.DataFrame:
    """Run all 8 trees on the AI features. Skips cells that already have a prediction."""
    if os.path.isfile(RESULT_CSV):
        df_result = pd.read_csv(RESULT_CSV, encoding="utf-8-sig")
        print(f"  Loaded {len(df_result)} existing rows from {os.path.basename(RESULT_CSV)}")
    else:
        df_result = pd.DataFrame(columns=ID_COLUMNS + PRED_COLUMNS)

    for col in PRED_COLUMNS:
        if col not in df_result.columns:
            df_result[col] = ""

    existing_files = set(df_result["File Name"].astype(str).str.strip()) if len(df_result) > 0 else set()

    new_result_rows = []
    for _, row in df_ai.iterrows():
        fn = str(row["File Name"]).strip()
        if fn not in existing_files:
            new_row = {"File Name": fn, "URL": row.get("URL", "")}
            for col in PRED_COLUMNS:
                new_row[col] = ""
            new_result_rows.append(new_row)
            existing_files.add(fn)

    if new_result_rows:
        df_result = pd.concat([df_result, pd.DataFrame(new_result_rows)], ignore_index=True)

    feature_cols_present = [c for c in FEATURE_COLUMNS if c in df_ai.columns]

    # coerce booleans to int - the CSV sometimes has string "True"/"False"
    for col in BOOLEAN_FEATURES:
        if col in df_ai.columns:
            df_ai[col] = df_ai[col].map(
                {True: 1, False: 0, "True": 1, "False": 0,
                 "true": 1, "false": 0, "1": 1, "0": 0, 1: 1, 0: 0}
            ).fillna(0).astype(int)

    predictions_made = 0

    for target in TARGET_COLUMNS:
        for balance_mode, col_suffix in [("unbalanced", "without_balancing"), ("balanced", "with_balancing")]:
            pred_col  = f"{target}_{col_suffix}"
            model_key = (target, balance_mode)

            if model_key not in models:
                print(f"  No model for {target} ({balance_mode}) - skipping")
                continue

            pipe = models[model_key]

            for idx, result_row in df_result.iterrows():
                fn           = str(result_row["File Name"]).strip()
                existing_val = str(result_row.get(pred_col, "")).strip()

                if existing_val:
                    continue  # already predicted

                ai_match = df_ai[df_ai["File Name"].astype(str).str.strip() == fn]
                if ai_match.empty:
                    continue

                X_row = ai_match[feature_cols_present]
                try:
                    pred = pipe.predict(X_row)[0]
                    df_result.at[idx, pred_col] = pred
                    predictions_made += 1
                except Exception as e:
                    print(f"  Prediction failed for {fn} / {pred_col}: {e}")

    print(f"  {predictions_made} new predictions made")

    df_result.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")
    print(f"  Saved {len(df_result)} rows to {os.path.basename(RESULT_CSV)}")

    return df_result


def main():
    print("Decision Tree + AI Feature Extraction Pipeline\n")

    if not os.path.isfile(MANUAL_FEATURES_CSV):
        print(f"  CSV not found: {MANUAL_FEATURES_CSV}")
        sys.exit(1)

    print("Step 1: Load human-coded features")
    df_manual = load_manual_features()

    present = [c for c in FEATURE_COLUMNS if c in df_manual.columns]
    missing = [c for c in FEATURE_COLUMNS if c not in df_manual.columns]
    print(f"  {len(present)} feature columns present, {len(missing)} missing")
    if missing:
        print(f"  Missing: {missing}")

    print(f"\nStep 2: Train decision trees (max_depth={MAX_DEPTH}, min_samples_leaf={MIN_SAMPLES_LEAF})")
    models = train_all_models(df_manual)
    print(f"  {len(models)} models trained")

    print("\nStep 3: Extract features with GPT-5 mini")
    df_ai = run_feature_extraction(df_manual)

    print("\nStep 4: Decision tree predictions")
    if len(df_ai) == 0:
        print("  No AI features available - skipping predictions")
    else:
        run_predictions(models, df_ai)

    print("\nDone.")
    print(f"  AI features:  {AI_FEATURES_CSV}")
    print(f"  Predictions:  {RESULT_CSV}")


if __name__ == "__main__":
    main()
