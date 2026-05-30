#!/usr/bin/env python3
"""
Runs GPT, Gemini, and Claude on each diagram in manual_features.csv
and writes the classifications to llm_ground_truth.csv.
Skips rows already analyzed so it's safe to rerun.
"""

import csv
import os
import sys
import base64
import time
import json
from openai import OpenAI
from google import genai
from anthropic import Anthropic
from google.genai import types


# -- model config --

GPT_MODEL    = "gpt-5-mini"
GEMINI_MODEL = "gemini-2.5-flash"
CLAUDE_MODEL = "claude-haiku-4-5"

MODELS = [
    {"provider": "openai",  "model": GPT_MODEL},
    {"provider": "gemini",  "model": GEMINI_MODEL},
    {"provider": "claude",  "model": CLAUDE_MODEL},
]

# 0 = all
MAX_ANALYZE = 0

OPENAI_API_KEY  = "open ai key"
GEMINI_API_KEY  = "gemini key"
CLAUDE_API_KEY  = "claude key"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_FOLDER = os.path.join(PROJECT_DIR, "data")
DIAGRAMS_FOLDER = os.path.join(PROJECT_DIR, "diagrams")

CSV_PATH = os.path.join(DATA_FOLDER, "llm_ground_truth.csv")
MANUAL_FEATURES_CSV = os.path.join(DATA_FOLDER, "manual_features.csv")

DIAGRAM_SUBFOLDERS = [
    os.path.join(DIAGRAMS_FOLDER, "Lindholmen subset"),
    os.path.join(DIAGRAMS_FOLDER, "Mined dataset"),
]

# Set True to test without calling APIs
DRY_RUN = False


ANALYSIS_PROMPT = """\
You are a strict software-architecture diagram classifier.

Analyze the provided diagram, given either as an image or textual diagram source. Classify it into four fields:
1. UML
2. UML_Type
3. Formality
4. Architectural_Viewpoint

Rules:
- Base decisions only on what is visible or explicitly defined in the diagram.
- For text-based sources such as PlantUML, Mermaid, Graphviz, XMI, BPMN, SVG, or Draw.io XML, classify the diagram the source defines, not the raw file syntax.
- Do not infer from file name, repository name, or surrounding project context.
- UML detection, formality, and architectural viewpoint are independent.
- A diagram may be UML and Formal but still Not_Architecture, for example a UML use case diagram.
- File format alone does not determine formality.
- Apply the disambiguation rules before using judgement.
- Return JSON only.

Step 1 - UML detection

Set "UML" to "Yes" if the diagram uses recognizable OMG UML notation or semantics, such as:
- classes with attributes/methods, associations, inheritance, realization, composition, multiplicities
- sequence lifelines, activation bars, ordered messages
- activity nodes, decisions, forks/joins, swimlanes
- actors with use-case ellipses
- states and transitions
- UML components, interfaces, ports, deployment nodes, artifacts, stereotypes

Set "UML" to "No" if it uses only generic boxes/arrows, informal labels, BPMN, ArchiMate, ER/database notation, C4-style boxes, Mermaid/Graphviz flowcharts without UML semantics, or pure text.

Step 2 - UML type

If UML = "No", set "UML_Type" to "None".

If UML = "Yes", choose exactly one dominant type:
- Class Diagram: classes, attributes, methods, associations, inheritance, composition
- Sequence Diagram: lifelines, activation bars, ordered runtime messages
- Activity Diagram: activity/control flows, decisions, forks/joins, swimlanes
- Use Case Diagram: actors, use-case ellipses, system boundary, include/extend
- Component Diagram: UML components, interfaces, ports, connectors
- Deployment Diagram: UML nodes, devices, execution environments, artifacts
- State Machine Diagram: states, transitions, guards, entry/exit actions
- Package Diagram: packages, dependencies, containment
- Object Diagram: object instances, slots, links
- Other UML: valid UML but not one of the above

Step 3 - Formality

Classify notation formality, not amount of detail.

Formal:
- Consistently follows a standardized or clearly defined modeling notation such as UML, ArchiMate, SysML, BPMN, or well-formed C4 notation.
- Minor omissions are acceptable if the notation remains recognizable and consistent.

Informal:
- Uses ad-hoc boxes/arrows, free-form labels, mixed notation, inconsistent symbols, or only partial UML-like elements.
- PlantUML, Mermaid, Graphviz, Draw.io, SVG, or XML syntax is not automatically Formal. The actual diagram notation must be standardized or clearly structured.

Step 4 - Architectural viewpoint

Choose exactly one:

Module:
Static compile-time code structure. Elements are source-code units such as classes, packages, interfaces, modules, libraries, or namespaces. Relationships are code-level relations such as inheritance, realization, import, package dependency, composition, class association, or compile-time usage.

Component-and-Connector:
Runtime or logical interaction structure. Elements are services, APIs, applications, databases, queues, processes, layers, or runtime components. Arrows represent calls, messages, data flow, protocol exchange, control flow, or runtime communication. Sequence diagrams with runtime messages belong here. Logical layered architectures without physical hosts belong here.

Allocation:
Mapping of software onto physical or execution environments. Main elements are servers, VMs, containers, Docker/Kubernetes nodes, pods, cloud regions, clusters, devices, CI/CD stages, or deployment environments that host or run software.

Hybrid:
Use only when two or more architecture viewpoints are clearly and substantially present with no single dominant viewpoint. Examples: class/module detail inside deployment nodes; runtime service topology combined with physical infrastructure at similar importance.

Not_Architecture:
Does not depict software or infrastructure structure. Includes use-case diagrams, BPMN business workflows, ER/database schemas focused on data modeling, Gantt charts, UI wireframes/mockups, screenshots, pure business processes, or unreadable/insufficient diagrams.

Disambiguation rules:
Apply the first matching rule where possible.

1. If the diagram is unreadable, broken, empty, or insufficient -> Not_Architecture.
2. Use-case diagram with actors and use-case ellipses -> Not_Architecture.
3. BPMN/business process or activity workflow without software structure -> Not_Architecture.
4. ER/database schema with tables, PK/FK, entities, attributes, or relationships only -> Not_Architecture.
5. Docker/Kubernetes/cloud/server nodes containing internal class/module/code detail -> Hybrid.
6. Deployment/infrastructure nodes are primary and contain only services/apps/databases -> Allocation.
7. Physical hosts are absent and the diagram shows services/apps/layers/databases exchanging calls, data, messages, or events -> Component-and-Connector.
8. Sequence diagram with runtime messages -> Component-and-Connector.
9. Elements are classes/packages/interfaces/modules with attributes, methods, inheritance, realization, imports, or code dependencies -> Module.
10. "uses" or "depends on" means Module only when used between code units such as classes, packages, interfaces, modules, or libraries. Between services, apps, APIs, databases, queues, or systems, treat it as Component-and-Connector.
11. If one viewpoint clearly dominates, choose it. Use Hybrid only when multiple viewpoints are substantial and balanced.

Decision order:
1. Not_Architecture rules
2. Explicit Hybrid cases
3. Allocation
4. Component-and-Connector
5. Module
6. Hybrid only if no dominant viewpoint exists

Output:
Return only the structured JSON object matching the provided schema.

Keep Visual_Reasoning concise, maximum 3 sentences. It must mention visible evidence and the rule applied.
"""


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}

TEXT_EXTENSIONS = {
    ".puml", ".plantuml", ".wsd",
    ".mmd", ".mermaid",
    ".drawio",
    ".bpmn",
    ".dot", ".gv",
    ".c4", ".dsl",
    ".uml", ".xmi",
    ".archimate",
}

ALL_EXTENSIONS = IMAGE_EXTENSIONS | TEXT_EXTENSIONS


def get_file_extension(filename: str) -> str:
    _, ext = os.path.splitext(filename.lower())
    return ext

def is_image_file(filename: str) -> bool:
    return get_file_extension(filename) in IMAGE_EXTENSIONS

def is_supported_file(filename: str) -> bool:
    return get_file_extension(filename) in ALL_EXTENSIONS

def get_mime_type(filename: str) -> str:
    ext = get_file_extension(filename)
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
        ".webp": "image/webp",
    }
    return mime_map.get(ext, "image/png")


CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "Visual_Reasoning": {"type": "string"},
        "UML": {
            "type": "string",
            "enum": ["Yes", "No"]
        },
        "UML_Type": {
            "type": "string",
            "enum": [
                "Class Diagram", "Sequence Diagram", "Activity Diagram",
                "Use Case Diagram", "Component Diagram", "Deployment Diagram",
                "State Machine Diagram", "Package Diagram", "Object Diagram",
                "Other UML", "None"
            ],
        },
        "Formality": {
            "type": "string",
            "enum": ["Formal", "Informal"],
        },
        "Architectural_Viewpoint": {
            "type": "string",
            "enum": ["Module", "Component-and-Connector", "Allocation", "Hybrid", "Not_Architecture"],
        },
    },
    "required": ["Visual_Reasoning", "UML", "UML_Type", "Formality", "Architectural_Viewpoint"],
    "additionalProperties": False,
}

OPENAI_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "name": "ground_truth_classification",
    "strict": True,
    "schema": CLASSIFICATION_SCHEMA,
}

# Gemini doesn't support additionalProperties, so strip it
GEMINI_RESPONSE_SCHEMA = {k: v for k, v in CLASSIFICATION_SCHEMA.items() if k != "additionalProperties"}


def _init_openai():
    return OpenAI(api_key=OPENAI_API_KEY)


def analyze_openai(client, model: str, image_path: str | None, text_content: str | None) -> dict:
    user_content = []

    if image_path:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        mime = get_mime_type(image_path)
        user_content.append({
            "type": "input_text",
            "text": "Please analyze this diagram. Use the image as the primary evidence.",
        })
        user_content.append({
            "type": "input_image",
            "image_url": f"data:{mime};base64,{image_data}",
            "detail": "high",
        })
    else:
        user_content.append({
            "type": "input_text",
            "text": (
                f"Please analyze this text-based diagram source.\n"
                f"Use the diagram source as the primary evidence.\n\n"
                f"```\n{text_content}\n```"
            ),
        })

    response = client.responses.create(
        model=model,
        instructions=ANALYSIS_PROMPT,
        input=[{"role": "user", "content": user_content}],
        text={"format": OPENAI_RESPONSE_SCHEMA},
    )
    return json.loads(response.output_text)


def _init_gemini():
    return genai.Client(api_key=GEMINI_API_KEY)


def analyze_gemini(client, model: str, image_path: str | None, text_content: str | None) -> dict:
    contents = []

    if image_path:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        mime = get_mime_type(image_path)
        contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime))
        contents.append("Please analyze this diagram. Use the image as the primary evidence.")
    else:
        contents.append(
            f"Please analyze this text-based diagram source.\n"
            f"Use the diagram source as the primary evidence.\n\n"
            f"```\n{text_content}\n```"
        )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config={
            "system_instruction": ANALYSIS_PROMPT,
            "response_mime_type": "application/json",
            "response_schema": GEMINI_RESPONSE_SCHEMA,
        },
    )
    return json.loads(response.text)


def _init_claude():
    return Anthropic(api_key=CLAUDE_API_KEY)


def analyze_claude(client, model: str, image_path: str | None, text_content: str | None) -> dict:
    user_content = []

    if image_path:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        mime = get_mime_type(image_path)
        user_content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime,
                "data": image_data,
            },
        })
        user_content.append({
            "type": "text",
            "text": "Please analyze this diagram. Use the image as the primary evidence.",
        })
    else:
        user_content.append({
            "type": "text",
            "text": (
                f"Please analyze this text-based diagram source.\n"
                f"Use the diagram source as the primary evidence.\n\n"
                f"```\n{text_content}\n```"
            ),
        })

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=ANALYSIS_PROMPT,
        messages=[{"role": "user", "content": user_content}],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": CLASSIFICATION_SCHEMA,
            }
        },
    )
    return json.loads(response.content[0].text)


def get_analyzable_rows(rows):
    """Return (row_index, filepath, filename, dataset) for rows with a valid file on disk.
    Per-model skip logic is applied later."""
    analyzable = []
    no_file = 0

    for i, row in enumerate(rows):
        filename = str(row.get("File Name", "")).strip()
        if not filename:
            no_file += 1
            continue

        filepath = None
        for folder in DIAGRAM_SUBFOLDERS:
            candidate = os.path.join(folder, filename)
            if os.path.isfile(candidate):
                filepath = candidate
                break

        if not filepath:
            no_file += 1
            continue

        if not is_supported_file(filename):
            continue

        analyzable.append((i, filepath, filename, ""))

    return analyzable, no_file


def run_model(model_cfg: dict, rows: list, analyzable: list, clients: dict):
    """Run a single model against rows not yet analyzed. Updates rows in-place."""
    provider = model_cfg["provider"]
    model_id = model_cfg["model"]
    client = clients[provider]

    col_uml       = f"Final_UML_{model_id}"
    col_format    = f"Final_Format_{model_id}"
    col_formality = f"Final_Formality_{model_id}"
    col_viewpoint = f"Final_Viewpoint_{model_id}"
    col_reasoning = f"Visual_Reasoning_{model_id}"

    to_analyze = []
    already_done = 0
    for entry in analyzable:
        row_idx = entry[0]
        if rows[row_idx].get(col_uml, "").strip():
            already_done += 1
        else:
            to_analyze.append(entry)

    print(f"\n  Model {model_id} ({provider}):")
    print(f"     {already_done} already analyzed, {len(to_analyze)} to do")

    if not to_analyze:
        print(f"  Nothing to do for {model_id}.")
        return 0

    if MAX_ANALYZE > 0 and len(to_analyze) > MAX_ANALYZE:
        print(f"  Limiting to {MAX_ANALYZE} diagram(s)")
        to_analyze = to_analyze[:MAX_ANALYZE]

    analyzed_count = 0

    for seq, (row_idx, filepath, filename, dataset) in enumerate(to_analyze, 1):
        image = is_image_file(filename)
        diagram_id = rows[row_idx].get("Diagram_ID", "?")

        print(f"  [{seq}/{len(to_analyze)}] ID={diagram_id} | {'Image' if image else 'Text '} | {dataset}")
        print(f"    File: {filename}")

        if DRY_RUN:
            result = {
                "UML": "No",
                "UML_Type": "None",
                "Formality": "Informal",
                "Architectural_Viewpoint": "Not_Architecture",
                "Visual_Reasoning": f"Dry run - no API call made for {filename}",
            }
            print(f"    DRY RUN - skipped API call")
        else:
            try:
                img_path = filepath if image else None
                txt_content = None
                if not image:
                    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                        txt_content = f.read()

                if provider == "openai":
                    result = analyze_openai(client, model_id, img_path, txt_content)
                elif provider == "gemini":
                    result = analyze_gemini(client, model_id, img_path, txt_content)
                elif provider == "claude":
                    result = analyze_claude(client, model_id, img_path, txt_content)
                else:
                    raise ValueError(f"Unknown provider: {provider}")

                if result:
                    print(f"    UML: {result.get('UML', '?')} | "
                          f"Type: {result.get('UML_Type', '?')} | "
                          f"Formality: {result.get('Formality', '?')} | "
                          f"Viewpoint: {result.get('Architectural_Viewpoint', '?')}")
                else:
                    result = {
                        "UML": "No",
                        "UML_Type": "None",
                        "Formality": "ERROR",
                        "Architectural_Viewpoint": "ERROR",
                        "Visual_Reasoning": "API returned an empty response.",
                    }
                    print(f"    Empty response")

            except Exception as e:
                result = {
                    "UML": "No",
                    "UML_Type": "None",
                    "Formality": "ERROR",
                    "Architectural_Viewpoint": "ERROR",
                    "Visual_Reasoning": f"Error: {e}",
                }
                print(f"    Error: {e}")

        uml_val = result.get("UML", "No")
        uml_type = result.get("UML_Type", "None")

        rows[row_idx][col_uml]       = uml_val
        rows[row_idx][col_format]    = uml_type if uml_val == "Yes" else "None"
        rows[row_idx][col_formality] = result.get("Formality", "")
        rows[row_idx][col_viewpoint] = result.get("Architectural_Viewpoint", "")
        rows[row_idx][col_reasoning] = result.get("Visual_Reasoning", "")

        analyzed_count += 1

        if not DRY_RUN:
            time.sleep(1)

    return analyzed_count


def save_csv(rows, fieldnames):
    """Write updated rows back to the CSV. Only saves rows where at least one model has a result."""
    llm_cols = [c for c in fieldnames if c.startswith("Final_UML_")]

    rows_to_save = [row for row in rows if any(row.get(col, "").strip() for col in llm_cols)]

    with open(CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=",", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_to_save)
    print(f"  CSV saved: {CSV_PATH} ({len(rows_to_save)} rows)")


def main():
    print(f"  Models: {', '.join(m['model'] for m in MODELS)}")

    if not os.path.isfile(MANUAL_FEATURES_CSV):
        print(f"CSV not found: {MANUAL_FEATURES_CSV}")
        sys.exit(1)

    with open(MANUAL_FEATURES_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        manual_rows = list(reader)

    print(f"  Loaded {len(manual_rows)} target diagrams from manual_features.csv")

    rows = [{"File Name": r.get("File Name", ""), "URL": r.get("URL", "")} for r in manual_rows]
    fieldnames = ["File Name", "URL"]

    for model_cfg in MODELS:
        m_id = model_cfg["model"]
        fieldnames.extend([
            f"Final_UML_{m_id}", f"Final_Format_{m_id}", f"Final_Formality_{m_id}",
            f"Final_Viewpoint_{m_id}", f"Visual_Reasoning_{m_id}"
        ])

    # load existing results if the file exists, so we can resume
    if os.path.isfile(CSV_PATH):
        try:
            with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
                content = f.read()
            if content.strip():
                # some older exports used semicolons - handle both
                delim = ";" if ";" in content.splitlines()[0] else ","
                reader = csv.DictReader(content.splitlines(), delimiter=delim)
                existing_rows = list(reader)

                existing_by_name = {r.get("File Name", r.get("File name", "")): r for r in existing_rows}
                for row in rows:
                    fname = row["File Name"]
                    if fname in existing_by_name:
                        row.update(existing_by_name[fname])
                print(f"  Loaded existing results from {os.path.basename(CSV_PATH)}")
        except Exception as e:
            print(f"  Could not load existing CSV: {e}")

    analyzable, no_file = get_analyzable_rows(rows)

    print(f"  {len(rows)} total rows, {no_file} missing files, {len(analyzable)} analyzable")

    if not analyzable:
        print("No analyzable diagrams found. Nothing to do.")
        sys.exit(0)

    clients = {
        "openai": OpenAI(api_key=OPENAI_API_KEY),
        "gemini": genai.Client(api_key=GEMINI_API_KEY),
        "claude": Anthropic(api_key=CLAUDE_API_KEY),
    }

    for model_cfg in MODELS:
        model_id = model_cfg["model"]
        print(f"\nRunning model: {model_id} ({model_cfg['provider']})")

        analyzed = run_model(model_cfg, rows, analyzable, clients)

        print(f"  {model_id}: {analyzed} diagram(s) analyzed.")
        save_csv(rows, fieldnames)  # save after each model so we can resume if it crashes

    print("\nFinal summary:")
    for model_cfg in MODELS:
        model_id = model_cfg["model"]
        col_uml = f"Final_UML_{model_id}"
        col_vp  = f"Final_Viewpoint_{model_id}"

        done    = sum(1 for r in rows if r.get(col_uml, "").strip())
        uml_yes = sum(1 for r in rows if r.get(col_uml, "").strip() == "Yes")

        print(f"\n  {model_id} ({model_cfg['provider']}):")
        print(f"      Analyzed: {done}/{len(rows)}")
        print(f"      UML: {uml_yes} | Non-UML: {done - uml_yes}")

        vp_counts = {}
        for r in rows:
            vp = r.get(col_vp, "").strip()
            if vp:
                vp_counts[vp] = vp_counts.get(vp, 0) + 1
        if vp_counts:
            print(f"      Viewpoints: {vp_counts}")

    print("\n  All models complete!")


if __name__ == "__main__":
    main()