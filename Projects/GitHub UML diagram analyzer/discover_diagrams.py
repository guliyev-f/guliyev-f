#!/usr/bin/env python3
"""
Searches GitHub for software diagram files and downloads them to the Mined dataset folder.
Runs a dimension check and GPT pre-filter on images before saving.
Resume-safe: skips anything already in the CSV index.
"""

import base64
import csv
import io
import json
import os
import re
import time

import requests
from PIL import Image
from openai import OpenAI


GITHUB_TOKEN   = "GITHUB TOKEN"
OPENAI_API_KEY = "OPEN AI KEY"

TARGET_DIAGRAMS = 1

# max per repo to keep things diverse
MAX_DIAGRAMS_PER_REPO = 10

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR     = os.path.dirname(SCRIPT_DIR)
DIAGRAMS_FOLDER = os.path.join(PROJECT_DIR, "diagrams", "Mined dataset")
DATA_FOLDER     = os.path.join(PROJECT_DIR, "data")

INDEX_CSV = "diagram_index.csv"

SEARCH_QUERIES = [
    # text diagram formats
    "@startuml extension:puml",
    "classDiagram extension:mmd",
    "sequenceDiagram extension:mmd",
    "mxfile extension:drawio",
    "digraph extension:dot",
    "workspace extension:dsl",
    "C4Context extension:puml",
    "archimate extension:xml",
    # repos with diagram images in docs
    "architecture.png extension:md",
    "deployment-diagram extension:md",
    "component-diagram extension:md",
    "software architecture extension:md",
    "system design diagram extension:md",
]

REPO_SEARCH_QUERIES = [
    "software architecture diagram",
    "system design architecture",
    "microservice architecture",
    "deployment architecture diagram",
    "C4 model architecture",
    "cloud architecture",
]
REPO_SEARCH_PAGES = 1

SEARCH_PAGES_PER_QUERY = 1

SEARCH_START_PAGE      = 1
REPO_SEARCH_START_PAGE = 1

MAX_REPOS_TO_SCAN = 10

TEXT_DIAGRAM_EXTENSIONS = {
    ".puml", ".plantuml", ".wsd",
    ".mmd", ".mermaid",
    ".drawio", ".bpmn",
    ".dot", ".gv",
    ".c4", ".dsl",
    ".uml", ".xmi",
    ".archimate",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".webp"}

IMPLEMENTATION_EXTENSIONS = {
    ".java", ".py", ".js", ".ts", ".go", ".cs", ".cpp", ".c", ".h",
    ".rb", ".rs", ".kt", ".swift", ".scala", ".php", ".dart", ".ex",
    ".exs", ".hs", ".lua", ".r", ".m", ".mm", ".vue", ".jsx", ".tsx",
}

DIAGRAM_KEYWORDS = [
    "architecture", "diagram", "design", "topology", "deployment",
    "c4", "infrastructure", "context", "container", "uml", "model",
    "sequence", "component", "class-diagram", "erd", "flow",
    "overview", "system", "network",
]

MIN_IMAGE_DIMENSION = 200
MAX_ASPECT_RATIO    = 5.0
MIN_IMAGE_BYTES     = 5_000

GPT_PREFILTER_PROMPT = """
You are an image classifier. Your only job is to decide whether the given
image is a architecture diagram, system design diagram, or
technical infrastructure diagram.

Examples of diagrams to ACCEPT:
- UML class / component / deployment diagrams
- Microservice topologies
- Cloud infrastructure diagrams
- Sequence diagrams
- Data-flow / pipeline diagrams
- C4 model diagrams

Examples of images to REJECT:
- Logos, icons, badges, banners
- Screenshots of code or terminal output
- UI mockups, wireframes
- Photos, illustrations, memes
- ER / database-only diagrams
- Charts, graphs, plots
"""


GITHUB_HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}


def gh_get(url: str, params: dict = None, timeout: int = 30) -> requests.Response:
    """Authenticated GET with retry on rate-limit and server errors."""
    backoff_schedule = [10, 30, 60]

    for attempt in range(len(backoff_schedule) + 1):
        try:
            resp = requests.get(url, headers=GITHUB_HEADERS, params=params, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt < len(backoff_schedule):
                wait = backoff_schedule[attempt]
                print(f"  Connection error: {e}. Retry {attempt + 1}/{len(backoff_schedule)} in {wait}s ...")
                time.sleep(wait)
                continue
            else:
                raise

        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            reset_ts = resp.headers.get("X-RateLimit-Reset")
            wait_seconds = max(int(reset_ts) - int(time.time()), 5) if reset_ts else 60
            print(f"  Rate-limited. Sleeping {wait_seconds}s ...")
            time.sleep(wait_seconds)
            continue

        if resp.status_code >= 500 and attempt < len(backoff_schedule):
            wait = backoff_schedule[attempt]
            print(f"  Server error ({resp.status_code}). Retry {attempt + 1}/{len(backoff_schedule)} in {wait}s ...")
            time.sleep(wait)
            continue

        break

    resp.raise_for_status()
    return resp


def search_code(query: str, pages: int = SEARCH_PAGES_PER_QUERY, per_page: int = 100, start_page: int = 1) -> dict:
    """GitHub code search. Returns {full_name: {html_url, files: set}}."""
    results = {}
    for page in range(start_page, start_page + pages):
        try:
            resp = gh_get(
                "https://api.github.com/search/code",
                params={"q": query, "per_page": per_page, "page": page},
            )
            data = resp.json()
            for item in data.get("items", []):
                repo      = item["repository"]
                full_name = repo["full_name"]
                if full_name not in results:
                    results[full_name] = {"html_url": repo["html_url"], "files": set()}
                results[full_name]["files"].add(item.get("path", ""))

            if len(data.get("items", [])) < per_page:
                break

        except requests.HTTPError as e:
            print(f"  Code search error for '{query}' page {page}: {e}")
            break

        time.sleep(7)  # GitHub code search has a strict rate limit

    return results


def search_repositories(query: str, pages: int = REPO_SEARCH_PAGES, per_page: int = 100, start_page: int = 1) -> dict:
    """GitHub repo search by name/description. Files are filled in during tree walk."""
    results = {}
    for page in range(start_page, start_page + pages):
        try:
            resp = gh_get(
                "https://api.github.com/search/repositories",
                params={"q": query, "sort": "stars", "per_page": per_page, "page": page},
            )
            data = resp.json()
            for repo in data.get("items", []):
                full_name = repo["full_name"]
                if full_name not in results:
                    results[full_name] = {"html_url": repo.get("html_url", ""), "files": set()}

            if len(data.get("items", [])) < per_page:
                break

        except requests.HTTPError as e:
            print(f"  Repo search error for '{query}' page {page}: {e}")
            break

        time.sleep(3)

    return results


def get_repo_metadata(full_name: str) -> dict | None:
    """Returns branch, stars, forks or None if the repo is private/inaccessible."""
    try:
        meta = gh_get(f"https://api.github.com/repos/{full_name}").json()
        if meta.get("private"):
            return None
        return {
            "branch": meta.get("default_branch", "main"),
            "stars":  meta.get("stargazers_count", 0),
            "forks":  meta.get("forks_count", 0),
        }
    except requests.HTTPError:
        return None


def get_priority_tier(diagram_count: int) -> tuple[int, str]:
    if diagram_count <= 20:
        return (1, "high")
    elif diagram_count <= 100:
        return (2, "medium")
    elif diagram_count <= 300:
        return (3, "low")
    else:
        return (4, "lowest")


def get_repo_tree(full_name: str, branch: str) -> list:
    """Fetch the full recursive file tree. Returns [] on truncation or error."""
    try:
        data = gh_get(
            f"https://api.github.com/repos/{full_name}/git/trees/{branch}",
            params={"recursive": "1"},
        ).json()
        if data.get("truncated"):
            return []
        return data.get("tree", [])
    except requests.HTTPError:
        return []


def get_file_extension(path: str) -> str:
    _, ext = os.path.splitext(path.lower())
    return ext

def is_diagram_file(path: str) -> bool:
    """Text diagram formats are always accepted; images only if path has a diagram keyword."""
    ext = get_file_extension(path)
    if ext in TEXT_DIAGRAM_EXTENSIONS:
        return True
    if ext in IMAGE_EXTENSIONS:
        return any(kw in path.lower() for kw in DIAGRAM_KEYWORDS)
    return False

def is_image_file(path: str) -> bool:
    return get_file_extension(path) in IMAGE_EXTENSIONS


def score_diagram_path(path: str, all_paths: set = None) -> tuple:
    """
    Score a path for ordering - lower is better.
    Combines directory quality, keyword relevance, format, and proximity to code.
    """
    p = path.lower()

    dir_score = 2
    if any(d in p for d in ("test/", "example/", "sample/", "demo/", "tutorial/")):
        dir_score = 4
    elif any(d in p for d in ("doc/", "docs/", "architecture/", "design/", "diagrams/")):
        dir_score = 0

    SYSTEM_LEVEL = ["architecture", "system", "overview", "context"]
    DEPLOYMENT   = ["deployment", "infrastructure", "container"]
    STRUCTURAL   = ["component", "sequence", "topology", "class", "erd", "flow"]

    if any(kw in p for kw in SYSTEM_LEVEL):
        keyword_score = 0
    elif any(kw in p for kw in DEPLOYMENT):
        keyword_score = 1
    elif any(kw in p for kw in STRUCTURAL):
        keyword_score = 2
    else:
        keyword_score = 3

    format_score = 0 if is_image_file(path) else 3

    proximity_score = 2
    if all_paths:
        diagram_dir = os.path.dirname(path)
        parent_dir  = os.path.dirname(diagram_dir)
        for other_path in all_paths:
            ext = os.path.splitext(other_path.lower())[1]
            if ext in IMPLEMENTATION_EXTENSIONS:
                p_dir = os.path.dirname(other_path)
                if p_dir == diagram_dir or p_dir == parent_dir:
                    proximity_score = 0
                    break

    return (dir_score + keyword_score + format_score + proximity_score, path)


def build_raw_url(full_name: str, branch: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{full_name}/{branch}/{path}"

def build_html_url(full_name: str, branch: str, path: str) -> str:
    return f"https://github.com/{full_name}/blob/{branch}/{path}"


def passes_dimension_check(image_data: bytes) -> bool:
    """Rejects images that are too small or have extreme aspect ratios (banners)."""
    if len(image_data) < MIN_IMAGE_BYTES:
        return False
    try:
        img  = Image.open(io.BytesIO(image_data))
        w, h = img.size
        if w < MIN_IMAGE_DIMENSION or h < MIN_IMAGE_DIMENSION:
            return False
        ratio = max(w, h) / max(min(w, h), 1)
        if ratio > MAX_ASPECT_RATIO:
            return False
        return True
    except Exception:
        return False


def passes_gpt_prefilter(client: OpenAI, image_data: bytes, filename: str) -> bool:
    """Ask GPT-4o-mini whether this image is a software diagram. Fails open on error."""
    ext = get_file_extension(filename)
    mime_map = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".svg": "image/svg+xml", ".webp": "image/webp",
    }
    mime_type = mime_map.get(ext, "image/png")
    b64 = base64.b64encode(image_data).decode("utf-8")

    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            instructions=GPT_PREFILTER_PROMPT,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Is this image a diagram?"},
                    {"type": "input_image", "image_url": f"data:{mime_type};base64,{b64}", "detail": "low"},
                ],
            }],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "diagram_prefilter",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "is_diagram": {"type": "boolean"},
                            "reason":     {"type": "string"},
                        },
                        "required": ["is_diagram", "reason"],
                        "additionalProperties": False,
                    },
                }
            },
        )
        result = json.loads(response.output_text)
        return result.get("is_diagram", False)

    except Exception as e:
        print(f"    GPT pre-filter error: {e}")
        return True  # fail open so we don't lose diagrams on transient errors


def sanitize_filename(full_name: str, file_path: str) -> str:
    """Turn repo + path into a flat filename that's safe on all OSes."""
    combined = f"{full_name}/{file_path}"
    safe = re.sub(r'[/\\]', '__', combined)
    safe = re.sub(r'[^\w.\-]', '_', safe)
    return safe


def download_file(raw_url: str, save_path: str, is_image: bool,
                  openai_client: OpenAI = None, filename: str = "") -> str:
    """
    Download and optionally filter a file.
    Returns 'ok', 'filtered_dimension', 'filtered_gpt', or 'error'.
    """
    try:
        resp = requests.get(raw_url, timeout=30)
        if resp.status_code != 200:
            return "error"

        # skip very large files
        max_size = 5_000_000 if is_image else 1_000_000
        if len(resp.content) > max_size:
            print(f"    File too large ({len(resp.content)} bytes), skipping")
            return "error"

        if is_image:
            if not passes_dimension_check(resp.content):
                print(f"    Filtered out by dimension check")
                return "filtered_dimension"

            if openai_client:
                if not passes_gpt_prefilter(openai_client, resp.content, filename):
                    print(f"    Filtered out by GPT pre-filter (not a diagram)")
                    return "filtered_gpt"

            with open(save_path, "wb") as f:
                f.write(resp.content)
        else:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(resp.text)

        return "ok"

    except Exception as e:
        print(f"    Download failed: {e}")
        return "error"


def load_already_scraped(csv_path: str) -> tuple[set, set]:
    """Read existing CSV and return sets of already-scraped diagram URLs and repo URLs."""
    scraped_diagrams = set()
    scraped_repos    = set()
    if os.path.isfile(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = row.get("diagram_url", "").strip()
                if url:
                    scraped_diagrams.add(url)
                repo_url = row.get("repo_url", "").strip()
                if repo_url:
                    scraped_repos.add(repo_url)
    return scraped_diagrams, scraped_repos


def main():
    print("GitHub Diagram Discovery & Download\n")

    os.makedirs(DIAGRAMS_FOLDER, exist_ok=True)
    os.makedirs(DATA_FOLDER, exist_ok=True)

    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    csv_path = os.path.join(DATA_FOLDER, INDEX_CSV)
    already_scraped, already_scraped_repos = load_already_scraped(csv_path)
    if already_scraped:
        print(f"  Found {len(already_scraped)} already-scraped diagram(s) from {len(already_scraped_repos)} repo(s) - will skip\n")
    else:
        print(f"  No previous CSV found - starting fresh\n")

    # Phase 1: find repos via code search + repo search
    print("Phase 1: Searching GitHub for diagram repositories ...")

    all_repos = {}
    if SEARCH_START_PAGE > 1:
        print(f"  Code search starting from page {SEARCH_START_PAGE}")

    for i, query in enumerate(SEARCH_QUERIES, 1):
        print(f"\n  [{i}/{len(SEARCH_QUERIES)}] Searching: {query}")
        found = search_code(query, pages=SEARCH_PAGES_PER_QUERY, start_page=SEARCH_START_PAGE)
        for full_name, info in found.items():
            if full_name not in all_repos:
                all_repos[full_name] = info
            else:
                all_repos[full_name]["files"].update(info["files"])

    print(f"\n  Code search: {len(all_repos)} unique repositories")

    if REPO_SEARCH_START_PAGE > 1:
        print(f"  Repo search starting from page {REPO_SEARCH_START_PAGE}")

    for i, query in enumerate(REPO_SEARCH_QUERIES, 1):
        print(f"  [{i}/{len(REPO_SEARCH_QUERIES)}] Repo search: {query}")
        found = search_repositories(query, pages=REPO_SEARCH_PAGES, start_page=REPO_SEARCH_START_PAGE)
        for full_name, info in found.items():
            if full_name not in all_repos:
                all_repos[full_name] = info
    print(f"  After repo search: {len(all_repos)} total repositories")

    # Phase 2: pre-scan repos for metadata and diagram counts
    print("\nPhase 2: Pre-scanning repos ...")

    scanned_repos            = []
    total_candidate_diagrams = 0
    repos_to_prescan         = list(all_repos.items())[:MAX_REPOS_TO_SCAN]

    for idx, (full_name, info) in enumerate(repos_to_prescan, 1):
        if total_candidate_diagrams >= TARGET_DIAGRAMS * 3:
            print(f"\n  Early exit: {total_candidate_diagrams} candidates found, enough for target of {TARGET_DIAGRAMS}")
            break

        repo_url = f"https://github.com/{full_name}"
        if repo_url in already_scraped_repos:
            print(f"\n  [{idx}/{len(repos_to_prescan)}] Already scraped: {full_name}")
            continue

        print(f"\n  [{idx}/{len(repos_to_prescan)}] Pre-scanning: {full_name}")

        meta = get_repo_metadata(full_name)
        if not meta:
            print(f"    Could not get metadata, skipping")
            continue

        branch = meta["branch"]
        stars  = meta["stars"]

        candidate_paths = set(info.get("files", set()))
        tree = get_repo_tree(full_name, branch)
        for item in tree:
            if item.get("type") == "blob":
                candidate_paths.add(item.get("path", ""))

        diagram_paths = sorted(
            (fpath for fpath in candidate_paths if fpath and is_diagram_file(fpath)),
            key=lambda p: score_diagram_path(p, candidate_paths),
        )

        if not diagram_paths:
            print(f"    No diagram files found, skipping")
            continue

        tier_order, tier_label = get_priority_tier(len(diagram_paths))
        print(f"    {len(diagram_paths)} diagram(s) | {stars} stars | tier: {tier_label}")

        scanned_repos.append({
            "full_name":    full_name,
            "html_url":     info.get("html_url", ""),
            "branch":       branch,
            "stars":        stars,
            "tier_order":   tier_order,
            "tier_label":   tier_label,
            "diagram_count": len(diagram_paths),
            "diagram_paths": diagram_paths,
        })

        total_candidate_diagrams += min(len(diagram_paths), MAX_DIAGRAMS_PER_REPO)
        time.sleep(0.5)

    scanned_repos.sort(key=lambda r: (r["tier_order"], -r["stars"]))

    print(f"\n  Pre-scan complete: {len(scanned_repos)} repos with diagrams")
    print(f"     high (<=20):    {sum(1 for r in scanned_repos if r['tier_label'] == 'high')}")
    print(f"     medium (21-100): {sum(1 for r in scanned_repos if r['tier_label'] == 'medium')}")
    print(f"     low (101-300):  {sum(1 for r in scanned_repos if r['tier_label'] == 'low')}")
    print(f"     lowest (>300):  {sum(1 for r in scanned_repos if r['tier_label'] == 'lowest')}")

    # Phase 3: download
    print("\nPhase 3: Downloading diagrams ...")

    seen              = set()
    csv_rows          = []
    download_count    = 0
    skipped_existing  = 0
    filtered_dimension = 0
    filtered_gpt      = 0
    repo_download_counts = {}

    for idx, repo in enumerate(scanned_repos, 1):
        if download_count >= TARGET_DIAGRAMS:
            break

        full_name    = repo["full_name"]
        branch       = repo["branch"]
        repo_downloaded = 0

        print(f"\n  [{idx}/{len(scanned_repos)}] {full_name} "
              f"(tier: {repo['tier_label']}, {repo['stars']} stars, "
              f"total: {download_count}/{TARGET_DIAGRAMS})")

        for fpath in repo["diagram_paths"]:
            if download_count >= TARGET_DIAGRAMS:
                break
            if repo_downloaded >= MAX_DIAGRAMS_PER_REPO:
                print(f"    Per-repo cap ({MAX_DIAGRAMS_PER_REPO}) reached, moving on")
                break

            key = (full_name, fpath)
            if key in seen:
                continue
            seen.add(key)

            raw_url  = build_raw_url(full_name, branch, fpath)
            html_url = build_html_url(full_name, branch, fpath)

            if html_url in already_scraped:
                skipped_existing += 1
                continue

            local_filename = sanitize_filename(full_name, fpath)
            save_path      = os.path.join(DIAGRAMS_FOLDER, local_filename)

            image = is_image_file(fpath)
            print(f"    {'Image' if image else 'Text'}: {fpath}")
            status = download_file(
                raw_url, save_path, is_image=image,
                openai_client=openai_client, filename=local_filename,
            )

            if status == "ok":
                download_count  += 1
                repo_downloaded += 1
                csv_rows.append({
                    "repo_url":    f"https://github.com/{full_name}",
                    "diagram_url": html_url,
                    "filename":    local_filename,
                })
                print(f"       Saved as: {local_filename}")
            elif status == "filtered_dimension":
                filtered_dimension += 1
            elif status == "filtered_gpt":
                filtered_gpt += 1

        if repo_downloaded > 0:
            repo_download_counts[full_name] = {
                "count": repo_downloaded,
                "tier":  repo["tier_label"],
                "stars": repo["stars"],
            }

        time.sleep(0.5)

    # Phase 4: update CSV index
    print("\nPhase 4: Updating CSV index ...")

    csv_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["repo_url", "diagram_url", "filename"])
        if not csv_exists:
            writer.writeheader()
        writer.writerows(csv_rows)

    total_in_csv = len(already_scraped) + download_count
    new_repos    = len(repo_download_counts)
    print(f"\n  Done! {download_count} new diagram(s) from {new_repos} repo(s) -> {DIAGRAMS_FOLDER}/")
    if skipped_existing > 0:
        print(f"  Skipped {skipped_existing} already-scraped diagram(s)")
    print(f"  Filtered by dimension check: {filtered_dimension}")
    print(f"  Filtered by GPT pre-filter:  {filtered_gpt}")
    print(f"  CSV index: {total_in_csv} total entries ({csv_path})")

    if repo_download_counts:
        print(f"\n  Per-repo breakdown:")
        for name, info in sorted(repo_download_counts.items(), key=lambda x: x[1]["count"], reverse=True):
            dots = "." * max(1, 50 - len(name))
            print(f"    {name} {dots} {info['count']} diagram(s)  (tier: {info['tier']}, {info['stars']} stars)")
        print(f"  Total: {download_count} diagrams from {len(repo_download_counts)} repositories")


if __name__ == "__main__":
    main()
