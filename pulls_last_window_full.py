
#!/usr/bin/env python3
"""
pulls_last_window_full.py
-------------------------
Fetch all pull requests updated/created/merged within the last N hours (default: 168)
for a single GitHub repository, along with rich associated data, and emit a single
chronological (oldest->newest by created_at) JSON file.

This script ALWAYS uses the GitHub API (no local git needed) and ALWAYS enriches PRs.

EASY EDITS (top of file):
- GITHUB_OWNER, GITHUB_REPO: which repository to fetch
- OUTPUT_DIR: where to write the JSON
- HOURS_BACK, TIMEZONE: how far back to fetch

REQUIRES:
- Python 3.9+
- Environment variable GITHUB_TOKEN (classic PAT or fine-grained with read access to the repo)
"""

import json
import os
import time
import sys
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

# =========================
# Configuration (EDIT ME)
# =========================

GITHUB_OWNER = "Cortado-Group"          # <-- EDIT: GitHub owner/org
GITHUB_REPO  = "django-project"         # <-- EDIT: GitHub repository name
OUTPUT_DIR   = "./reports"         # <-- EDIT: folder to write JSON into
HOURS_BACK   = 168                 # <-- EDIT: look back window (hours); 168 = 7 days
TIMEZONE     = "America/New_York"  # <-- EDIT: used to compute the window
FILENAME_PREFIX = "pulls_full"     # filename will be pulls_full_<since>_to_<until>.json

# Safety/size knobs
PAGE_SIZE = 100
INCLUDE_PR_COMMITS = True
INCLUDE_PR_FILES   = True
INCLUDE_REVIEWS    = True
INCLUDE_REVIEW_COMMENTS = True      # inline comments on diff
INCLUDE_ISSUE_COMMENTS  = True      # top-level conversation comments
INCLUDE_CHECKS_AND_STATUS = True    # check-runs + combined status per commit in the PR
MAX_ITEMS_PER_SECTION = None        # e.g., set 500 to cap very large PRs; None = no cap

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


class GitHub:
    def __init__(self, token: str, base: str = "https://api.github.com", debug: bool = False):
        cleaned_token = (token or "").strip()
        if not cleaned_token:
            raise ValueError("GitHub token must be non-empty")
        self.base = base
        self.token = cleaned_token
        self.debug = debug

    def headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"token {self.token}",
            "User-Agent": "pulls-last-window-full/1.0",
        }

    def _request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, str]]:
        if params:
            qs = urllib.parse.urlencode(params, doseq=True)
            url = f"{self.base}{path}?{qs}"
        else:
            url = f"{self.base}{path}"
        req = urllib.request.Request(url, method=method, headers=self.headers())
        backoff = 2.0
        for attempt in range(6):
            try:
                if self.debug:
                    print(f"[DEBUG] GitHub API {method} {url} (attempt {attempt + 1})", file=sys.stderr)
                with urllib.request.urlopen(req, timeout=45) as resp:
                    data = resp.read().decode("utf-8", "ignore")
                    if self.debug:
                        remaining = resp.headers.get("x-ratelimit-remaining")
                        print(
                            f"[DEBUG] -> status={resp.status} ratelimit_remaining={remaining}",
                            file=sys.stderr,
                        )
                    return json.loads(data), dict(resp.headers)
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "ignore")
                if self.debug:
                    print(
                        f"[DEBUG] HTTPError {e.code} for {method} {url}: {body[:200]}",
                        file=sys.stderr,
                    )
                # naive rate limit/backoff handling
                if e.code in (429, 502, 503) or (e.code == 403 and "rate limit" in body.lower()):
                    if attempt == 5:
                        raise
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise
            except urllib.error.URLError as e:
                if self.debug:
                    print(
                        f"[DEBUG] URLError for {method} {url}: {e}",
                        file=sys.stderr,
                    )
                if attempt == 5:
                    raise
                time.sleep(backoff)
                backoff *= 2
        raise RuntimeError("Unreachable retry loop")

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._request("GET", path, params)[0]

    def paginated(self, path: str, params: Optional[Dict[str, Any]] = None, per_page: int = PAGE_SIZE,
                  limit: Optional[int] = MAX_ITEMS_PER_SECTION) -> List[Any]:
        params = dict(params or {})
        page = 1
        out: List[Any] = []
        while True:
            params.update({"per_page": per_page, "page": page})
            data, headers = self._request("GET", path, params)
            if not isinstance(data, list) or not data:
                break
            out.extend(data)
            if limit and len(out) >= limit:
                return out[:limit]
            page += 1
        return out


def compute_window(hours_back: int, tz_name: str) -> Tuple[str, str]:
    tz = ZoneInfo(tz_name)
    now = datetime.now(tz)
    since_dt = now - timedelta(hours=hours_back)
    # Use ISO8601 with timezone offsets (GitHub supports RFC3339)
    return since_dt.isoformat(timespec="seconds"), now.isoformat(timespec="seconds")


def iso_date_only(iso_str: str) -> str:
    # For filenames
    return iso_str.replace(":", "-")


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def env_flag(name: str) -> bool:
    val = os.getenv(name)
    if val is None:
        return False
    return val.strip().lower() in {"1", "true", "yes", "on"}


def search_prs_in_window(gh: GitHub, owner: str, repo: str, since_iso: str, until_iso: str) -> List[Dict[str, Any]]:
    """
    Use the Search API to find PRs that were updated within the window.
    We grab enough fields in detail later; here we only need PR numbers.
    """
    # GitHub search uses dates with YYYY-MM-DDTHH:MM:SSZ or offset; we'll pass the since in UTC form if needed
    # Query both 'created' and 'updated' to not miss newly created + updated ones; also include 'merged' cutoff via filtering later.
    # We'll fetch in pages until empty.
    # Example q: repo:owner/name is:pr updated:>=2025-09-08T00:00:00-04:00
    all_numbers = set()
    for qualifier in ("created", "updated", "merged"):
        page = 1
        while True:
            q = f"repo:{owner}/{repo} is:pr {qualifier}:>={since_iso}"
            params = {"q": q, "sort": "updated", "order": "desc", "per_page": 100, "page": page}
            data, headers = gh._request("GET", "/search/issues", params)
            items = data.get("items", [])
            if not items:
                break
            for it in items:
                # item number exists for PRs via search/issues; ensure it's a PR
                if "pull_request" in it:
                    all_numbers.add(it["number"])
            page += 1
            # Basic guard against excessive pages
            if page > 50:
                break
    # We'll still filter precisely by timestamps once we fetch each PR detail below.
    return sorted(all_numbers)


def fetch_pr_full(gh: GitHub, owner: str, repo: str, number: int, since_iso: str, until_iso: str) -> Optional[Dict[str, Any]]:
    """
    Fetch full PR details + associated data. Returns None if PR does not fall within the time window
    by created/updated/merged at (defensive extra filter).
    """
    pr = gh.get(f"/repos/{owner}/{repo}/pulls/{number}")
    # Timestamps
    created_at = pr.get("created_at")
    updated_at = pr.get("updated_at")
    merged_at  = pr.get("merged_at")
    # Window filter (inclusive on since, exclusive on until)
    def in_window(ts: Optional[str]) -> bool:
        if not ts:
            return False
        return since_iso <= ts < until_iso

    if not (in_window(created_at) or in_window(updated_at) or in_window(merged_at)):
        # Outside window; drop
        return None

    # Core PR block
    pr_block: Dict[str, Any] = {
        "number": pr.get("number"),
        "title": pr.get("title"),
        "state": pr.get("state"),
        "draft": pr.get("draft"),
        "locked": pr.get("locked"),
        "created_at": created_at,
        "updated_at": updated_at,
        "merged_at": merged_at,
        "closed_at": pr.get("closed_at"),
        "user": pr.get("user", {}).get("login"),
        "author_association": pr.get("author_association"),
        "url": pr.get("url"),
        "html_url": pr.get("html_url"),
        "base": {
            "ref": pr.get("base", {}).get("ref"),
            "sha": pr.get("base", {}).get("sha"),
            "repo_full_name": pr.get("base", {}).get("repo", {}).get("full_name"),
        },
        "head": {
            "ref": pr.get("head", {}).get("ref"),
            "sha": pr.get("head", {}).get("sha"),
            "repo_full_name": pr.get("head", {}).get("repo", {}).get("full_name") if pr.get("head", {}).get("repo") else None,
        },
        "mergeable": pr.get("mergeable"),
        "merged": pr.get("merged"),
        "maintainer_can_modify": pr.get("maintainer_can_modify"),
        "rebaseable": pr.get("rebaseable"),
        "commits_count": pr.get("commits"),
        "additions": pr.get("additions"),
        "deletions": pr.get("deletions"),
        "changed_files": pr.get("changed_files"),
        "labels": [lbl.get("name") for lbl in (pr.get("labels") or [])],
        "milestone": (pr.get("milestone") or {}).get("title") if pr.get("milestone") else None,
        "assignees": [u.get("login") for u in (pr.get("assignees") or [])],
        "requested_reviewers": [u.get("login") for u in (pr.get("requested_reviewers") or [])],
        "requested_teams": [t.get("slug") for t in (pr.get("requested_teams") or [])],
        "body": pr.get("body"),
    }

    # Reviews
    if INCLUDE_REVIEWS:
        reviews = gh.paginated(f"/repos/{owner}/{repo}/pulls/{number}/reviews")
        pr_block["reviews"] = [
            {
                "user": rv.get("user", {}).get("login"),
                "state": rv.get("state"),
                "submitted_at": rv.get("submitted_at"),
            }
            for rv in reviews
        ]
        # Simple review summary
        counts: Dict[str, int] = {}
        for rv in pr_block["reviews"]:
            s = rv.get("state") or "UNKNOWN"
            counts[s] = counts.get(s, 0) + 1
        pr_block["review_summary"] = counts

    # Review comments (inline)
    if INCLUDE_REVIEW_COMMENTS:
        rc = gh.paginated(f"/repos/{owner}/{repo}/pulls/{number}/comments")
        pr_block["review_comments"] = [
            {
                "user": c.get("user", {}).get("login"),
                "created_at": c.get("created_at"),
                "path": c.get("path"),
                "position": c.get("position"),
                "original_position": c.get("original_position"),
                "body": c.get("body"),
                "diff_hunk": c.get("diff_hunk"),
                "html_url": c.get("html_url"),
            } for c in rc
        ]

    # Issue comments (conversation)
    if INCLUDE_ISSUE_COMMENTS:
        ic = gh.paginated(f"/repos/{owner}/{repo}/issues/{number}/comments")
        pr_block["issue_comments"] = [
            {
                "user": c.get("user", {}).get("login"),
                "created_at": c.get("created_at"),
                "body": c.get("body"),
                "html_url": c.get("html_url"),
            } for c in ic
        ]

    # Commits in PR
    if INCLUDE_PR_COMMITS:
        commits = gh.paginated(f"/repos/{owner}/{repo}/pulls/{number}/commits")
        pr_block["commits"] = [
            {
                "sha": cm.get("sha"),
                "author_login": (cm.get("author") or {}).get("login") if cm.get("author") else None,
                "commit": {
                    "author": cm.get("commit", {}).get("author"),
                    "committer": cm.get("commit", {}).get("committer"),
                    "message": cm.get("commit", {}).get("message"),
                    "verification": (cm.get("commit", {}).get("verification") or {}),
                },
                "html_url": cm.get("html_url"),
            } for cm in commits[:(MAX_ITEMS_PER_SECTION or len(commits))]
        ]

        # Checks & status per commit (head + others)
        if INCLUDE_CHECKS_AND_STATUS:
            for cm in pr_block["commits"]:
                sha = cm["sha"]
                try:
                    checks = gh.get(f"/repos/{owner}/{repo}/commits/{sha}/check-runs")
                    runs = checks.get("check_runs", [])
                    cm["checks"] = [
                        {
                            "name": r.get("name"),
                            "status": r.get("status"),
                            "conclusion": r.get("conclusion"),
                            "details_url": r.get("details_url"),
                        }
                        for r in runs
                    ]
                except Exception as e:
                    cm["checks_error"] = str(e)
                try:
                    comb = gh.get(f"/repos/{owner}/{repo}/commits/{sha}/status")
                    cm["combined_status"] = {
                        "state": comb.get("state"),
                        "statuses": [
                            {
                                "context": s.get("context"),
                                "state": s.get("state"),
                                "description": s.get("description"),
                                "target_url": s.get("target_url"),
                            } for s in comb.get("statuses", [])
                        ]
                    }
                except Exception as e:
                    cm["status_error"] = str(e)

    # Files in PR
    if INCLUDE_PR_FILES:
        files = gh.paginated(f"/repos/{owner}/{repo}/pulls/{number}/files")
        pr_block["files"] = [
            {
                "filename": f.get("filename"),
                "status": f.get("status"),
                "additions": f.get("additions"),
                "deletions": f.get("deletions"),
                "changes": f.get("changes"),
                "previous_filename": f.get("previous_filename"),
                "sha": f.get("sha"),
                "blob_url": f.get("blob_url"),
                "raw_url": f.get("raw_url"),
                "patch_truncated": ("patch" not in f),  # GitHub may omit patch for large files
            } for f in files[:(MAX_ITEMS_PER_SECTION or len(files))]
        ]

    return pr_block


def main() -> None:
    token = os.getenv("GITHUB_TOKEN")
    if token:
        token = token.strip()
    if not token:
        die("GITHUB_TOKEN is not set. Please export a GitHub token with repo read access.")
    print(f"Using token: {token[:6]}...{token[-4:]}")
    if not GITHUB_OWNER or not GITHUB_REPO:
        die("Please edit GITHUB_OWNER and GITHUB_REPO at the top of the script.")

    since_iso, until_iso = compute_window(HOURS_BACK, TIMEZONE)

    debug_requests = env_flag("GITHUB_API_DEBUG") or env_flag("DEBUG_GITHUB_API")
    if debug_requests:
        print("[DEBUG] GitHub API debugging enabled.", file=sys.stderr)

    gh = GitHub(token, debug=debug_requests)

    # Find candidate PR numbers via Search API (created/updated/merged in window)
    candidate_numbers = search_prs_in_window(gh, GITHUB_OWNER, GITHUB_REPO, since_iso, until_iso)

    # Fetch and filter each PR fully
    prs: List[Dict[str, Any]] = []
    for num in candidate_numbers:
        try:
            block = fetch_pr_full(gh, GITHUB_OWNER, GITHUB_REPO, num, since_iso, until_iso)
            if block:
                prs.append(block)
        except Exception as e:
            # Keep going; record a placeholder with error
            prs.append({"number": num, "error": str(e)})

    # Sort oldest -> newest by created_at
    def keyfn(pr: Dict[str, Any]) -> str:
        return pr.get("created_at") or ""

    prs.sort(key=keyfn)

    # Build report
    report = {
        "repo": f"{GITHUB_OWNER}/{GITHUB_REPO}",
        "window": {
            "since": since_iso,
            "until": until_iso,
            "timezone": TIMEZONE,
            "hours_back": HOURS_BACK,
        },
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "version": 1,
        "count_prs": len(prs),
        "pulls": prs,
    }

    # Ensure output dir and write
    ensure_output_dir(OUTPUT_DIR)
    filename = f"{FILENAME_PREFIX}_{iso_date_only(since_iso)}_to_{iso_date_only(until_iso)}.json"
    out_path = os.path.join(OUTPUT_DIR, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Wrote {out_path} with {len(prs)} pull requests.")

if __name__ == "__main__":
    main()
