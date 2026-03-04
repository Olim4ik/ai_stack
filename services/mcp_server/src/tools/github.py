"""GitHub tools — PR lookup and file contents."""

import json
from base64 import b64decode

import httpx
import structlog

logger = structlog.get_logger()

GITHUB_API = "https://api.github.com"


def _github_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
    }


async def github_get_pr(
    repo: str,
    pr_number: int,
    *,
    token: str,
) -> str:
    """Get details of a GitHub pull request."""
    if not token:
        return json.dumps({"error": True, "code": "GITHUB_NOT_CONFIGURED", "message": "GitHub token not configured"})

    url = f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}"

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=_github_headers(token), timeout=15)
            resp.raise_for_status()
            data = resp.json()

        return json.dumps({
            "number": data["number"],
            "title": data["title"],
            "state": data["state"],
            "author": data["user"]["login"],
            "description": (data.get("body") or "")[:500],
            "created_at": data["created_at"],
            "updated_at": data["updated_at"],
            "mergeable": data.get("mergeable"),
            "changed_files": data.get("changed_files", 0),
            "additions": data.get("additions", 0),
            "deletions": data.get("deletions", 0),
        })
    except httpx.HTTPError as e:
        logger.error("GitHub PR lookup failed", error=str(e))
        return json.dumps({"error": True, "code": "GITHUB_REQUEST_FAILED", "message": str(e)})


async def github_get_file(
    repo: str,
    path: str,
    ref: str = "main",
    *,
    token: str,
) -> str:
    """Read a file's contents from a GitHub repository."""
    if not token:
        return json.dumps({"error": True, "code": "GITHUB_NOT_CONFIGURED", "message": "GitHub token not configured"})

    url = f"{GITHUB_API}/repos/{repo}/contents/{path}"
    params = {"ref": ref}

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=_github_headers(token), params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

        if data.get("encoding") == "base64":
            content = b64decode(data["content"]).decode("utf-8")
        else:
            content = data.get("content", "")

        return json.dumps({
            "path": data["path"],
            "size": data["size"],
            "sha": data["sha"],
            "content": content[:10000],  # Limit content size
        })
    except httpx.HTTPError as e:
        logger.error("GitHub file fetch failed", error=str(e))
        return json.dumps({"error": True, "code": "GITHUB_REQUEST_FAILED", "message": str(e)})
