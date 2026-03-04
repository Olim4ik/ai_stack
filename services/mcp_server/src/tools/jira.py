"""Jira tools — search and create tickets."""

import json
from base64 import b64encode

import httpx
import structlog

logger = structlog.get_logger()


def _jira_headers(email: str, token: str) -> dict:
    auth = b64encode(f"{email}:{token}".encode()).decode()
    return {
        "Authorization": f"Basic {auth}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


async def jira_search(
    jql: str,
    max_results: int = 10,
    *,
    base_url: str,
    email: str,
    token: str,
) -> str:
    """Search Jira tickets by JQL query."""
    if not base_url or not token:
        return json.dumps({"error": True, "code": "JIRA_NOT_CONFIGURED", "message": "Jira credentials not configured"})

    url = f"{base_url}/rest/api/3/search"
    params = {"jql": jql, "maxResults": max_results, "fields": "summary,status,assignee,priority"}

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=_jira_headers(email, token), params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

        tickets = []
        for issue in data.get("issues", []):
            fields = issue.get("fields", {})
            tickets.append({
                "key": issue["key"],
                "summary": fields.get("summary", ""),
                "status": fields.get("status", {}).get("name", ""),
                "assignee": (fields.get("assignee") or {}).get("displayName", "Unassigned"),
                "priority": (fields.get("priority") or {}).get("name", "None"),
            })

        return json.dumps({"tickets": tickets, "total": data.get("total", 0)})
    except httpx.HTTPError as e:
        logger.error("Jira search failed", error=str(e))
        return json.dumps({"error": True, "code": "JIRA_REQUEST_FAILED", "message": str(e)})


async def jira_create(
    project: str,
    summary: str,
    description: str = "",
    issue_type: str = "Task",
    priority: str = "Medium",
    *,
    base_url: str,
    email: str,
    token: str,
) -> str:
    """Create a new Jira ticket."""
    if not base_url or not token:
        return json.dumps({"error": True, "code": "JIRA_NOT_CONFIGURED", "message": "Jira credentials not configured"})

    url = f"{base_url}/rest/api/3/issue"
    payload = {
        "fields": {
            "project": {"key": project},
            "summary": summary,
            "description": {"type": "doc", "version": 1, "content": [{"type": "paragraph", "content": [{"type": "text", "text": description or summary}]}]},
            "issuetype": {"name": issue_type},
            "priority": {"name": priority},
        }
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=_jira_headers(email, token), json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()

        return json.dumps({"key": data["key"], "url": f"{base_url}/browse/{data['key']}", "message": f"Ticket {data['key']} created"})
    except httpx.HTTPError as e:
        logger.error("Jira create failed", error=str(e))
        return json.dumps({"error": True, "code": "JIRA_CREATE_FAILED", "message": str(e)})
