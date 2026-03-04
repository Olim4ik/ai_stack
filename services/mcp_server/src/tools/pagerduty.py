"""PagerDuty tools — list and acknowledge incidents."""

import json

import httpx
import structlog

logger = structlog.get_logger()

PAGERDUTY_API = "https://api.pagerduty.com"


def _pd_headers(api_key: str) -> dict:
    return {
        "Authorization": f"Token token={api_key}",
        "Content-Type": "application/json",
        "Accept": "application/vnd.pagerduty+json;version=2",
    }


async def pagerduty_incidents(
    status: str = "triggered",
    service_name: str = "",
    *,
    api_key: str,
) -> str:
    """List current PagerDuty incidents."""
    if not api_key:
        return json.dumps({"error": True, "code": "PD_NOT_CONFIGURED", "message": "PagerDuty API key not configured"})

    url = f"{PAGERDUTY_API}/incidents"
    params = {"statuses[]": status, "sort_by": "created_at:desc", "limit": 25}

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=_pd_headers(api_key), params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

        incidents = []
        for inc in data.get("incidents", []):
            service = inc.get("service", {}).get("summary", "")
            if service_name and service_name.lower() not in service.lower():
                continue
            incidents.append({
                "id": inc["id"],
                "title": inc.get("title", ""),
                "status": inc.get("status", ""),
                "urgency": inc.get("urgency", ""),
                "service": service,
                "created_at": inc.get("created_at", ""),
            })

        return json.dumps({"incidents": incidents, "total": len(incidents)})
    except httpx.HTTPError as e:
        logger.error("PagerDuty incidents fetch failed", error=str(e))
        return json.dumps({"error": True, "code": "PD_REQUEST_FAILED", "message": str(e)})


async def pagerduty_ack(
    incident_id: str,
    *,
    api_key: str,
) -> str:
    """Acknowledge a PagerDuty incident."""
    if not api_key:
        return json.dumps({"error": True, "code": "PD_NOT_CONFIGURED", "message": "PagerDuty API key not configured"})

    url = f"{PAGERDUTY_API}/incidents/{incident_id}"
    payload = {
        "incident": {
            "type": "incident_reference",
            "status": "acknowledged",
        }
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.put(url, headers=_pd_headers(api_key), json=payload, timeout=15)
            resp.raise_for_status()

        return json.dumps({"incident_id": incident_id, "status": "acknowledged", "message": f"Incident {incident_id} acknowledged"})
    except httpx.HTTPError as e:
        logger.error("PagerDuty ack failed", error=str(e))
        return json.dumps({"error": True, "code": "PD_ACK_FAILED", "message": str(e)})
