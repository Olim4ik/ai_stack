"""Unit tests for MCP server tools (mocked external APIs)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.mcp_server.src.tools.jira import jira_create, jira_search
from services.mcp_server.src.tools.github import github_get_file, github_get_pr
from services.mcp_server.src.tools.pagerduty import pagerduty_ack, pagerduty_incidents


class TestJiraTools:
    @pytest.mark.asyncio
    async def test_jira_search_not_configured(self):
        result = json.loads(await jira_search(
            jql="project = PLAT", base_url="", email="", token=""
        ))
        assert result["error"] is True
        assert result["code"] == "JIRA_NOT_CONFIGURED"

    @pytest.mark.asyncio
    @patch("services.mcp_server.src.tools.jira.httpx.AsyncClient")
    async def test_jira_search_success(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "issues": [
                {"key": "PLAT-123", "fields": {
                    "summary": "Fix auth bug",
                    "status": {"name": "Open"},
                    "assignee": {"displayName": "Alice"},
                    "priority": {"name": "High"},
                }}
            ],
            "total": 1,
        }
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        mock_client_cls.return_value = mock_client

        result = json.loads(await jira_search(
            jql="project = PLAT", base_url="https://jira.example.com", email="a@b.com", token="tok"
        ))
        assert result["total"] == 1
        assert result["tickets"][0]["key"] == "PLAT-123"

    @pytest.mark.asyncio
    async def test_jira_create_not_configured(self):
        result = json.loads(await jira_create(
            project="PLAT", summary="Test", base_url="", email="", token=""
        ))
        assert result["error"] is True
        assert result["code"] == "JIRA_NOT_CONFIGURED"


class TestGitHubTools:
    @pytest.mark.asyncio
    async def test_github_get_pr_not_configured(self):
        result = json.loads(await github_get_pr(repo="org/repo", pr_number=1, token=""))
        assert result["error"] is True
        assert result["code"] == "GITHUB_NOT_CONFIGURED"

    @pytest.mark.asyncio
    async def test_github_get_file_not_configured(self):
        result = json.loads(await github_get_file(repo="org/repo", path="README.md", token=""))
        assert result["error"] is True
        assert result["code"] == "GITHUB_NOT_CONFIGURED"


class TestPagerDutyTools:
    @pytest.mark.asyncio
    async def test_pagerduty_incidents_not_configured(self):
        result = json.loads(await pagerduty_incidents(api_key=""))
        assert result["error"] is True
        assert result["code"] == "PD_NOT_CONFIGURED"

    @pytest.mark.asyncio
    async def test_pagerduty_ack_not_configured(self):
        result = json.loads(await pagerduty_ack(incident_id="INC123", api_key=""))
        assert result["error"] is True
        assert result["code"] == "PD_NOT_CONFIGURED"
