"""Locust load tests for the Knowledge Base Assistant.

Run:
    locust -f tests/load/locustfile.py --host=http://localhost:8080

Target: 10 concurrent sessions (NFR-5).
"""

import json
import random
import uuid

from locust import HttpUser, between, task, tag


SAMPLE_QUESTIONS = [
    "How do I restart the authentication service?",
    "What is the deployment process for the API gateway?",
    "How do I create a new Jira ticket for a production incident?",
    "What database does the user service use?",
    "How do I configure PagerDuty escalation policies?",
    "What are the SLO targets for the platform API?",
    "How do I roll back a failed deployment?",
    "What monitoring tools do we use?",
    "Where are the Kubernetes manifests stored?",
    "How do I add a new environment variable to the CI pipeline?",
]

SAMPLE_TEAMS = ["platform", "backend", "frontend", "devops", "data"]


class ChatUser(HttpUser):
    """Simulates a user interacting with the chat interface."""

    wait_time = between(2, 8)

    def on_start(self):
        self.session_id = str(uuid.uuid4())
        self.team = random.choice(SAMPLE_TEAMS)

    @tag("chat")
    @task(5)
    def send_chat_message(self):
        """Send a chat message and consume the SSE stream."""
        question = random.choice(SAMPLE_QUESTIONS)
        payload = {
            "session_id": self.session_id,
            "message": question,
            "team": self.team,
        }

        with self.client.post(
            "/api/chat",
            json=payload,
            stream=True,
            catch_response=True,
            name="/api/chat",
        ) as response:
            if response.status_code == 200:
                # Consume SSE stream
                content = b""
                for chunk in response.iter_content(chunk_size=1024):
                    content += chunk
                    if b"event: done" in content:
                        break
                response.success()
            else:
                response.failure(f"Status {response.status_code}")

    @tag("health")
    @task(1)
    def check_health(self):
        """Check gateway health."""
        self.client.get("/health", name="/health")


class DocumentUser(HttpUser):
    """Simulates a user managing documents."""

    wait_time = between(5, 15)

    @tag("documents")
    @task(3)
    def list_documents(self):
        """List documents with pagination."""
        team = random.choice(SAMPLE_TEAMS)
        self.client.get(
            f"/api/documents?team={team}&page=1&page_size=20",
            name="/api/documents",
        )

    @tag("documents", "ingest")
    @task(1)
    def ingest_document(self):
        """Upload a sample document for ingestion."""
        sample_content = f"""# Sample Runbook {uuid.uuid4().hex[:6]}

## Overview
This is a sample runbook for load testing the ingestion pipeline.

## Steps
1. Check the service health dashboard
2. Verify the database connections
3. Review recent deployment logs
4. Restart the service if needed

## Contacts
- On-call: platform-oncall@company.com
- Escalation: eng-leads@company.com
"""
        files = {
            "file": (f"runbook_{uuid.uuid4().hex[:6]}.md", sample_content, "text/markdown"),
        }
        data = {
            "team": random.choice(SAMPLE_TEAMS),
            "tags": "runbook,loadtest",
        }

        with self.client.post(
            "/api/documents/ingest",
            files=files,
            data=data,
            catch_response=True,
            name="/api/documents/ingest",
        ) as response:
            if response.status_code in (200, 201):
                response.success()
            else:
                response.failure(f"Status {response.status_code}")


class MixedUser(HttpUser):
    """Simulates a realistic user mixing chat and document operations."""

    wait_time = between(3, 10)

    def on_start(self):
        self.session_id = str(uuid.uuid4())
        self.team = random.choice(SAMPLE_TEAMS)

    @tag("chat")
    @task(6)
    def chat(self):
        """Send a chat message."""
        question = random.choice(SAMPLE_QUESTIONS)
        payload = {
            "session_id": self.session_id,
            "message": question,
            "team": self.team,
        }

        with self.client.post(
            "/api/chat",
            json=payload,
            stream=True,
            catch_response=True,
            name="/api/chat [mixed]",
        ) as response:
            if response.status_code == 200:
                for chunk in response.iter_content(chunk_size=1024):
                    if b"event: done" in chunk:
                        break
                response.success()
            else:
                response.failure(f"Status {response.status_code}")

    @tag("documents")
    @task(2)
    def list_docs(self):
        """List documents."""
        self.client.get(
            f"/api/documents?team={self.team}",
            name="/api/documents [mixed]",
        )

    @tag("health")
    @task(1)
    def health(self):
        self.client.get("/health", name="/health [mixed]")
