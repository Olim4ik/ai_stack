"""Sample evaluation dataset for RAG quality testing."""

from .metrics import EvalSample, RetrievalResult

# Sample Q&A pairs for evaluation
EVAL_DATASET: list[EvalSample] = [
    EvalSample(
        question="How do I restart the authentication service?",
        expected_answer="SSH into the auth server and run `systemctl restart auth-service`. Check logs with `journalctl -u auth-service -f`.",
        relevant_doc_ids=["doc_runbook_auth", "doc_runbook_services"],
        tags=["runbook", "auth"],
    ),
    EvalSample(
        question="What is the deployment process for the API gateway?",
        expected_answer="The API gateway is deployed via GitHub Actions CI/CD pipeline. Merge to main triggers a build, push to ECR, and ECS deployment.",
        relevant_doc_ids=["doc_deploy_gateway", "doc_cicd_pipeline"],
        tags=["deployment", "gateway"],
    ),
    EvalSample(
        question="How do I create a new Jira ticket for a production incident?",
        expected_answer="Use the /incident slash command in Slack or create a ticket in the PROD project with type 'Incident' and assign it to the on-call team.",
        relevant_doc_ids=["doc_incident_process", "doc_jira_guide"],
        tags=["incident", "jira"],
    ),
    EvalSample(
        question="What database does the user service use?",
        expected_answer="The user service uses PostgreSQL 15 on RDS with read replicas for high availability.",
        relevant_doc_ids=["doc_user_service_arch", "doc_database_guide"],
        tags=["architecture", "database"],
    ),
    EvalSample(
        question="How do I configure PagerDuty escalation policies?",
        expected_answer="Navigate to PagerDuty > Services > Escalation Policies. Create a new policy with L1 (on-call) and L2 (team lead) escalation levels with 15-minute timeout.",
        relevant_doc_ids=["doc_pagerduty_setup", "doc_oncall_guide"],
        tags=["pagerduty", "oncall"],
    ),
    EvalSample(
        question="What are the SLO targets for the platform API?",
        expected_answer="Platform API SLOs: 99.9% availability, p99 latency < 500ms, error rate < 0.1%.",
        relevant_doc_ids=["doc_slo_targets", "doc_platform_api"],
        tags=["slo", "platform"],
    ),
    EvalSample(
        question="How do I roll back a failed deployment?",
        expected_answer="Run `./scripts/rollback.sh <service-name> <previous-version>` or revert the merge commit in GitHub and re-deploy.",
        relevant_doc_ids=["doc_rollback_procedure", "doc_deploy_gateway"],
        tags=["deployment", "rollback"],
    ),
    EvalSample(
        question="What monitoring tools do we use?",
        expected_answer="We use Datadog for metrics and APM, PagerDuty for alerting, and Grafana dashboards for visualization.",
        relevant_doc_ids=["doc_monitoring_stack", "doc_observability"],
        tags=["monitoring", "observability"],
    ),
]
