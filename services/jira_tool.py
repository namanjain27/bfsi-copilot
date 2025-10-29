from jira import JIRA
import os
from dotenv import load_dotenv
from typing import List, Optional

class JiraTool:
    _instance: Optional['JiraTool'] = None
    _jira_client: Optional[JIRA] = None
    _project_key: Optional[str] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize JIRA client connection"""
        load_dotenv()
        host = os.environ.get("JIRA_URL")
        api_token = os.environ.get("JIRA_API_TOKEN")
        self._project_key = os.environ.get("JIRA_PROJECT_KEY")
        email = os.environ.get("JIRA_EMAIL")

        self._jira_client = JIRA(
            server=host,
            basic_auth=(email, api_token)
        )
        
        server_info = self._jira_client.server_info()
        print("Connected to Jira: ", server_info['baseUrl'])

    def create_ticket(self, summary: str, description: str, intent: str, urgency: str, sentiment: str) -> str:
        """Creates a jira ticket for service request, complaints and feature request.
            Args: 
                summary: this would be the subject of the ticket. Keep it short and self-defining
                description: description should contain all the necessary details to help the associates resolve the issue
                intent (one of): service_request, complaints or feature_request
                urgency (one of): high (if told critical or urgent), medium (default for complaint and service_request), low (else)
                sentiment (one of): positive (on receiving good remark), neutral (default), negative (if user expresses bad experience)
            returns: ticket id as string if success. None if failed to create ticket.
        """
        if not self._jira_client or not self._project_key:
            return "Jira client not initialized properly"
        
        issue_data = {
            "project": {"key": self._project_key},
            "summary": summary,
            "description": description,
            "issuetype": {"name": "Story"},
            "labels": [intent, urgency, sentiment]
        }
        try:
            new_issue = self._jira_client.create_issue(fields=issue_data)
            print("Issue created successfully! with issue key:", new_issue.key)
            return new_issue.key
        except Exception as e:
            print("Failed to create issue:", str(e))
            return f"Failed to create issue: {str(e)}"

