# Goal
a multi agent workflow that follows a well-defined processing flow. We will use langgraph. This workflow is designed to facilitate claim verification particulary for BFSI sector. 

- initial commits are made in [EchoPilot repository](https://github.com/namanjain27/EchoPilot).

# Agents and their duties: 
1. intent classifier and information gatherer agent:
finds the intent of the human query from among 'query', 'complaint', 'service request' and 'feature request'. Fetches relevant data by using RAG tool if required. It should assume facts or policies and answer only the data from knowledge base using RAG tool. responses the answer if simple query, else make request to the report_maker_agent.

2. report maker agent:
It arranges the input information as issue, user-demand, company_docs_about_issue and support_info_from_user. It tells crisp details for all the entries. Do not assume any information and do not use its self training data. Use only the information provided as input. 

3. claim verifier agent:
it has tool access for: RAG, fetch user-data, make entry into incident DB table. 
It verifies claim and checks the best resolution for the issue. It prioritizes the company's policies for resolution in such cases than user-demand. It receives information from report-maker-agent. It checks, is the user-issue valid. It may need to fetch user-data from the by making a tool call. if invalid then give proper reasoning and ask user if unsatisfied then a complaint ticket can be generated. Else if issue is valid and the agent is confident to take the decision and there are appropriate action tools available to the agent for issue resolution then execute it. A ticket is created for tracking purposes and logging of the AI action made. Else if issue is valid and the agent is not confident then create a ticket as a fallback for a human member to look into and share the details with user.      

# Tool call/Actions to make:
1. RAG tool
2. fetch user data from DB
3. make DB entry for incident
4. create jira ticket
5. take action. example: calling the refund API

# Tool Calling Workflow Diagram

## Complete Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER QUERY                               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │  Intent Gatherer      │
                │  - Classify intent    │
                │  - Extract aspects    │
                │  - Retrieve KB docs   │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │  Report Maker         │
                │  - Structure info     │
                │  - Extract issue      │
                │  - Format report      │
                └───────────┬───────────┘
                            │
                            ▼
        ┌───────────────────────────────────────────┐
        │      Claim Verifier (First Pass)          │
        │  - Review report & intent                 │
        │  - Check if more info needed              │
        └───────────────┬───────────────────────────┘
                        │
                        │ Decision Point
                        ▼
        ┌───────────────┴──────────────────┐
        │                                   │
        ▼                                   ▼
┌───────────────┐              ┌────────────────────┐
│ Sufficient    │              │ Need More Info     │
│ Information   │              │ Make Tool Calls    │
└───────┬───────┘              └─────────┬──────────┘
        │                                 │
        │                                 ▼
        │                      ┌────────────────────┐
        │                      │ Tool Execution     │
        │                      │ - retriever_tool   │
        │                      │ - get_user_data    │
        │                      │ - list_policies    │
        │                      │ - create_ticket    │
        │                      └─────────┬──────────┘
        │                                │
        │                                ▼
        │              ┌─────────────────────────────────┐
        │              │  Claim Verifier (Second Pass)   │
        │              │  - Receives tool results        │
        │              │  - Reviews all information      │
        │              └─────────────────┬───────────────┘
        │                                │
        │◄───────────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  Final Decision          │
│  - is_valid (Yes/No)     │
│  - confidence (0-1)      │
│  - resolution text       │
│  - action_plan           │
│    • create_ticket       │
│    • make_db_entry       │
│    • call_refund_api     │
└────────────┬─────────────┘
             │
             ▼
        ┌────────┐
        │  END   │
        └────────┘
```

## Tool Calling Loop Detail

```
                    ┌─────────────────────────────┐
                    │   Claim Verifier Node       │
                    │                             │
    ┌───────────────┤  1. Check if ToolMessage    │
    │               │     in messages             │
    │               │                             │
    │  NO           │  2. If NO tool results yet: │
    │  (First Pass) │     _check_and_call_tools() │
    │               │                             │
    │               │  3. If YES tool results:    │
    │  YES          │     _make_final_decision()  │
    │  (After Tools)│                             │
    │               └──────────┬──────────────────┘
    │                          │
    │                          │
    │                    Check response
    │                          │
    │              ┌───────────┴──────────┐
    │              │                      │
    │              ▼                      ▼
    │     ┌────────────────┐    ┌──────────────────┐
    │     │  Has tool_calls│    │ No tool_calls    │
    │     │  in response   │    │ (Decision ready) │
    │     └────────┬───────┘    └────────┬─────────┘
    │              │                     │
    │              ▼                     │
    │     ┌────────────────┐            │
    │     │ Tool Execution │            │
    │     │ - Invoke tools │            │
    │     │ - Create       │            │
    │     │   ToolMessages │            │
    │     └────────┬───────┘            │
    │              │                    │
    └──────────────┘                    │
         Loop back                      │
         to Verifier                    ▼
                                   ┌─────────┐
                                   │   END   │
                                   └─────────┘
```

## State Flow

```
Initial State:
├─ messages: [HumanMessage]
├─ tenant_id: "default"
├─ user_id: "U001"
├─ email: "user@example.com"
└─ user_role: "customer"

After Intent:
└─ + intent_result: {intent, urgency, sentiment, aspects}
└─ + kb_docs: [KBDocument, ...]

After Report:
└─ + report: {issue, user_demand, company_docs, support_info, policy_refs}

After Verifier (First Pass - if tools needed):
└─ + messages: [..., AIMessage(tool_calls=[...])]

After Tool Execution:
└─ + messages: [..., AIMessage(tool_calls), ToolMessage, ToolMessage, ...]

After Verifier (Second Pass):
└─ + verification: {is_valid, resolution, confidence, policy_citations, action_plan}

Final State → END
```

## Tool Decision Logic in Claim Verifier

```python
def process(state):
    has_tool_results = any(msg is ToolMessage for msg in messages)
    
    if has_tool_results:
        # Second+ pass: Tool results available
        decision = _make_final_decision(report, intent, messages)
        return {'verification': decision}
    
    else:
        # First pass: Check if tools needed
        response = _check_and_call_tools(report, intent, user_id, email, messages)
        
        if response.tool_calls:
            # LLM wants to call tools
            return {'messages': [response]}
        else:
            # LLM has enough info
            decision = _make_final_decision(report, intent, messages)
            return {'verification': decision}
```

## Example Tool Call Sequence

```
1. User: "I want a refund for cancelled policy"

2. Verifier (First Pass):
   → Checks report (has general policy info from KB)
   → Decides: Need user's policy details
   → Makes tool_calls: [get_user_data(user_id), list_user_policies(user_id)]

3. Tool Execution Node:
   → Executes get_user_data → Returns user info
   → Executes list_user_policies → Returns policy list
   → Adds ToolMessages to state

4. Verifier (Second Pass):
   → Sees ToolMessages in state
   → Reviews: report + tool results
   → Has all info needed
   → Makes final decision with structured output

5. END → verification decision ready for effects layer
```

## Key Integration Points

### multi_agent_graph.py
- `get_all_tools(tenant_id, user_role)` → Returns 4 tools
- `tool_execution_node(state)` → Executes tool calls
- `route_after_verify(state)` → Routes based on tool_calls presence

### claim_verifier.py
- `_check_and_call_tools()` → Uses llm_with_tools
- `_make_final_decision()` → Uses structured_llm (no tools)
- `_format_decision_context()` → Includes tool results

### tools_service.py
- `retriever_tool` → KB search
- `create_jira_ticket` → Ticket creation
- `get_user_data` → DB user lookup
- `list_user_policies` → DB policy lookup

