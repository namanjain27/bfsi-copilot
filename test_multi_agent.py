"""
Test script for multi-agent graph
Quick validation of the orchestration flow
"""

from multi_agent_graph import invoke_graph
from services.logger_setup import setup_logger
logger = setup_logger()


def test_simple_query():
    """Test simple query flow (Intent -> Answer)"""
    print("\n" + "="*60)
    print("TEST 1: Simple Query")
    print("="*60)
    
    result = invoke_graph(
        query="What is your refund policy?",
        tenant_id="rentomojo",
        user_role="customer"
    )
    
    print("\n--- Final State ---")
    print(f"Intent: {result.get('intent_result', {}).get('intent')}")
    print(f"Final Answer: {result.get('final_answer', 'N/A')[:200]}...")


def test_complaint_flow():
    """Test complaint flow (Intent -> Report -> Verify)"""
    print("\n" + "="*60)
    print("TEST 2: Complaint/Claim Flow (BFSI)")
    print("="*60)
    
    result = invoke_graph(
        query="I want to file a complaint. My insurance claim was rejected but I have all the necessary documents.",
        tenant_id="rentomojo",
        user_role="customer",
        user_id="U001",
        email="user001@example.com"
    )
    
    print("\n--- Final State ---")
    print(f"Intent: {result.get('intent_result', {}).get('intent')}")
    print(f"Out of Scope: {result.get('intent_result', {}).get('out_of_scope')}")
    print(f"Report Created: {result.get('report') is not None}")
    if result.get('report'):
        print(f"  Issue: {result['report'].get('issue', 'N/A')[:100]}...")
        print(f"  Company Docs: {len(result['report'].get('company_docs_about_issue', ''))} chars")
    print(f"Verification: {result.get('verification') is not None}")
    if result.get('verification'):
        print(f"  Valid: {result['verification'].get('is_valid')}")
        print(f"  Confidence: {result['verification'].get('confidence')}")
        print(f"  Resolution: {result['verification'].get('resolution', 'N/A')[:150]}...")
        action_plan = result['verification'].get('action_plan', {})
        if action_plan:
            print(f"  Create Ticket: {action_plan.get('create_ticket')}")
            print(f"  Ticket Type: {action_plan.get('ticket_type')}")
    
    # Check for tool calls in messages
    messages = result.get('messages', [])
    tool_messages = [m for m in messages if m.__class__.__name__ == 'ToolMessage']
    print(f"\nTool Calls Made: {len(tool_messages)}")
    for i, tm in enumerate(tool_messages, 1):
        print(f"  Tool {i}: {getattr(tm, 'name', 'Unknown')}")

def test_claim_with_tool_calls():
    """Test claim verification with tool calls for user data"""
    print("\n" + "="*60)
    print("TEST 3: Claim with Tool Calls")
    print("="*60)
    
    result = invoke_graph(
        query="I paid my insurance premium but my policy was cancelled. I need a refund immediately.",
        tenant_id="rentomojo",
        user_role="customer",
        user_id="U001",
        email="john.doe@example.com"
    )
    
    print("\n--- Final State ---")
    print(f"Intent: {result.get('intent_result', {}).get('intent')}")
    print(f"Report Created: {result.get('report') is not None}")
    print(f"Verification: {result.get('verification') is not None}")
    
    if result.get('verification'):
        print(f"  Valid: {result['verification'].get('is_valid')}")
        print(f"  Confidence: {result['verification'].get('confidence'):.2f}")
        print(f"  Resolution: {result['verification'].get('resolution', 'N/A')[:200]}...")
    
    # Check for tool calls
    messages = result.get('messages', [])
    tool_messages = [m for m in messages if m.__class__.__name__ == 'ToolMessage']
    print(f"\nTotal Tool Calls: {len(tool_messages)}")
    for i, tm in enumerate(tool_messages, 1):
        tool_name = getattr(tm, 'name', 'Unknown')
        content_preview = str(tm.content)[:100]
        print(f"  Tool {i}: {tool_name}")
        print(f"    Result: {content_preview}...")

def test_out_of_scope():
    """Test out-of-scope detection for non-BFSI queries"""
    print("\n" + "="*60)
    print("TEST 4: Out of Scope Detection")
    print("="*60)
    
    result = invoke_graph(
        query="I rented a chair but it got faulty just after 1 week. I want to exchange it.",
        tenant_id="rentomojo",
        user_role="customer",
        user_id="U001"
    )
    
    print("\n--- Final State ---")
    print(f"Intent: {result.get('intent_result', {}).get('intent')}")
    print(f"Out of Scope: {result.get('intent_result', {}).get('out_of_scope')}")
    print(f"Final Answer: {result.get('final_answer', 'N/A')[:150]}...")


if __name__ == "__main__":
    print("\nMulti-Agent Graph Test Suite")
    print("="*60)
    
    # try:
    #     test_simple_query()
    # except Exception as e:
    #     logger.error(f"Test 1 failed: {e}", exc_info=True)
    
    try:
        test_complaint_flow()
    except Exception as e:
        logger.error(f"Test 2 failed: {e}", exc_info=True)
    
    try:
        test_claim_with_tool_calls()
    except Exception as e:
        logger.error(f"Test 3 failed: {e}", exc_info=True)
    
    # try:
    #     test_out_of_scope()
    # except Exception as e:
    #     logger.error(f"Test 4 failed: {e}", exc_info=True)
    
    print("\n" + "="*60)
    print("Tests completed!")
    print("="*60)

