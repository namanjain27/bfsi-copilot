allowed_keywords = [
    "expense", "income", "budget", "savings", "report", "transaction", 
    "payment", "investment", "account", "finance", "tax", "chair", "website", "app", "table",
    "delivery", "location", "referral"
]
blocked_keywords = [
    "movie", "game", "dating", "joke", "sports", "celebrity", "weather", "travel", "politics"
]

def is_relevant(query):
    query_lower = query.lower()
    
    # Check for blocked content first
    for word in blocked_keywords:
        if word in query_lower:
            return False, "This question is outside the scope of our service."
    
    return True, None
    # # Check for allowed content
    # for word in allowed_keywords:
    #     if word in query_lower:
    #         return True, None
    
    # # Neither explicitly allowed nor blocked
    # return False, "I'm sorry, I can only help with finance-related questions."
