import os

# Chat files paths
CHAT_SUMMARY_FILE = "chat_summary.txt"

def load_chat_summary():
    """Load chat summary from file if it exists."""
    chat_summary = ""
    if os.path.exists(CHAT_SUMMARY_FILE):
        try:
            with open(CHAT_SUMMARY_FILE, 'r', encoding='utf-8') as f:
                chat_summary = f.read()
        except Exception as e:
            print(f"Error loading chat summary: {e}")
            chat_summary = ""
    return chat_summary


def save_chat_summary(chat_summary):
    """Save chat summary to file."""
    try:
        with open(CHAT_SUMMARY_FILE, 'w', encoding='utf-8') as f:
            f.write(chat_summary)
        print(f"Chat summary saved to {CHAT_SUMMARY_FILE}")
    except Exception as e:
        print(f"Error saving chat summary: {e}")
