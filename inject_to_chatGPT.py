# inject_to_chatgpt.py
import sys
from playwright.sync_api import sync_playwright
import time

def inject_to_chatgpt(message):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(storage_state="chatgpt_state.json")
        page = context.new_page()

        page.goto("https://chat.openai.com/chat")
        page.wait_for_selector("textarea", timeout=15000)
        page.fill("textarea", message)
        time.sleep(0.5)
        page.keyboard.press("Enter")
        time.sleep(5)

        context.storage_state(path="chatgpt_state.json")
        browser.close()

if __name__ == "__main__":
    msg = sys.argv[1] if len(sys.argv) > 1 else "[Error: No message provided]"
    inject_to_chatgpt(msg)

