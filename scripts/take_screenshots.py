"""
scripts/take_screenshots.py
----------------------------
Takes screenshots of each of the 4 tabs of the running Streamlit dashboard
and saves them to docs/screenshots/.
"""
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT_DIR = Path("docs/screenshots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    
    print("Navigating to http://localhost:8501...")
    page.goto("http://localhost:8501", timeout=60000)
    page.wait_for_load_state("networkidle", timeout=30000)
    time.sleep(3)  # allow charts to render
    
    # Tab 1: Simulator
    print("Capturing Tab 1...")
    page.screenshot(path=str(OUT_DIR / "tab1_scenario_simulator.png"), full_page=True)
    
    # Wait for tabs to render
    page.wait_for_selector("[role='tab']", timeout=15000)
    tab_buttons = page.locator("[role='tab']")
    count = tab_buttons.count()
    print(f"Found {count} tabs.")
    
    tab_names = [
        "tab1_scenario_simulator",
        "tab2_historical_replay",
        "tab3_model_insights",
        "tab4_about_limitations"
    ]
    
    for i in range(count):
        print(f"Clicking Tab {i+1} ({tab_names[i]})...")
        tab_buttons.nth(i).click()
        time.sleep(3.0)  # wait for tab content / charts to mount
        page.screenshot(path=str(OUT_DIR / f"{tab_names[i]}.png"), full_page=True)
        print(f"Captured {tab_names[i]}.png")
        
    browser.close()
    print("All screenshots successfully captured!")
