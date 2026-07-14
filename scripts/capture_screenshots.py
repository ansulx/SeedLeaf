"""Capture live Streamlit screenshots for the README."""

from __future__ import annotations

import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.microsoft import EdgeChromiumDriverManager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)
BASE = "http://localhost:8501"


def make_driver():
    opts = EdgeOptions()
    opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1440,900")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--hide-scrollbars")
    service = EdgeService(EdgeChromiumDriverManager().install())
    return webdriver.Edge(service=service, options=opts)


def wait_ready(driver, timeout=45):
    WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )
    time.sleep(3.5)  # let Streamlit hydrate


def shot(driver, name: str):
    path = OUT / f"{name}.png"
    driver.save_screenshot(str(path))
    print(f"saved {path}")


def main():
    driver = make_driver()
    try:
        # Home
        driver.get(BASE)
        wait_ready(driver)
        shot(driver, "01_home")

        # Live Diagnosis (Streamlit multipage URL)
        driver.get(f"{BASE}/Live_Diagnosis")
        wait_ready(driver)
        shot(driver, "02_live_diagnosis")

        # Research Results
        driver.get(f"{BASE}/Research_Results")
        wait_ready(driver)
        # scroll a bit for leaderboard
        driver.execute_script("window.scrollTo(0, 120);")
        time.sleep(1.0)
        shot(driver, "03_research_results")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
