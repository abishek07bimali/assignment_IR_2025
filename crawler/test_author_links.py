#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
import time

def build_chrome_options():
    opts = Options()
    opts.add_argument("--window-size=1366,900")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--lang=en-US")
    opts.add_argument("--disable-notifications")
    opts.add_argument("--no-first-run")
    opts.add_argument("--no-default-browser-check")
    opts.add_argument("--disable-extensions")
    opts.add_argument("--disable-popup-blocking")
    opts.add_argument("--disable-renderer-backgrounding")
    opts.add_argument("--disable-backgrounding-occluded-windows")
    opts.add_argument("--disable-features=CalculateNativeWinOcclusion,MojoVideoDecoder")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-logging", "enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.page_load_strategy = "eager"
    opts.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
    )
    return opts

def accept_cookies_if_present(driver):
    try:
        btn = WebDriverWait(driver, 6).until(
            EC.presence_of_element_located((By.ID, "onetrust-accept-btn-handler"))
        )
        driver.execute_script("arguments[0].click();", btn)
        time.sleep(0.25)
    except:
        pass

# Test with the provided publication URL
test_url = "https://pureportal.coventry.ac.uk/en/publications/a-qard-hassan-benevolent-loan-crowdfunding-model-for-refugee-fina"

service = ChromeService(ChromeDriverManager().install(), log_output=os.devnull)
driver = webdriver.Chrome(service=service, options=build_chrome_options())
driver.set_page_load_timeout(45)

try:
    print(f"Testing URL: {test_url}")
    driver.get(test_url)
    accept_cookies_if_present(driver)
    
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, "h1")))
    
    print("\n=== Looking for author links ===")
    
    # First, try to expand any hidden author sections
    try:
        btns = driver.find_elements(
            By.XPATH,
            "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'show') or "
            "contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'more')]"
        )
        for b in btns[:2]:
            try:
                driver.execute_script("arguments[0].scrollIntoView({block:'center'});", b)
                time.sleep(0.15)
                b.click()
                time.sleep(0.25)
                print(f"Clicked expand button: {b.text}")
            except:
                continue
    except:
        pass
    
    # Try different selectors to find author links
    selectors = [
        ".relations.persons a[href*='/en/persons/']",
        "section#persons a[href*='/en/persons/']",
        ".person-relation-list a[href*='/en/persons/']",
        ".persons-container a[href*='/en/persons/']",
        "div.persons a[href*='/en/persons/']"
    ]
    
    author_info = []
    found_selector = None
    for sel in selectors:
        elements = driver.find_elements(By.CSS_SELECTOR, sel)
        if elements:
            print(f"\nFound {len(elements)} element(s) with selector: {sel}")
            for el in elements:
                name_el = el.find_element(By.CSS_SELECTOR, "span") if el.find_elements(By.CSS_SELECTOR, "span") else el
                name = name_el.text.strip()
                link = el.get_attribute("href")
                # Filter out generic links
                if name and link and "/en/persons/" in link and link != "https://pureportal.coventry.ac.uk/en/persons/":
                    author_info.append({"name": name, "profile_link": link})
                    print(f"  - {name}: {link}")
                    found_selector = sel
            if author_info:
                break
    
    # If still no author links found, try broader search
    if not author_info:
        print("\nTrying broader search for author links...")
        all_links = driver.find_elements(By.CSS_SELECTOR, "a[href*='/en/persons/']")
        for link in all_links:
            href = link.get_attribute("href") or ""
            text = link.text.strip()
            # Filter out generic /en/persons/ link
            if text and href != "https://pureportal.coventry.ac.uk/en/persons/" and "/en/persons/" in href:
                # Check if it's actually a person's profile (has more path after /en/persons/)
                if len(href.split("/en/persons/")[1]) > 1:
                    author_info.append({"name": text, "profile_link": href})
                    print(f"Found author link: {text} -> {href}")
    
    if not author_info:
        print("\nNo author links found with standard selectors. Examining page structure...")
        
        # Look for any links that might be author profiles
        all_links = driver.find_elements(By.TAG_NAME, "a")
        for link in all_links:
            href = link.get_attribute("href") or ""
            text = link.text.strip()
            if "/en/persons/" in href and text:
                author_info.append({"name": text, "profile_link": href})
                print(f"Found author link: {text} -> {href}")
    
    print(f"\n=== Summary ===")
    print(f"Total author profiles found: {len(author_info)}")
    for info in author_info:
        print(f"  {info['name']}: {info['profile_link']}")
    
finally:
    driver.quit()