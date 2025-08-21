#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os

def build_chrome_options():
    opts = Options()
    # opts.add_argument("--headless=new")  # Disabled for testing
    opts.add_argument("--window-size=1366,900")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-logging", "enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.page_load_strategy = "normal"  # Changed to normal
    opts.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
    )
    return opts

def extract_authors_with_profiles(driver):
    """Extract authors and their profile links"""
    authors = []
    author_profiles = {}
    
    # Try to expand any hidden author sections
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
            except:
                continue
    except:
        pass
    
    # Try different selectors to find author links
    selectors = [
        ".relations.persons a[href*='/en/persons/']",
        "section#persons a[href*='/en/persons/']",
        ".person-relation-list a[href*='/en/persons/']",
        "div.persons a[href*='/en/persons/']",
    ]
    
    for sel in selectors:
        elements = driver.find_elements(By.CSS_SELECTOR, sel)
        for el in elements:
            # Get the text (might be in a span inside the link)
            name_el = el.find_element(By.CSS_SELECTOR, "span") if el.find_elements(By.CSS_SELECTOR, "span") else el
            name = name_el.text.strip()
            profile_link = el.get_attribute("href")
            
            # Filter out generic /en/persons/ link
            if name and profile_link and "/en/persons/" in profile_link and profile_link != "https://pureportal.coventry.ac.uk/en/persons/":
                authors.append(name)
                # Check if it's actually a person's profile (has more path after /en/persons/)
                if len(profile_link.split("/en/persons/")[1]) > 1:
                    author_profiles[name] = profile_link
        if authors:
            break
    
    return authors, author_profiles

# Test URL
test_url = "https://pureportal.coventry.ac.uk/en/publications/a-qard-hassan-benevolent-loan-crowdfunding-model-for-refugee-fina"

service = ChromeService(ChromeDriverManager().install(), log_output=os.devnull)
driver = webdriver.Chrome(service=service, options=build_chrome_options())

try:
    print(f"Testing: {test_url}")
    driver.get(test_url)
    
    # Accept cookies if present
    try:
        btn = WebDriverWait(driver, 6).until(
            EC.presence_of_element_located((By.ID, "onetrust-accept-btn-handler"))
        )
        driver.execute_script("arguments[0].click();", btn)
        time.sleep(0.5)
        print("Accepted cookies")
    except:
        pass
    
    # Wait for page to load
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, "h1")))
    
    # Extract title
    title = driver.find_element(By.CSS_SELECTOR, "h1").text.strip()
    
    # Extract authors with profiles
    authors, author_profiles = extract_authors_with_profiles(driver)
    
    # If no authors found with links, try other methods
    if not authors:
        # Try from meta tags
        for meta in driver.find_elements(By.CSS_SELECTOR, 'meta[name="citation_author"], meta[property="citation_author"]'):
            content = meta.get_attribute("content")
            if content:
                authors.append(content.strip())
    
    # Extract date
    published_date = None
    try:
        date_el = driver.find_element(By.CSS_SELECTOR, "span.date")
        published_date = date_el.text.strip()
    except:
        pass
    
    # Extract abstract
    abstract = ""
    for sel in ["section#abstract .textblock", "div.abstract .textblock", "div.textblock"]:
        try:
            el = driver.find_element(By.CSS_SELECTOR, sel)
            abstract = el.text.strip()
            if abstract and len(abstract) > 15:
                break
        except:
            continue
    
    # Print results
    result = {
        "title": title,
        "link": test_url,
        "authors": authors,
        "author_profiles": author_profiles,
        "published_date": published_date,
        "abstract": abstract[:200] + "..." if len(abstract) > 200 else abstract
    }
    
    print("\nExtracted data:")
    print(json.dumps(result, indent=2))
    
finally:
    driver.quit()