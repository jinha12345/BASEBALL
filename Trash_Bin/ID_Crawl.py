from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import re
import time
import pickle

def Id_Crawl(year):
    # Setup Chrome options
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # Run in headless mode (commented out for debugging)
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")

    # Initialize the Chrome driver
    driver = webdriver.Chrome(options=chrome_options)

    try:
        # Navigate to the KBO schedule page
        url = "https://www.koreabaseball.com/Schedule/Schedule.aspx"
        driver.get(url)
        
        print(f"Page title: {driver.title}")
        
        # Give it more time to load and bypass potential bot detection
        time.sleep(5)
        
        # Function to extract gameIDs from the current page
        def extract_game_ids():
            game_ids = []
            
            try:
                # Try different selectors to find game links
                selectors = [
                    "a[href*='GameCenter/Main.aspx']",
                    "a.btnGameCenter",
                    "a.btnReview",
                    "a[href*='gameId=']"
                ]
                
                for selector in selectors:
                    try:
                        links = driver.find_elements(By.CSS_SELECTOR, selector)
                        if links:
                            print(f"Found {len(links)} links with selector: {selector}")
                            break
                    except Exception as e:
                        print(f"Error with selector {selector}: {str(e)}")
                
                # If no predefined selector worked, dump the page structure
                if not links:
                    print("Could not find game links with predefined selectors.")
                    print("Page source excerpt:")
                    print(driver.page_source[:1000])  # Print first 1000 chars to see structure
                    
                    # Take a screenshot for debugging
                    driver.save_screenshot("kbo_page.png")
                    print("Screenshot saved as kbo_page.png")
                    return []
                
                for link in links:
                    href = link.get_attribute('href')
                    if href:
                        # Extract gameID using regex
                        match = re.search(r'gameId=([^&]+)', href)
                        if match:
                            game_id = match.group(1)
                            game_ids.append(game_id)
            
            except Exception as e:
                print(f"Error extracting game IDs: {str(e)}")
            
            return game_ids

        # Try to select a specific year and month first
        try:
            # Try to interact with the date selector if it exists
            #year = "2025"
            month = "03"  # Start with March
            
            try:
                WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.ID, "ddlYear"))
                )
                year_dropdown = driver.find_element(By.ID, "ddlYear")
                year_dropdown.click()
                time.sleep(1)
                year_option = driver.find_element(By.XPATH, f"//select[@id='ddlYear']/option[@value='{year}']")
                year_option.click()
                
                month_dropdown = driver.find_element(By.ID, "ddlMonth")
                month_dropdown.click()
                time.sleep(1)
                month_option = driver.find_element(By.XPATH, f"//select[@id='ddlMonth']/option[@value='{month}']")
                month_option.click()
                
                # Wait for the page to update
                time.sleep(3)
                
                print(f"Successfully set date to {year}-{month}")
            except Exception as e:
                print(f"Error setting date: {str(e)}")
        
        except Exception as e:
            print(f"Date selection failed: {str(e)}")
        
        # Initialize a set to store all gameIDs (using a set to avoid duplicates)
        all_game_ids = set()
        
        # Get the current page's gameIDs
        current_page_game_ids = extract_game_ids()
        all_game_ids.update(current_page_game_ids)
        print(f"Found {len(current_page_game_ids)} games on the current page")
        
        # If we have a date selector, try other months
        months_to_check = ["03", "04", "05", "06", "07", "08", "09", "10"]
        #year = "2025"  # Change this to the desired year
        
        date_selector_works = False
        
        try:
            # Test if date selector works
            test_dropdown = driver.find_element(By.ID, "ddlYear")
            date_selector_works = True
        except NoSuchElementException:
            print("Date selector not found. Will only process the current page.")
        
        if date_selector_works:
            for month in months_to_check:
                try:
                    # Select the year dropdown
                    year_dropdown = driver.find_element(By.ID, "ddlYear")
                    year_dropdown.click()
                    time.sleep(1)
                    year_option = driver.find_element(By.XPATH, f"//select[@id='ddlYear']/option[@value='{year}']")
                    year_option.click()
                    
                    # Select the month dropdown
                    month_dropdown = driver.find_element(By.ID, "ddlMonth")
                    month_dropdown.click()
                    time.sleep(1)
                    month_option = driver.find_element(By.XPATH, f"//select[@id='ddlMonth']/option[@value='{month}']")
                    month_option.click()
                    
                    # Wait for the page to update
                    time.sleep(3)
                    
                    # Extract gameIDs from this month
                    month_game_ids = extract_game_ids()
                    all_game_ids.update(month_game_ids)
                    print(f"Found {len(month_game_ids)} games for {year}-{month}")
                
                except Exception as e:
                    print(f"Error processing {year}-{month}: {str(e)}")
        
        # Print the total number of unique gameIDs found
        print(f"\nTotal unique gameIDs found: {len(all_game_ids)}")
        print("\nGameIDs:")
        for game_id in sorted(all_game_ids):
            print(game_id)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    finally:
        # Take a final screenshot before closing
        try:
            driver.save_screenshot("final_state.png")
            print("Final screenshot saved as final_state.png")
        except:
            pass
        
        # Close the browser
        driver.quit()

    with open('./GAME_IDS/' + year + ".pickle","wb") as f:
        pickle.dump(all_game_ids, f)