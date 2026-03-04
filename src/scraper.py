# %%
from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
import time

start_time = time.time()

website = "https://efdsearch.senate.gov/"                                                                         # The desired website to scrape
browser = webdriver.Chrome()    # Modern Selenium automatically finds the driver
browser.maximize_window()
browser.get(website)

browser.find_element(By.XPATH, "//*[@id='agree_statement']").click()                                               # Completes preliminary check to agree to terms
time.sleep(1)
browser.find_element(By.XPATH,                                                                                    # Set search criteria, in this case we are searching for senate periodic transactions
    "//*[contains(concat( ' ', @class, ' ' ), concat( ' ', 'form-check-input', ' ' ))]").click()
browser.find_element(By.XPATH, "//*[@id='reportTypeLabelPtr']").click()
time.sleep(1)
browser.find_element(By.XPATH, "//*[@id='searchForm']/div/button").click()                                         # Hit the search button

data = {'Name': [], 'Transaction Date': [], 'Owner': [], 'Ticker': [],
        'Asset Name': [], 'Asset Type': [], 'Type': [], 'Amount': [], 'Comment': []}
df = pd.DataFrame(data)

time.sleep(1)
pagenumber = (browser.find_element(By.XPATH, '//*[contains(concat( " ", @class, " " ), concat( " ", "current", " " ))]')).text
pagenumber = int(pagenumber)

while pagenumber <= 1:    # Will stop scraping once page 1 ends.
    print("--- %s seconds ---" % (time.time() - start_time))
    count = 1
    while count <= 25:     # Should be while count <= 25 to go through entire page. 25 is the number of entries per page.
        time.sleep(2)
        caps = str(browser.find_element(By.XPATH, '//*[@id="filedReports"]/tbody/tr[' + str(count) + ']/td[1]').text)
        testcaps = (caps.isupper())
        if testcaps == False:
            button = browser.find_element(By.XPATH, '/html/body/div[1]/main/div/div/div[6]/div/div/div/table/tbody/tr[' + str(count) + ']/td[4]/a')
            browser.execute_script("arguments[0].click();", button)
            time.sleep(2)
            browser.switch_to.window(browser.window_handles[1])
            # Making a loop to sort through each individual report
            individual = 1
            while individual <= int(
                    browser.find_element(By.XPATH, '//*[@id="content"]/div/div/section/div/div/table/tbody/tr[1]/td[1]').text):
                # 36) Finds the number of transactions per report and cycles until individual hits that number.
                df.loc[-1] = {'Name': browser.find_element(By.XPATH, '//*[@id="content"]/div/div/div[2]/div[1]/h2').text,
                              'Transaction Date': browser.find_element(By.XPATH,
                                  '//*[@id="content"]/div/div/section/div/div/table/tbody/tr[' + str(individual) + ']/td[2]').text,
                              'Owner': browser.find_element(By.XPATH,
                                  '//*[@id="content"]/div/div/section/div/div/table/tbody/tr[' + str(
                                      individual) + ']/td[3]').text,
                              'Ticker': browser.find_element(By.XPATH,
                                  '//*[@id="content"]/div/div/section/div/div/table/tbody/tr[' + str(
                                      individual) + ']/td[4]').text,
                              'Asset Name': browser.find_element(By.XPATH,
                                  '//*[@id="content"]/div/div/section/div/div/table/tbody/tr[' + str(
                                      individual) + ']/td[5]').text,
                              'Asset Type': browser.find_element(By.XPATH,
                                  '//*[@id="content"]/div/div/section/div/div/table/tbody/tr[' + str(
                                      individual) + ']/td[6]').text,
                              'Type': browser.find_element(By.XPATH,
                                  '//*[@id="content"]/div/div/section/div/div/table/tbody/tr[' + str(
                                      individual) + ']/td[7]').text,
                              'Amount': browser.find_element(By.XPATH,
                                  '//*[@id="content"]/div/div/section/div/div/table/tbody/tr[' + str(
                                      individual) + ']/td[8]').text,
                              'Comment': browser.find_element(By.XPATH,
                                  '//*[@id="content"]/div/div/section/div/div/table/tbody/tr[' + str(
                                      individual) + ']/td[9]').text
                              }
                individual = individual + 1
                df.index = df.index + 1
                df = df.sort_index()
            browser.close()
            browser.switch_to.window(browser.window_handles[0])
            # Print the output.
            print(df)
            count = count + 1
        else:
            count = count + 1
    pagenumber = pagenumber+1

print("--- %s seconds ---" % (time.time() - start_time))
df.to_csv('CongressInvestments.csv', index=False)