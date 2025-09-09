import requests
import json
import os

def get_cik(ticker):
    """
    Retrieves the CIK for a given ticker symbol from the SEC's AutilPI.
    """
    url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=&dateb=&owner=exclude&start=0&count=1&output=atom"
    headers = {'User-Agent': "wildandzaky4@gmail.com"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        # Extract CIK from the response
        try:
            cik = response.text.split('<cik>')[1].split('</cik>')[0]
            return cik.zfill(10)  # Pad with leading zeros to ensure 10 digits
        except IndexError:
            print("Could not parse CIK from SEC response.")
            return None
    else:
        print(f"Failed to retrieve data for ticker {ticker}. Status code: {response.status_code}")
        return None

def get_fundamental_data(ticker):
    """
    Retrieves fundamental data for a given ticker symbol from the SEC's AutilPI.
    """
    # Main part of the script
    ticker = ticker  # Example ticker, replace with desired ticker
    cik = get_cik(ticker)

    if cik:
        print(f"CIK for {ticker}: {cik}")
        # Construct the URL
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

        # Make the request
        headers = {'User-Agent': "wildandzaky4@gmail.com"}  # Required by SEC
        response = requests.get(url, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()

            # Create directory if it doesn't exist
            output_dir = "/root/vynixmodelling/dataset/sec_data"
            os.makedirs(output_dir, exist_ok=True)

            # Save the JSON data to a file
            filename = os.path.join(output_dir, f"{ticker}.json")
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)  # indent for pretty printing

            print(f"Data saved to {filename}")
            return data
        else:
            print(f"Failed to retrieve data. Status code: {response.status_code}")
            print(response.text)  # Print the response text for debugging
            return None
    else:
        print(f"Could not retrieve CIK for ticker {ticker}.")
        return None