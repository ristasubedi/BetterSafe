import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import re
import json
import requests
from datetime import datetime, timedelta, timezone

def get_sitemap_urls(sitemap_urls, initial_pull = False):
    urls_to_update = []
    for sitemap_url in sitemap_urls:
        try:
            response = requests.get(sitemap_url)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            for url_entry in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}url"):
                url = url_entry.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc").text.strip()
                lastmod_element = url_entry.find("{http://www.sitemaps.org/schemas/sitemap/0.9}lastmod")
                lastmod = lastmod_element.text.strip() if lastmod_element is not None else None
                if lastmod:
                    lastmod_datetime = datetime.strptime(lastmod, "%Y-%m-%dT%H:%M:%S%z")
                    if not initial_pull:
                        if (datetime.now(timezone.utc) - lastmod_datetime)  <= timedelta(days=1):
                            urls_to_update.append(url)
                    else:
                        urls_to_update.append(url)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching sitemap: {e}")
    return urls_to_update

def fetch_site_data(urls_to_update, file_path):
    content = {}
    for itemNum, url in enumerate(urls_to_update):
        print("On item: " + str(itemNum) + " of " + str(len(urls_to_update)))
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            main_content = soup.find_all(lambda tag: (tag.name == 'div' or tag.name == 'main') and ('text-container' in tag.get('class', []) or 'main-content' in tag.get('id', [])))
            if main_content:
                print("Found main content")
                print(url)
                main_content_texts = [element.text.strip() for element in main_content]
                main_content = ' '.join(main_content_texts)
                main_content = main_content.replace('\n', ' ').replace('\r', '').replace('\xa0', ' ')
                main_content = re.sub(r'\s+', ' ', main_content).strip()
                content[url] = main_content
                # Save updated content to JSON file after each URL is processed
                save_to_json(content, file_path)
            else:
                print("No main content found")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from {url}: {e}")
    return content

def save_to_json(content, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(content, json_file, indent=4)

def pull_AI_Search_data(sitemap_urls, initial_pull = False):
    urls_to_update = get_sitemap_urls(sitemap_urls, initial_pull)
    file_path = "output.json"  # Define the file path here so it can be used in fetch_site_data
    content = fetch_site_data(urls_to_update, file_path)
    return content

sitemap_urls = [
    "https://divinity.howard.edu/sitemap.xml",
    "https://hubison.com/sitemap.xml",
    "https://medicine.howard.edu/sitemap.xml",
    "https://gs.howard.edu/sitemap.xml",
    "https://law.howard.edu/sitemap.xml",
    "https://technology.howard.edu/sitemap.xml",
    "https://physics.howard.edu/sitemap.xml",
    "https://alum.howard.edu/sitemap.xml",
    "https://founders.howard.edu/sitemap.xml",
    "https://www.huhealthcare.com/sitemap.xml",
    "https://socialwork.howard.edu/sitemap.xml",
    "https://history.howard.edu/sitemap.xml",
    "https://sociologyandcriminology.howard.edu/sitemap.xml"
]

content = pull_AI_Search_data(sitemap_urls=sitemap_urls, initial_pull=True)