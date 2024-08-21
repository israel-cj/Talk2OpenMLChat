import requests
from bs4 import BeautifulSoup
import csv
import os

def extract_text_from_tags(soup, tag):
    return ' '.join([element.get_text(strip=True) for element in soup.find_all(tag)])

def crawl(url, visited, writer, file, base_urls):
    if url in visited:
        return
    visited.add(url)
    
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to retrieve {url}: {e}")
        return
    
    soup = BeautifulSoup(response.text, 'html.parser')
    title = soup.title.string if soup.title else 'No title'
    body_text = soup.body.get_text(separator=' ', strip=True) if soup.body else 'No body text'
    header_links_text = ' '.join([link.get_text(strip=True) for link in soup.find_all('a', href=True)])
    h1 = extract_text_from_tags(soup, 'h1')
    h2 = extract_text_from_tags(soup, 'h2')
    h3 = extract_text_from_tags(soup, 'h3')
    h4 = extract_text_from_tags(soup, 'h4')
    
    writer.writerow([url, body_text, header_links_text, h1, h2, h3, h4, title])
    file.flush()  # Flush the buffer to ensure data is written to disk
    
    # Print the URL and the current size of the CSV file
    file_size = os.path.getsize(file.name)
    print(f"Crawled URL: {url}, CSV size: {file_size} bytes")
    
    for link in soup.find_all('a', href=True):
        full_url = requests.compat.urljoin(url, link['href'])
        if any(full_url.startswith(base_url) for base_url in base_urls):
            crawl(full_url, visited, writer, file, base_urls)
        # time.sleep(1)  # Be polite and avoid overloading the server

def main():
    file_path = 'openml_docs_API_together.csv'
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['URL', 'Body Text', 'Header Links Text', 'H1', 'H2', 'H3', 'H4', 'Title'])
        
        visited = set()
        base_urls = [
            "https://openml.github.io/openml-python/main/",
            "https://docs.openml.org/"
        ]
        for start_url in base_urls:
            crawl(start_url, visited, writer, file, base_urls)

if __name__ == "__main__":
    main()