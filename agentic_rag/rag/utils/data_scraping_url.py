import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


class SimpleWebCrawler:
    def __init__(self, base_url, output_dir="../data/scraped_pages", url_log_file="../data/read_urls.txt", max_depth=1):
        self.base_url = base_url
        self.visited = set()
        self.max_depth = max_depth
        self.output_dir = output_dir
        self.url_log_file = url_log_file

        # Ensure output directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.url_log_file), exist_ok=True)

        # Clear old URL log file
        open(self.url_log_file, "w").close()

    def is_valid_url(self, url):
        parsed_base = urlparse(self.base_url)
        parsed_url = urlparse(url)
        return parsed_base.netloc == parsed_url.netloc

    def get_filename_from_url(self, url):
        """
        Converts a URL into a clean .txt filename.
        Example:
          https://example.com/about/team -> team.txt
          https://example.com -> index.txt
        """
        path = urlparse(url).path.strip("/")
        if not path:
            return "index.txt"
        last_part = path.split("/")[-1]
        last_part = re.sub(r'[^A-Za-z0-9_\-]', '_', last_part)
        return f"{last_part}.txt"

    def scrape_page(self, url, depth=0):
        if depth > self.max_depth or url in self.visited:
            return

        try:
            print(f"Scraping: {url}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract visible text
            text = " ".join([p.get_text(strip=True) for p in soup.find_all("p")])
            if not text.strip():
                text = soup.get_text(separator=" ", strip=True)

            # Save page content to file
            filename = self.get_filename_from_url(url)
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)

            # Log the scraped URL
            with open(self.url_log_file, "a", encoding="utf-8") as log:
                log.write(url + "\n")

            print(f"Saved content to {filepath}")
            self.visited.add(url)

            # Recursively scrape internal links
            for link_tag in soup.find_all("a", href=True):
                next_url = urljoin(url, link_tag["href"])
                if self.is_valid_url(next_url):
                    self.scrape_page(next_url, depth + 1)

        except Exception as e:
            print(f"Failed to scrape {url}: {e}")

    def run(self):
        self.scrape_page(self.base_url)
        print(f"\n Finished crawling.")
        print(f"Text files saved in: {self.output_dir}")
        print(f"URL list written to: {self.url_log_file}")


# # Example usage
# if __name__ == "__main__":
#     homepage = "https://www.gainwelltechnologies.com"
#     crawler = SimpleWebCrawler(homepage, max_depth=1)
#     crawler.run()
