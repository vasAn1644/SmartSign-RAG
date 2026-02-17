"""
Responsible for:
1) Scraping German road sign images and metadata from Wikipedia
2) Downloading images locally
3) Crawling text pages related to German driving rules from iamexpat.de
4) Saving structured JSON + text corpus for RAG pipeline

JSON schema is NOT changed to preserve pipeline compatibility.
"""

import os
import re
import json
import time
import hashlib
import requests
from typing import List, Set
from urllib.parse import urljoin, urlparse
from collections import deque

from bs4 import BeautifulSoup


# --- Sources ---
WIKI_SIGNS_URL = "https://en.wikipedia.org/wiki/Road_signs_in_Germany"

BASE_URL = "https://www.iamexpat.de"
ROAD_SIGNS_URL = f"{BASE_URL}/expat-info/driving-germany/road-signs"

START_URLS = [
    ROAD_SIGNS_URL,
    f"{BASE_URL}/expat-info/driving-germany"
]

BASE_DOMAIN = "www.iamexpat.de"

# --- Paths ---
IMAGE_JSON_PATH = "data/germany_road_signs.json"
TEXT_OUTPUT_DIR = "data/text_files"
LOCAL_IMAGE_DIR = "data/images/wiki_signs"

MIN_TEXT_LENGTH = 30

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}



def safe_filename(text: str, ext: str = ".png") -> str:
    base = re.sub(r"[^a-zA-Z0-9_-]", "_", text.lower()).strip("_")
    if not base:
        base = hashlib.md5(text.encode()).hexdigest()[:12]
    return base + ext


def download_image(url: str, title: str) -> str:
    """Download image locally and return local path"""
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
    except requests.RequestException:
        return ""

    os.makedirs(LOCAL_IMAGE_DIR, exist_ok=True)

    ext = os.path.splitext(urlparse(url).path)[1]
    if not ext:
        ext = ".png"

    filename = safe_filename(title, ext)
    local_path = os.path.join(LOCAL_IMAGE_DIR, filename)

    with open(local_path, "wb") as f:
        f.write(r.content)

    return local_path


def clean_filename(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if not path:
        return "index.txt"
    name = path.split("/")[-1]
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name) + ".txt"



def extract_text(soup: BeautifulSoup) -> str:
    content: List[str] = []

    for tag in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        text = tag.get_text(" ", strip=True)
        if len(text) > MIN_TEXT_LENGTH:
            content.append(text)

    return "\n\n".join(content)


def is_valid_link(link: str) -> bool:
    parsed = urlparse(link)
    return (
        parsed.netloc == BASE_DOMAIN
        and parsed.path.startswith("/expat-info/driving-germany")
    )



def scrape_wikipedia_signs(output_path: str = IMAGE_JSON_PATH):
    print(f"Connecting to Wikipedia: {WIKI_SIGNS_URL}")

    try:
        response = requests.get(WIKI_SIGNS_URL, headers=HEADERS, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to load Wikipedia page: {e}")

    soup = BeautifulSoup(response.content, "html.parser")

    content = soup.find("div", id="mw-content-text")

    all_data = []

    # Wikipedia uses tables + galleries
    images = content.find_all("img")

    for img in images:
        src = img.get("src", "")
        alt = img.get("alt", "").strip()

        if not src:
            continue

        # Wikipedia images are often //upload.wikimedia.org
        if src.startswith("//"):
            img_url = "https:" + src
        elif src.startswith("http"):
            img_url = src
        else:
            continue

        title = alt if alt else "German road sign"

        # Category from nearest header
        category = "General"
        prev_h = img.find_previous(["h2", "h3"])
        if prev_h:
            category = prev_h.get_text(strip=True)

        # download locally
        local_path = download_image(img_url, title)

        all_data.append({
            "category": category,
            "title": title,
            "image_url": img_url,
            "status": "ok"
        })

    # deduplicate by image_url
    seen: Set[str] = set()
    unique_data = [
        item for item in all_data
        if item["image_url"] not in seen and not seen.add(item["image_url"])
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(unique_data, f, indent=2, ensure_ascii=False)

    print(f"{len(unique_data)} Wikipedia road signs collected")
    print(f"Images saved locally in: {LOCAL_IMAGE_DIR}")


class PageCrawler:
    def __init__(self, output_dir: str):
        self.visited: Set[str] = set()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def crawl(self, start_urls: List[str], max_pages: int = 200):
        queue = deque(start_urls)

        while queue and len(self.visited) < max_pages:
            url = queue.popleft()

            if url in self.visited:
                continue

            print(f"Processing: {url}")
            self.visited.add(url)

            try:
                r = requests.get(url, headers=HEADERS, timeout=30)
                r.raise_for_status()
            except requests.RequestException:
                continue

            soup = BeautifulSoup(r.text, "html.parser")

            text = extract_text(soup)
            if text:
                filename = clean_filename(url)
                filepath = os.path.join(self.output_dir, filename)

                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(text)

            for a in soup.find_all("a", href=True):
                link = urljoin(url, a["href"])
                if is_valid_link(link) and link not in self.visited:
                    queue.append(link)

            time.sleep(0.3)  # polite crawling


def run_scraper():
    print("\n--- SCRAPING WIKIPEDIA IMAGES ---")
    scrape_wikipedia_signs()

    print("\n--- SCRAPING IAMEXPAT TEXTS ---")
    crawler = PageCrawler(TEXT_OUTPUT_DIR)
    crawler.crawl(START_URLS)

    print(f"\nTexts saved: {len(os.listdir(TEXT_OUTPUT_DIR))}")
    print("\nPipeline data collection finished successfully ")


if __name__ == "__main__":
    run_scraper()
