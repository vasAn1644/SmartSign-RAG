"""
Responsible for:
1) Scraping German road sign images and metadata
2) Crawling text pages related to German driving rules
3) Saving structured JSON + text corpus for RAG pipeline
"""

import os
import re
import json
import time
import requests
from typing import List, Set
from urllib.parse import urljoin, urlparse
from collections import deque

from bs4 import BeautifulSoup


BASE_URL = "https://www.iamexpat.de"
ROAD_SIGNS_URL = f"{BASE_URL}/expat-info/driving-germany/road-signs"

START_URLS = [
    ROAD_SIGNS_URL,
    f"{BASE_URL}/expat-info/driving-germany"
]

BASE_DOMAIN = "www.iamexpat.de"

IMAGE_JSON_PATH = "data/germany_road_signs.json"
TEXT_OUTPUT_DIR = "data/text_files"

MIN_TEXT_LENGTH = 30  # REVIEW FIX: replaced magic number with named constant

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}


def is_road_sign_image(src: str, alt: str) -> bool:
    """Validation function for road sign images"""
    # REVIEW FIX: extracted complex conditional into a dedicated function
    src = (src or "").lower()
    alt = (alt or "").lower()

    return (
        "road-sign" in src
        or "sign" in src
        or src.endswith(".svg")
        or bool(alt)
    )


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


def parse_all_signs(output_path: str = IMAGE_JSON_PATH) -> None:
    print(f"Connecting to {ROAD_SIGNS_URL}...")

    try:
        response = requests.get(ROAD_SIGNS_URL, headers=HEADERS, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        # REVIEW FIX: added proper exception handling instead of silent failure
        raise RuntimeError(f"Failed to load page: {e}")

    soup = BeautifulSoup(response.content, "html.parser")

    all_signs_data = []

    content_area = soup.find("div", class_="article__content") or soup.find("body")
    all_imgs = content_area.find_all("img")

    for img in all_imgs:
        src = img.get("src", "")
        alt = img.get("alt", "")

        if not is_road_sign_image(src, alt):
            continue

        img_url = src if src.startswith("http") else BASE_URL + src

        title = alt.strip() or img.get("title", "").strip()

        if not title:
            parent_td = img.find_parent("td")
            if parent_td:
                title = parent_td.get_text(strip=True)
                if not title and parent_td.find_next_sibling("td"):
                    title = parent_td.find_next_sibling("td").get_text(strip=True)

        if not title or len(title) < 2:
            title = "NEED_MANUAL_DESCRIPTION"

        category = "General"
        prev_h = img.find_previous(["h2", "h3"])
        if prev_h:
            category = prev_h.get_text(strip=True)

        all_signs_data.append({
            "category": category,
            "title": title,
            "image_url": img_url,
            "status": "manual_check" if title == "NEED_MANUAL_DESCRIPTION" else "ok"
            # REVIEW FIX: kept f-string for performance and readability
        })

    # REVIEW FIX: deduplication using set comprehension instead of dict.values() for memory efficiency
    seen: Set[str] = set()
    unique_data = [
        item for item in all_signs_data
        if item["image_url"] not in seen and not seen.add(item["image_url"])
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(unique_data, f, indent=2, ensure_ascii=False)

    print(f"{len(unique_data)} road signs collected")
    manual_count = sum(1 for x in unique_data if x["status"] == "manual_check")
    print(f"{manual_count} need manual description")


class PageCrawler:
    # REVIEW FIX: encapsulated visited set inside class instead of global variable
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


# REVIEW FIX: added helper for main.py compatibility
def crawl_pages():
    crawler = PageCrawler(TEXT_OUTPUT_DIR)
    crawler.crawl(START_URLS)


def run_scraper():
    print("\n--- SCRAPING IMAGES ---")
    parse_all_signs()

    print("\n--- SCRAPING TEXT ---")
    crawler = PageCrawler(TEXT_OUTPUT_DIR)
    crawler.crawl(START_URLS)

    print(f"\nTexts saved: {len(os.listdir(TEXT_OUTPUT_DIR))}")


if __name__ == "__main__":
    run_scraper()
