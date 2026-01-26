import os
import json
import pytest
from src.scraper import parse_all_signs

TEST_OUTPUT_PATH = "data/test_germany_road_signs.json"

def test_parse_all_signs_creates_json():
    if os.path.exists(TEST_OUTPUT_PATH):
        os.remove(TEST_OUTPUT_PATH)

    parse_all_signs(output_path=TEST_OUTPUT_PATH)

    assert os.path.exists(TEST_OUTPUT_PATH), "JSON file was not created"

    with open(TEST_OUTPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list), "Parsed data is not a list"
    assert len(data) > 0, "Parsed data is empty"

    sample = data[0]
    assert "title" in sample
    assert "category" in sample
    assert "image_url" in sample
    assert "status" in sample

    os.remove(TEST_OUTPUT_PATH)
