#!/usr/bin/env python3

import os
import re
import sys
from glob import glob

import zerohertzLib as zz
from tqdm import tqdm


def parse_html_img_tags(md_content):
    html_img_pattern = r"<img\s+(.*?)\s*/?>"
    result = []
    matches = re.finditer(html_img_pattern, md_content)
    for match in matches:
        full_tag = match.group(0)
        attributes = match.group(1)
        src_match = re.search(r'src=["\'](.*?)["\']', attributes)
        alt_match = re.search(r'alt=["\'](.*?)["\']', attributes)
        src = src_match.group(1) if src_match else ""
        alt = alt_match.group(1) if alt_match else ""
        cleaned_args = re.sub(
            r'(src=["\'].*?["\']|alt=["\'].*?["\'])', "", attributes
        ).strip()
        cleaned_args = " ".join(cleaned_args.split())  # Remove excess spaces
        result.append({"org": full_tag, "src": src, "alt": alt, "args": cleaned_args})
    return result


def parse_markdown_img_tags(md_content):
    markdown_img_pattern = r"(!\[)(.*?)(\]\()(.*?)(\))"
    result = []
    matches = re.findall(markdown_img_pattern, md_content)
    for match in matches:
        full_tag = "".join(match)
        alt = match[1]
        src = match[3]
        result.append({"org": full_tag, "src": src, "alt": alt})
    return result


def parse_image_urls_from_md(md_content):
    return parse_markdown_img_tags(md_content) + parse_html_img_tags(md_content)


def valid_image_urls(md_content, image_urls):
    for image_url in image_urls:
        org, src, alt = image_url["org"], image_url["src"], image_url["alt"]
        # logger.info(images)
        # logger.info(images + src)
        if os.path.isfile(images + src):
            continue
        logger.critical(src)


def read_md_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        md_content = file.read()
    return md_content


def main(post):
    md_content = read_md_file(post)
    image_urls = parse_image_urls_from_md(md_content)
    valid_image_urls(md_content, image_urls)


if __name__ == "__main__":
    logger = zz.logging.Logger("VALID IMAGES")
    posts = glob(os.path.join(os.path.dirname(__file__), "..", "_posts", "*.md"))
    images = os.path.join(os.path.dirname(__file__), "..")
    for post in posts:
        main(post)
