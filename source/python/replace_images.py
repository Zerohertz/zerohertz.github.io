#!/usr/bin/env python3

import os
import re
import shutil
import sys

import zerohertzLib as zz


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


def replace_image_urls(md_content, image_urls):
    image_move = {}
    for image_url in image_urls:
        logger.info(image_url)
        org, src, alt = image_url["org"], image_url["src"], image_url["alt"]
        if not org or not src or not alt:
            logger.critical(f"org: {org}\nsrc: {src}\nalt: {alt}")
            exit()
        src = src.split("?")[0]
        ext = src.split(".")[-1]
        alt = (
            alt.replace(" - ", "-")
            .replace(" + ", "+")
            .replace(" ", "-")
            .replace("_", "-")
            .replace("&", "-")
            .replace(":", "-")
            .replace("(", "-")
            .replace(")", "-")
            .replace(",", "-")
            .lower()
            .replace("--", "-")
        )
        new_src = os.path.join(os.path.dirname(src), f"{alt}.{ext}")
        if org.startswith("!["):
            new = f"""![{alt}]({new_src})"""
        elif org.startswith("<img"):
            args = image_url["args"]
            if args:
                new = f"""<img src="{new_src}" alt="{alt}" {args} />"""
            else:
                new = f"""<img src="{new_src}" alt="{alt}" />"""
        logger.info(org)
        logger.info(new)
        if org not in md_content:
            logger.error(org)
            exit()
        md_content = md_content.replace(org, new)
        if os.path.join(images, os.path.basename(src)) in image_move.values():
            logger.error(src)
            exit()
        if os.path.join(images, os.path.basename(new_src)) in image_move:
            logger.error(new_src)
            exit()
        image_move[os.path.join(images, os.path.basename(new_src))] = os.path.join(
            images, os.path.basename(src)
        )
    for k, v in image_move.items():
        shutil.move(v, k)
    return md_content


def read_md_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        md_content = file.read()
    return md_content


def write_md_file(file_path, md_content):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(md_content)


def main(post):
    md_content = read_md_file(post)
    image_urls = parse_image_urls_from_md(md_content)
    md_content = replace_image_urls(md_content, image_urls)
    write_md_file(post, md_content)


if __name__ == "__main__":
    file_name = sys.argv[1]
    logger = zz.logging.Logger("REPLACE IMAGES")
    post = os.path.join(os.path.dirname(__file__), "..", "_posts", f"{file_name}.md")
    images = os.path.join(os.path.dirname(__file__), "..", "images", file_name)
    main(post)
