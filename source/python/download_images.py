import os
import re
import shutil
from glob import glob

import requests
import zerohertzLib as zz
from tqdm import tqdm


def parse_image_urls_from_md(md_content):
    markdown_img_pattern = r"(!\[.*?\]\()(.*?)(\))"
    html_img_pattern = r'(<img [^>]*src=["\'])(.*?)["\']'
    markdown_img_urls = re.findall(markdown_img_pattern, md_content)
    html_img_urls = re.findall(html_img_pattern, md_content)
    return markdown_img_urls, html_img_urls


def replace_image_urls(md_content, image_urls, redirect_urls, save_directory):
    for (prefix, url, suffix), new_url in zip(image_urls[0], redirect_urls[0]):
        new_url = f"/images/{save_directory}/{os.path.basename(new_url)}"
        md_content = md_content.replace(
            f"{prefix}{url}{suffix}", f"{prefix}{new_url}{suffix}"
        )
    for (prefix, url), new_url in zip(image_urls[1], redirect_urls[1]):
        new_url = f"/images/{save_directory}/{os.path.basename(new_url)}"
        md_content = md_content.replace(f'{prefix}{url}"', f'{prefix}{new_url}"')
    return md_content


def download_image(url, save_directory):
    logger.info(url)
    try:
        response = requests.get(url, stream=True, allow_redirects=True)
        response.raise_for_status()
        url = response.url
        filename = os.path.basename(url.split("?")[0])
        save_path = os.path.join(save_directory, filename)
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        logger.info(f"Downloaded: {filename}")
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
    return url


def read_md_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        md_content = file.read()
    return md_content


def write_md_file(file_path, md_content):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(md_content)


def main(post):
    save_directory = os.path.join(images, os.path.basename(post).replace(".md", ""))
    md_content = read_md_file(post)
    image_urls = parse_image_urls_from_md(md_content)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    redirect_urls = [[], []]
    for _, url, _ in image_urls[0]:
        redirect_urls[0].append(download_image(url, save_directory))
    for _, url in image_urls[1]:
        redirect_urls[1].append(download_image(url, save_directory))
    if not os.listdir(save_directory):
        shutil.rmtree(save_directory)
        logger.info(f"Deleted empty directory: {save_directory}")
    md_content = replace_image_urls(
        md_content, image_urls, redirect_urls, os.path.basename(post).replace(".md", "")
    )
    write_md_file(post, md_content)


if __name__ == "__main__":
    logger = zz.logging.Logger("DOWNLOAD IMAGES")
    posts = os.path.join(os.path.dirname(__file__), "..", "_posts", "*.md")
    images = os.path.join(os.path.dirname(__file__), "..", "images")
    # zz.util.rmtree(images)
    for post in tqdm(glob(posts)):
        main(post)
