import re
import requests
import os

def parse_image_urls_from_md(md_content):
    markdown_img_pattern = r'!\[.*?\]\((.*?)\)'
    html_img_pattern = r'<img [^>]*src=["\'](.*?)["\']'
    markdown_img_urls = re.findall(markdown_img_pattern, md_content)
    html_img_urls = re.findall(html_img_pattern, md_content)
    all_img_urls = markdown_img_urls + html_img_urls
    return all_img_urls

def download_image(url, save_directory):
    print(url)
    filename = os.path.basename(url)
    save_path = os.path.join(save_directory, filename)
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() 
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded: {filename}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")
def read_md_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        md_content = file.read()
    return md_content

def main(md_file_path, save_directory):
    md_content = read_md_file(md_file_path)
    image_urls = parse_image_urls_from_md(md_content)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    for url in image_urls:
        download_image(url, save_directory)

md_file_path = '_posts/inverse-kinematics-in-unreal-engine-1.md'
save_directory = 'downloaded_images'

main(md_file_path, save_directory)
