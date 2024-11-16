import argparse
import re
from glob import glob

import requests
import zerohertzLib as zz
from tqdm import tqdm


def redirect(url):
    try:
        response = requests.get(url, allow_redirects=True)
        redirected_url = response.url
        return True, redirected_url
    except Exception as e:
        print(e)
        print("ERROR:\t", url)
        return False, None


def parse(line, img=True):
    try:
        if img:
            pattern = r"!\[.*?\]\((.*?)\)"
            links = re.findall(pattern, line)
        else:
            pattern = r"\[.*?\]\((.*?)\)"
            links = re.findall(pattern, line)
    except:
        print("ERROR:\t", line)
    return links


def convert(post):
    with open(post, "r+") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            links = parse(line, False)
            for link in links:
                if "coupang" in link:
                    continue
                Edit, new_link = redirect(link)
                if Edit and not link == new_link:
                    conv = line.replace(link, new_link)
                    print("CONVERTED POST:", post)
                    print(f"LINE {i} [BEFORE]:\t", line)
                    print(f"LINE {i} [AFTER]:\t", conv)
                    if input("IF YOU WANT TO CHANGE, THEN INSERT 0:\t") == "0":
                        lines[i] = conv
        f.seek(0)
        f.writelines(lines)
        f.truncate()


if __name__ == "__main__":
    logger = zz.logging.Logger("DOWNLOAD IMAGES")
    parser = argparse.ArgumentParser()
    parser.add_argument("-t")
    args = parser.parse_args()
    if args.t:
        posts = glob(f"_posts/{args.t}.md")
        for post in tqdm(posts):
            convert(post)
    else:
        posts = glob("_posts/*.md")
        for post in tqdm(posts):
            convert(post)
