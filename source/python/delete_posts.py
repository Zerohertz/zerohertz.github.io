from glob import glob


def delete(post, tar):
    with open(post, "r+") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if tar in line:
                print("DELTED POST:", post)
                print(f"LINE {i}:\t", line)
                lines[i] = ""
        f.seek(0)
        f.writelines(lines)
        f.truncate()


if __name__ == "__main__":
    posts = glob("_posts/*.md")
    tar = "thumbnail"
    for post in posts:
        delete(post, tar)
