from glob import glob


def convert(post, tar):
    tar_before, tar_after = tar
    with open(post, "r+") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            conv = line.replace(tar_before, tar_after)
            if not line == conv:
                print("CONVERTED POST:", post)
                print(f"LINE {i} [BEFORE]:\t", line)
                print(f"LINE {i} [AFTER]:\t", conv)
            lines[i] = conv
        f.seek(0)
        f.writelines(lines)
        f.truncate()


if __name__ == "__main__":
    posts = glob("_posts/*.md")
    tar = ("- 2. MLOps", "- 3. MLOps")
    for post in posts:
        convert(post, tar)
