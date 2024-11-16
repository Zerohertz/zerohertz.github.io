import os

import requests
import zerohertzLib as zz

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OWNER = "Zerohertz"
REPO = "zerohertz.github.io"

headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}


def delete_comment(comment_id):
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/issues/comments/{comment_id}"
    response = requests.delete(url, headers=headers)
    return response.status_code == 204


def delete_all_issue_comments():
    page = 1
    while True:
        url = f"https://api.github.com/repos/{OWNER}/{REPO}/issues/comments?page={page}"
        response = requests.get(url, headers=headers)
        comments = response.json()
        logger.info(comments)
        if not comments:
            break
        for comment in comments:
            comment_id = comment["id"]
            if delete_comment(comment_id):
                logger.info(f"Deleted comment ID: {comment_id}")
            else:
                logger.info(f"Failed to delete comment ID: {comment_id}")
        page += 1


if __name__ == "__main__":
    logger = zz.logging.Logger("DELETE COMMENTS")
    delete_all_issue_comments()
