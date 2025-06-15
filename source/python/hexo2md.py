#!/usr/bin/env python3

# NOTE:
# sudo ln -sf ${PWD}/hexo2md.py /usr/bin/hexo2md

import os
import re
import sys

from loguru import logger


def convert_latex_references(text):
    r"""
    Convert LaTeX-style references to markdown format.
    Example: $\_[$[$\_{1}$](url1)$\_,$[$\_{2}$](url2)$\_]$ -> [[1](url1), [2](url2)]
    """
    # Pattern to match individual references within the block
    ref_pattern = r"\$\[\$\\_\{([^}]+)\}\$\]\(([^)]+)\)\$\\_"

    # Pattern to match the entire block: $\_[...content...]$
    full_pattern = r"\$\\_\[(.*?)\]\$"

    def replace_block(match):
        content = match.group(1)
        # Extract all references from the content
        refs = re.findall(ref_pattern, content)
        if refs:
            # Convert to markdown format
            markdown_refs = [f"[{num}]({url})" for num, url in refs]
            return f"[{', '.join(markdown_refs)}]"
        return match.group(0)

    return re.sub(full_pattern, replace_block, text)


def convert_image_tags(text):
    """
    Convert image tags to markdown format and add GitHub raw URL prefix for relative paths.
    Examples:
    - <img src="/images/test.png" alt="test"> -> ![test](https://raw.githubusercontent.com/Zerohertz/zerohertz.github.io/main/source/images/test.png)
    - ![alt](/images/test.png) -> ![alt](https://raw.githubusercontent.com/Zerohertz/zerohertz.github.io/main/source/images/test.png)
    """
    # Pattern to match HTML img tags
    html_img_pattern = r'<img\s+[^>]*src="([^"]*)"[^>]*alt="([^"]*)"[^>]*>'

    # Pattern to match markdown image syntax
    md_img_pattern = r"!\[([^\]]*)\]\(([^)]*)\)"

    def replace_html_img(match):
        src = match.group(1)
        alt = match.group(2)

        # Add GitHub raw URL prefix if src starts with /
        if src.startswith("/"):
            src = f"https://raw.githubusercontent.com/Zerohertz/zerohertz.github.io/main/source{src}"

        return f"![{alt}]({src})"

    def replace_md_img(match):
        alt = match.group(1)
        src = match.group(2)

        # Add GitHub raw URL prefix if src starts with /
        if src.startswith("/"):
            src = f"https://raw.githubusercontent.com/Zerohertz/zerohertz.github.io/main/source{src}"

        return f"![{alt}]({src})"

    # Apply HTML img tag conversion first
    text = re.sub(html_img_pattern, replace_html_img, text)

    # Apply markdown image conversion
    text = re.sub(md_img_pattern, replace_md_img, text)

    return text


def remove_comments(text):
    """
    Remove HTML/markdown comments.
    Example: <!-- comment --> -> (removed)
    """
    # Pattern to match HTML/markdown comments
    comment_pattern = r"<!--.*?-->"

    return re.sub(comment_pattern, "", text, flags=re.DOTALL)


def convert_code_blocks(text):
    """
    Convert code blocks with file names and links to blockquote + code block format.
    Examples:
    - ```python filename.py -> > filename.py\n```python
    - ```python filename.py filelink -> > [filename.py](filelink)\n```python
    """
    # Pattern to match code blocks with file names and optional links (only on the same line)
    # Only matches when there's actual filename with extension (contains a dot)
    pattern = r"```(\w+)\s+([^\s\n\r]+\.[^\s\n\r]+(?:\s+[^\n\r]+)?)"

    def replace_code_block(match):
        language = match.group(1)
        file_info = match.group(2).strip()

        # Split file info into filename and optional link
        parts = file_info.split(" ", 1)
        filename = parts[0]

        # If there's a link, create markdown link format
        if len(parts) > 1:
            link = parts[1]
            return f"> [{filename}]({link})\n\n```{language}\n"
        else:
            return f"> {filename}\n\n```{language}\n"

    return re.sub(pattern, replace_code_block, text)


def convert_special_tags(text):
    """
    Convert {% tag %} content {% endtag %} to blockquote format.
    Example: {% cq %} hi {% endcq %} -> > hi
    Example: {% note 이름 %} content {% endnote %} -> > content
    Example: {% note danger %} content {% endnote %} -> > content
    """
    # Pattern to match {% tag [optional_text] %} content {% endtag %}
    pattern = r"\{%\s+(\w+)(?:\s+([^%]*))?\s+%\}(.*?)\{%\s+end\1\s+%\}"

    def replace_tag(match):
        tag_name = match.group(1)
        tag_param = match.group(2).strip() if match.group(2) else ""
        content = match.group(3).strip()

        # List of note types that should be excluded from title
        excluded_note_types = {"info", "warning", "danger", "success", "tip", "note"}

        # If it's a note tag with parameter that's not a standard note type
        if (
            tag_name == "note"
            and tag_param
            and tag_param.lower() not in excluded_note_types
        ):
            # For notes with custom parameters, include the title in the blockquote as bold
            lines = [f"**{tag_param}**"] + content.split("\n")
            quoted_lines = ["> " + line.strip() for line in lines if line.strip()]
            return "\n".join(quoted_lines)

        # Convert to blockquote for all other cases
        lines = content.split("\n")
        quoted_lines = ["> " + line.strip() for line in lines if line.strip()]
        return "\n".join(quoted_lines)

    return re.sub(pattern, replace_tag, text, flags=re.DOTALL)


def process_markdown(content):
    """
    Process markdown content by applying all transformations.
    """
    # Apply special tag conversion first (before LaTeX references are converted)
    content = convert_special_tags(content)

    # Apply code block conversion
    content = convert_code_blocks(content)

    # Apply LaTeX reference conversion
    content = convert_latex_references(content)

    # Apply image tag conversion
    content = convert_image_tags(content)

    # Remove HTML/markdown comments
    content = remove_comments(content)

    return content


def main():
    if len(sys.argv) != 2:
        logger.info("Usage: hexo2md {POST_NAME}", file=sys.stderr)
        sys.exit(1)

    post_name = sys.argv[1]
    filename = f"/home/zerohertz/Zerohertz/blog/source/_posts/{post_name}.md"

    # Check if file exists
    if not os.path.exists(filename):
        logger.info(f"Error: File '{filename}' not found", file=sys.stderr)
        sys.exit(1)

    try:
        # Read the markdown file
        with open(filename, "r", encoding="utf-8") as file:
            content = file.read()

        # Process the content
        processed_content = process_markdown(content)

        # Output to stdout
        print(processed_content)

    except Exception as e:
        logger.exception(f"Error processing file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
