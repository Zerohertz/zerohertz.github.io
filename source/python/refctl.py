#!/usr/bin/env python3

# NOTE:
# sudo ln -sf ${PWD}/refctl.py /usr/bin/refctl

import argparse
import os
import re
import sys
import uuid
from typing import Dict, List, Tuple

from loguru import logger

PREFIX = r"$\_[$"
TEMPLATE = r"[$\_{{REFERENCE_INDEX}}$](REFERENCE_URL)"
JOIN = r"$\_,$"
POSTFIX = r"$\_]$"


class RefCtl:
    def __init__(self, posts_dir: str):
        self.posts_dir = posts_dir

    def get_post_path(self, post_name: str) -> str:
        return os.path.join(self.posts_dir, f"{post_name}.md")

    def read_post_file(self, post_name: str) -> List[str]:
        post_path = self.get_post_path(post_name)
        try:
            with open(post_path, "r", encoding="utf-8") as file:
                return file.readlines()
        except FileNotFoundError:
            logger.error(f"Post file not found: {post_path}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error reading post file: {e}")
            sys.exit(1)

    def write_post_file(self, post_name: str, lines: List[str], suffix: str = ""):
        if suffix:
            post_path = os.path.join(self.posts_dir, f"{post_name}{suffix}.md")
        else:
            post_path = self.get_post_path(post_name)

        try:
            with open(post_path, "w", encoding="utf-8") as file:
                file.writelines(lines)
            logger.info(f"File written: {post_path}")
        except Exception as e:
            logger.error(f"Error writing file: {e}")
            sys.exit(1)

    def find_references_section(self, lines: List[str]) -> Tuple[int, int]:
        """Find References section boundaries. Returns (start_line, end_line)."""
        start_line = -1
        end_line = -1
        references_count = 0

        for i, line in enumerate(lines):
            line = line.strip()
            if line == "{% note References %}":
                references_count += 1
                if references_count > 1:
                    logger.error("Multiple References sections found")
                    sys.exit(1)
                start_line = i
            elif line == "{% endnote %}" and start_line != -1:
                end_line = i
                break

        return start_line, end_line

    def parse_reference_line(self, line: str) -> Tuple[int, str, str, str]:
        """Parse reference line and return (index, name, url, uuid)."""
        # Pattern: f'{REF_INDEX} [REF_NAME](REF_URL) <!-- {UUID} -->'
        pattern = r"(\d+)\.\s*\[(.*?)\]\((.*?)\)\s*<!--\s*(.*?)\s*-->"
        match = re.match(pattern, line.strip())
        if match:
            return int(match.group(1)), match.group(2), match.group(3), match.group(4)
        return None

    def parse_references_with_uuid(self, lines: List[str]) -> Dict[str, Dict[str, str]]:
        """Parse references and return {UUID: {'index': REF_INDEX, 'url': REF_URL}}"""
        start_line, end_line = self.find_references_section(lines)
        refs = {}

        if start_line == -1:
            return refs

        for i in range(start_line + 1, end_line):
            line = lines[i]
            parsed = self.parse_reference_line(line)
            if parsed:
                index, name, url, ref_uuid = parsed
                refs[ref_uuid] = {"index": str(index), "url": url, "name": name}

        return refs

    def parse_all_references(self, lines: List[str]) -> Dict[int, Dict[str, str]]:
        """Parse all references (both with and without UUID)."""
        start_line, end_line = self.find_references_section(lines)
        references = {}

        if start_line == -1:
            return references

        for i in range(start_line + 1, end_line):
            line = lines[i].strip()
            if line:
                # First try to parse with UUID
                uuid_match = re.match(
                    r"(\d+)\.\s*\[(.*?)\]\((.*?)\)\s*<!--\s*(.*?)\s*-->", line
                )
                if uuid_match:
                    references[int(uuid_match.group(1))] = {
                        "name": uuid_match.group(2),
                        "url": uuid_match.group(3),
                        "uuid": uuid_match.group(4),
                    }
                else:
                    # Try to parse without UUID
                    no_uuid_match = re.match(r"(\d+)\.\s*\[(.*?)\]\((.*?)\)", line)
                    if no_uuid_match:
                        references[int(no_uuid_match.group(1))] = {
                            "name": no_uuid_match.group(2),
                            "url": no_uuid_match.group(3),
                            "uuid": None,
                        }

        return references

    def parse_old_references(self, lines: List[str]) -> Dict[int, Dict[str, str]]:
        """Parse old-style references (without UUID)."""
        start_line, end_line = self.find_references_section(lines)
        references = {}

        if start_line == -1:
            return references

        for i in range(start_line + 1, end_line):
            line = lines[i].strip()
            if line:
                match = re.match(r"(\d+)\.\s*\[(.*?)\]\((.*?)\)", line)
                if match:
                    references[int(match.group(1))] = {
                        "name": match.group(2),
                        "url": match.group(3),
                    }

        return references

    def add_reference(
        self, post_name: str, ref_url: str, ref_name: str = None, ref_index: int = None
    ):
        """Add a reference to the post."""
        lines = self.read_post_file(post_name)
        start_line, end_line = self.find_references_section(lines)

        # If no References section exists, add one at the end
        if start_line == -1:
            if not lines[-1].endswith("\n"):
                lines.append("\n")
            lines.extend(
                ["---\n", "\n", "{% note References %}\n", "\n", "{% endnote %}\n"]
            )
            start_line = len(lines) - 3
            end_line = len(lines) - 1

        # Parse existing references
        existing_refs = self.parse_all_references(lines)

        # Determine the index to use
        if ref_index is None:
            ref_index = max(existing_refs.keys(), default=0) + 1
        else:
            # Shift existing references if index already exists
            if ref_index in existing_refs:
                new_refs = {}
                for idx, ref_info in existing_refs.items():
                    new_idx = idx + 1 if idx >= ref_index else idx
                    new_refs[new_idx] = ref_info
                existing_refs = new_refs

        # Generate UUID
        ref_uuid = str(uuid.uuid4()).replace("-", "")[:10]

        # Use URL as name if name not provided
        if ref_name is None:
            ref_name = ref_url

        # Rebuild references section
        new_ref_lines = []
        all_indices = sorted(list(existing_refs.keys()) + [ref_index])

        for idx in all_indices:
            if idx == ref_index:
                new_ref_lines.append(
                    f"{idx}. [{ref_name}]({ref_url}) <!-- {ref_uuid} -->\n"
                )
            else:
                # Find the old reference and preserve UUID if it exists
                old_ref = existing_refs[idx]
                if old_ref.get("uuid"):
                    # Keep existing UUID
                    old_uuid = old_ref["uuid"]
                else:
                    # Generate new UUID for references without UUID
                    old_uuid = str(uuid.uuid4()).replace("-", "")[:10]
                new_ref_lines.append(
                    f"{idx}. [{old_ref['name']}]({old_ref['url']}) <!-- {old_uuid} -->\n"
                )

        # Replace the references section
        new_lines = (
            lines[: start_line + 1] + ["\n"] + new_ref_lines + ["\n"] + lines[end_line:]
        )

        self.write_post_file(post_name, new_lines)
        logger.info(
            f"Added reference {ref_index}: [{ref_name}]({ref_url}) with UUID {ref_uuid}"
        )

    def run_conversion(self, post_name: str, reverse: bool = False):
        """Convert [REF: UUID] patterns to LaTeX-style references or vice versa."""
        lines = self.read_post_file(post_name)
        refs = self.parse_references_with_uuid(lines)

        if not refs:
            logger.error("No references with UUIDs found")
            return

        if reverse:
            # Reverse conversion: LaTeX-style to [REF: UUID]
            self._reverse_conversion(lines, refs, post_name)
        else:
            # Forward conversion: [REF: UUID] to LaTeX-style
            self._forward_conversion(lines, refs, post_name)

    def _forward_conversion(
        self, lines: List[str], refs: Dict[str, Dict[str, str]], post_name: str
    ):
        """Convert [REF: UUID] patterns to LaTeX-style references."""
        # Process each line
        for i, line in enumerate(lines):
            # Find [REF: UUID1, UUID2, ...] patterns
            pattern = r"\[REF:\s*([^\]]+)\]"

            def replace_ref(match):
                uuid_list = [uuid.strip() for uuid in match.group(1).split(",")]
                templates = []

                for ref_uuid in uuid_list:
                    if ref_uuid in refs:
                        ref_info = refs[ref_uuid]
                        template = TEMPLATE.replace(
                            "{REFERENCE_INDEX}", ref_info["index"]
                        )
                        template = template.replace("REFERENCE_URL", ref_info["url"])
                        templates.append(template)
                    else:
                        logger.warning(f"UUID {ref_uuid} not found in references")
                        templates.append(f"[MISSING: {ref_uuid}]")

                if templates:
                    return PREFIX + JOIN.join(templates) + POSTFIX
                return match.group(0)

            lines[i] = re.sub(pattern, replace_ref, line)

        self.write_post_file(post_name, lines, suffix="-ref")
        logger.info(f"Conversion completed: {post_name}-ref.md created")

    def _reverse_conversion(
        self, lines: List[str], refs: Dict[str, Dict[str, str]], post_name: str
    ):
        """Convert LaTeX-style references back to [REF: UUID] patterns."""
        # Build reverse lookup: index -> uuid
        index_to_uuid = {ref_info["index"]: uuid for uuid, ref_info in refs.items()}

        # Process each line
        for i, line in enumerate(lines):
            # Find LaTeX-style reference patterns
            latex_pattern = re.escape(PREFIX) + r"(.*?)" + re.escape(POSTFIX)

            def replace_latex(match):
                content = match.group(1)
                # Split by JOIN pattern
                templates = content.split(JOIN)
                uuid_list = []

                for template in templates:
                    # Extract index from template
                    template_pattern = re.escape(TEMPLATE)
                    template_pattern = template_pattern.replace(
                        r"\{REFERENCE_INDEX\}", r"(\d+)"
                    )
                    template_pattern = template_pattern.replace(
                        "REFERENCE_URL", r"([^)]+)"
                    )

                    template_match = re.match(template_pattern, template.strip())
                    if template_match:
                        found_index = template_match.group(1)
                        if found_index in index_to_uuid:
                            uuid_list.append(index_to_uuid[found_index])
                        else:
                            logger.warning(
                                f"Index {found_index} not found in references"
                            )

                if uuid_list:
                    return f"[REF: {', '.join(uuid_list)}]"
                return match.group(0)

            lines[i] = re.sub(latex_pattern, replace_latex, line)

        self.write_post_file(post_name, lines, suffix="-rev")
        logger.info(f"Reverse conversion completed: {post_name}-rev.md created")

    def validate_references(self, post_name: str):
        """Validate that LaTeX-style references match the References section."""
        lines = self.read_post_file(post_name)
        refs = self.parse_references_with_uuid(lines)

        if not refs:
            logger.error("No references with UUIDs found")
            return

        errors_found = False

        for line_num, line in enumerate(lines, 1):
            # Find LaTeX-style reference patterns
            latex_pattern = re.escape(PREFIX) + r"(.*?)" + re.escape(POSTFIX)
            matches = re.finditer(latex_pattern, line)

            for match in matches:
                content = match.group(1)
                # Split by JOIN pattern
                templates = content.split(JOIN)

                for template in templates:
                    # Extract index and URL from template
                    template_pattern = re.escape(TEMPLATE)
                    template_pattern = template_pattern.replace(
                        r"\{REFERENCE_INDEX\}", r"(\d+)"
                    )
                    template_pattern = template_pattern.replace(
                        "REFERENCE_URL", r"([^)]+)"
                    )

                    template_match = re.match(template_pattern, template.strip())
                    if template_match:
                        found_index = template_match.group(1)
                        found_url = template_match.group(2)

                        # Check if this index exists in references
                        matching_uuid = None
                        for uuid, ref_info in refs.items():
                            if ref_info["index"] == found_index:
                                matching_uuid = uuid
                                break

                        if matching_uuid:
                            expected_url = refs[matching_uuid]["url"]
                            if found_url != expected_url:
                                logger.error(
                                    f"Line {line_num}: URL mismatch for index {found_index}"
                                )
                                logger.error(f"  Found: {found_url}")
                                logger.error(f"  Expected: {expected_url}")
                                errors_found = True
                        else:
                            logger.error(
                                f"Line {line_num}: Index {found_index} not found in references"
                            )
                            errors_found = True

        if not errors_found:
            logger.info("All references validated successfully")


def main():
    parser = argparse.ArgumentParser(description="Reference Control CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add command
    add_parser = subparsers.add_parser("add", help="Add a reference")
    add_parser.add_argument("post_name", help="Post name (without .md extension)")
    add_parser.add_argument("ref_url", help="Reference URL")
    add_parser.add_argument("--name", dest="ref_name", help="Reference name")
    add_parser.add_argument("--idx", dest="ref_index", type=int, help="Reference index")

    # Run command
    run_parser = subparsers.add_parser(
        "run", help="Convert REF patterns to LaTeX style"
    )
    run_parser.add_argument("post_name", help="Post name (without .md extension)")
    run_parser.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse conversion: LaTeX style to [REF: UUID]",
    )

    # Val command
    val_parser = subparsers.add_parser("val", help="Validate references")
    val_parser.add_argument("post_name", help="Post name (without .md extension)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    refctl = RefCtl("/home/zerohertz/Zerohertz/blog/source/_posts")

    if args.command == "add":
        refctl.add_reference(
            args.post_name, args.ref_url, args.ref_name, args.ref_index
        )
    elif args.command == "run":
        refctl.run_conversion(args.post_name, args.reverse)
    elif args.command == "val":
        refctl.validate_references(args.post_name)


if __name__ == "__main__":
    main()
