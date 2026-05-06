import re
from pathlib import Path

import griffe
import griffe2md


def generate_docs(search_path: str) -> str:
    package = griffe.load(
        next(Path(search_path).iterdir()).name,
        search_paths=[search_path],
        docstring_parser="google",
    )
    config = {
        **griffe2md.default_config,
        "summary": False,
        "show_root_full_path": False,
        "show_root_members_full_path": False,
        "show_object_full_path": False,
    }
    return griffe2md.render_object_docs(package, config)  # ty: ignore[invalid-argument-type]


def inject_into_readme(search_path: str, readme: str) -> None:
    """Replace content between griffe markers in the README."""
    docs = generate_docs(search_path)
    content = Path(readme).read_text(encoding="utf-8")
    pattern = r"(<!-- griffe -->)(.*?)(<!-- /griffe -->)"
    replacement = rf"\1\n{docs}\n\3"
    new_content, count = re.subn(pattern, replacement, content, flags=re.DOTALL)
    if count == 0:
        raise RuntimeError("Markers not found in README. Add:\n<!-- griffe -->\n<!-- /griffe -->")
    Path(readme).write_text(new_content, encoding="utf-8")


if __name__ == "__main__":
    inject_into_readme("./src", "./README.md")
