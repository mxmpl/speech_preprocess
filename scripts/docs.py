import re
import tempfile
from pathlib import Path

from pydoc_markdown import PydocMarkdown
from pydoc_markdown.contrib.loaders.python import PythonLoader
from pydoc_markdown.contrib.renderers.markdown import MarkdownRenderer


def generate_docs(search_path: str) -> str:
    with tempfile.NamedTemporaryFile() as tmp:
        session = PydocMarkdown(
            loaders=[PythonLoader(search_path=[search_path])],
            renderer=MarkdownRenderer(
                filename=tmp.name,
                render_module_header=False,
                insert_header_anchors=False,
                header_level_by_type={"Function": 3},
            ),
        )
        modules = session.load_modules()
        session.process(modules)
        session.render(modules)
        return Path(tmp.name).read_text(encoding="utf-8")


def inject_into_readme(search_path: str, readme: str) -> None:
    """Replace content between pydoc-markdown markers in the README."""
    docs = generate_docs(search_path)
    content = Path(readme).read_text(encoding="utf-8")
    pattern = r"(<!-- pydoc-markdown -->)(.*?)(<!-- /pydoc-markdown -->)"
    replacement = rf"\1\n{docs}\n\3"
    new_content, count = re.subn(pattern, replacement, content, flags=re.DOTALL)
    if count == 0:
        raise RuntimeError("Markers not found in README. Add:\n<!-- pydoc-markdown -->\n<!-- /pydoc-markdown -->")
    Path(readme).write_text(new_content, encoding="utf-8")


if __name__ == "__main__":
    inject_into_readme("./src", "./README.md")
