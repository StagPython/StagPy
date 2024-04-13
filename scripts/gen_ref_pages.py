"""Generate API reference documentation."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
src = root

for path in sorted((src / "stagpy").rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")

    parts = module_path.parts

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")

    full_doc_path = Path("reference", doc_path)

    if any(p.startswith("_") for p in parts):
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        print("::: " + identifier, file=fd)
        if parts[-1] == "config":
            print("    options:", file=fd)
            print("      show_if_no_docstring: true", file=fd)
            print('      filters: ["!^_"]', file=fd)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
