"""Generate the code reference pages and navigation."""

from pathlib import Path
import mkdocs_gen_files

# Set the root directory (where opto package is located)
root = Path(__file__).parent.parent
src = root / "opto"

# Generate navigation file
nav = mkdocs_gen_files.Nav()

# Walk through the opto package and generate API pages
for path in sorted(src.rglob("*.py")):
    # Skip __pycache__ and other non-module files
    if "__pycache__" in str(path):
        continue
        
    # Get module path relative to src
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("api") / doc_path

    # Convert path to module name (e.g., trace/nodes.py -> opto.trace.nodes)
    parts = tuple(module_path.parts)
    
    if parts[-1] == "__init__":
        # For __init__.py files, use the parent directory name
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = Path("api") / doc_path
    elif parts[-1] == "__init__":
        continue
    
    # Skip empty parts
    if not parts:
        continue
        
    # Create the module identifier
    module_ident = "opto." + ".".join(parts)
    
    # Add to navigation
    nav[parts] = doc_path.as_posix()
    
    # Generate the markdown file
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        # Write the module documentation
        print(f"# {module_ident}", file=fd)
        print(f"::: {module_ident}", file=fd)
    
    # Set the edit path for GitHub integration
    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

# Write the navigation file
with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())