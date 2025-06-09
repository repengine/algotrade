import os
import re
from collections import defaultdict


def analyze_codebase():
    errors = defaultdict(lambda: defaultdict(int))

    # Patterns to check
    patterns = {
        "missing_type_annotation": r"def\s+\w+\s*\([^)]*\)\s*(?\!->)",
        "untyped_parameter": r"def\s+\w+\s*\([^:)]+\)",
        "bare_except": r"except\s*:",
        "wildcard_import": r"from\s+\S+\s+import\s+\*",
        "long_line": r".{121,}",
        "missing_docstring": r'def\s+\w+.*:\s*\n\s*(?\!["\'])',
        "unused_variable": r"^\s*(\w+)\s*=.*(?\!.*\1)",
        "todo_fixme": r"#\s*(TODO < /dev/null | FIXME)",
    }

    for root, dirs, files in os.walk("."):
        # Skip virtual environments and archive
        dirs[:] = [
            d
            for d in dirs
            if d not in ["venv", "test_venv", "__pycache__", "archive", ".git"]
        ]

        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                module = root.replace("./", "").replace("/", ".")

                try:
                    with open(filepath, encoding="utf-8") as f:
                        content = f.read()
                        lines = content.split("\n")

                    # Check each pattern
                    for error_type, pattern in patterns.items():
                        if error_type == "long_line":
                            for line in lines:
                                if len(line) > 120:
                                    errors[module][error_type] += 1
                        else:
                            matches = re.findall(pattern, content, re.MULTILINE)
                            errors[module][error_type] += len(matches)

                except Exception as e:
                    print(f"Error reading {filepath}: {e}")

    return errors


# Run analysis
errors = analyze_codebase()

# Summary by error type
print("=== ERROR SUMMARY BY TYPE ===")
total_by_type = defaultdict(int)
for module, module_errors in errors.items():
    for error_type, count in module_errors.items():
        total_by_type[error_type] += count

for error_type, count in sorted(
    total_by_type.items(), key=lambda x: x[1], reverse=True
):
    print(f"{error_type}: {count}")

print("\n=== TOP 10 FILES WITH MOST ERRORS ===")
file_totals = []
for module, module_errors in errors.items():
    total = sum(module_errors.values())
    if total > 0:
        file_totals.append((module, total, module_errors))

for module, total, module_errors in sorted(
    file_totals, key=lambda x: x[1], reverse=True
)[:10]:
    print(f"\n{module}: {total} errors")
    for error_type, count in sorted(
        module_errors.items(), key=lambda x: x[1], reverse=True
    ):
        if count > 0:
            print(f"  - {error_type}: {count}")

print("\n=== ERROR COUNT BY MODULE ===")
module_summary = defaultdict(int)
for module, module_errors in errors.items():
    base_module = module.split(".")[0] if "." in module else module
    module_summary[base_module] += sum(module_errors.values())

for module, count in sorted(module_summary.items(), key=lambda x: x[1], reverse=True):
    print(f"{module}: {count}")
