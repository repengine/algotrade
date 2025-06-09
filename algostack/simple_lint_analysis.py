import os
import re


def analyze_file(filepath):
    errors = {
        "missing_return_type": 0,
        "missing_param_type": 0,
        "bare_except": 0,
        "long_lines": 0,
        "todo_fixme": 0,
    }

    try:
        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()

        for _i, line in enumerate(lines):
            # Check for functions without return type
            if re.match(r"^\s*def\s+[^_]\w*\s*\([^)]*\)\s*:\s*$", line):
                errors["missing_return_type"] += 1

            # Check for bare except
            if re.match(r"^\s*except\s*:\s*$", line):
                errors["bare_except"] += 1

            # Check for long lines
            if len(line.rstrip()) > 120:
                errors["long_lines"] += 1

            # Check for TODO/FIXME
            if "TODO" in line or "FIXME" in line:
                errors["todo_fixme"] += 1

    except Exception:
        pass

    return errors


# Analyze all Python files
total_errors = {
    "missing_return_type": 0,
    "missing_param_type": 0,
    "bare_except": 0,
    "long_lines": 0,
    "todo_fixme": 0,
}

file_errors = {}

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
            errors = analyze_file(filepath)

            file_total = sum(errors.values())
            if file_total > 0:
                file_errors[filepath] = (file_total, errors)

            for error_type, count in errors.items():
                total_errors[error_type] += count

print("=== TOTAL ERRORS BY TYPE ===")
for error_type, count in sorted(total_errors.items(), key=lambda x: x[1], reverse=True):
    print(f"{error_type}: {count}")

print("\n=== TOP 10 FILES WITH MOST ERRORS ===")
sorted_files = sorted(file_errors.items(), key=lambda x: x[1][0], reverse=True)[:10]
for filepath, (total, errors) in sorted_files:
    print(f"\n{filepath}: {total} errors")
    for error_type, count in errors.items():
        if count > 0:
            print(f"  - {error_type}: {count}")

# Count by module
module_counts = {}
for filepath, (total, errors) in file_errors.items():
    module = filepath.split("/")[1] if len(filepath.split("/")) > 1 else "root"
    module_counts[module] = module_counts.get(module, 0) + total

print("\n=== ERROR COUNT BY MODULE ===")
for module, count in sorted(module_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{module}: {count}")
