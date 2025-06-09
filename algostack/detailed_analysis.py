import ast
import os
import re


def analyze_imports(filepath):
    issues = []
    try:
        with open(filepath, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filepath)

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")

        # Check for common import issues
        if "typing" in str(imports) and "from __future__ import annotations" not in str(
            imports
        ):
            issues.append("missing_future_annotations")

    except:
        pass
    return issues


# Check type annotation coverage
def check_type_coverage(filepath):
    stats = {
        "functions_total": 0,
        "functions_typed": 0,
        "classes_total": 0,
        "missing_annotations": [],
    }

    try:
        with open(filepath, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filepath)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                stats["functions_total"] += 1
                if node.returns:
                    stats["functions_typed"] += 1
                else:
                    stats["missing_annotations"].append(node.name)
            elif isinstance(node, ast.ClassDef):
                stats["classes_total"] += 1

    except:
        pass

    return stats


# Analyze key files
key_files = [
    "./core/backtest_engine.py",
    "./core/portfolio.py",
    "./core/live_engine.py",
    "./strategies/base.py",
    "./api/app.py",
    "./dashboard.py",
]

print("=== TYPE ANNOTATION COVERAGE ===")
total_functions = 0
total_typed = 0

for filepath in key_files:
    if os.path.exists(filepath):
        stats = check_type_coverage(filepath)
        coverage = (
            (stats["functions_typed"] / stats["functions_total"] * 100)
            if stats["functions_total"] > 0
            else 0
        )
        print(f"\n{filepath}:")
        print(
            f"  Functions: {stats['functions_typed']}/{stats['functions_total']} ({coverage:.1f}% typed)"
        )
        if stats["missing_annotations"][:5]:
            print(
                f"  Missing annotations: {', '.join(stats['missing_annotations'][:5])}"
            )
        total_functions += stats["functions_total"]
        total_typed += stats["functions_typed"]

overall_coverage = (total_typed / total_functions * 100) if total_functions > 0 else 0
print(f"\nOverall: {total_typed}/{total_functions} ({overall_coverage:.1f}% typed)")

# Check for common type issues
print("\n=== COMMON TYPE ISSUES ===")
type_issues = {
    "any_usage": 0,
    "dict_instead_of_typing": 0,
    "list_instead_of_typing": 0,
    "optional_misuse": 0,
}

for root, dirs, files in os.walk("."):
    dirs[:] = [
        d
        for d in dirs
        if d not in ["venv", "test_venv", "__pycache__", "archive", ".git"]
    ]

    for file in files:
        if file.endswith(".py"):
            filepath = os.path.join(root, file)
            try:
                with open(filepath) as f:
                    content = f.read()

                type_issues["any_usage"] += len(
                    re.findall(r": Any(?:\s < /dev/null | ,|\))", content)
                )
                type_issues["dict_instead_of_typing"] += len(
                    re.findall(r": dict(?:\s|,|\))", content)
                )
                type_issues["list_instead_of_typing"] += len(
                    re.findall(r": list(?:\s|,|\))", content)
                )

            except:
                pass

for issue, count in type_issues.items():
    print(f"{issue}: {count}")
