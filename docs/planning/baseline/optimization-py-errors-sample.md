# Sample Errors from optimization.py

## Pylance Type Errors (13 in this file)

1. **Line 82**: ArgumentType error
   - `Argument of type "ArrayLike | Unknown | NDArray[Any]" cannot be assigned to parameter "f" of type "ArrayLike"`
   - Fix: Add explicit type conversion with `np.asarray()`

2. **Line 138**: GeneralTypeIssues
   - `"Literal[1]" is not iterable`
   - Fix: Ensure proper unpacking of optimization results

3. **Line 184**: OperatorIssue
   - `Operator "-" not supported for types "ArrayLike" and "float"`
   - Fix: Ensure arrays are proper numpy arrays

4. **Line 250**: CallIssue
   - `No overloads for "__getitem__" match the provided arguments`
   - Fix: Convert list to proper type for DataFrame indexing

5. **Line 254**: ArgumentType
   - `Argument of type "Scalar" cannot be assigned to parameter "best_value" of type "float"`
   - Fix: Explicit float conversion

6. **Line 308**: AssignmentType
   - `Type "int | None" is not assignable to declared type "int"`
   - Fix: Add proper None handling

7. **Line 454**: ArgumentType
   - `Argument of type "list[float | None] | None" cannot be assigned to parameter "convergence_history"`
   - Fix: Filter out None values or change type annotation

8. **Line 482**: ReturnType
   - `Type "floating[Any] | float64 | float" is not assignable to return type "float"`
   - Fix: Explicit float conversion

9. **Line 546**: CallIssue & ArgumentType
   - Index access issues with pandas
   - Fix: Proper type handling for index access

10. **Line 577**: ReturnType
    - `Type "floating[Any] | Literal[0]" is not assignable to return type "float"`
    - Fix: Explicit type conversion

## Ruff Linting Errors (140+ in this file)

### Whitespace Issues (100+ occurrences)
- W293: Blank line contains whitespace (majority)
- W291: Trailing whitespace (5 occurrences)
- W292: No newline at end of file (1 occurrence)

### Import Issues
- I001: Import block is un-sorted or un-formatted (2 blocks)
- UP035: Deprecated imports (Dict, List, Tuple)
- UP006: Use built-in types instead of typing (6 occurrences)

### Code Issues
- F541: f-string without any placeholders (2 occurrences)
- B007: Loop control variable not used within loop body (1 occurrence)

### Sourcery Suggestions (code quality)
- Multiple refactoring suggestions for cleaner code

## Pattern Analysis

### Common Type Error Patterns:
1. **Numpy/Pandas type compatibility**: Many errors involve numpy arrays not being properly typed
2. **Missing explicit conversions**: Return values need explicit float() conversions
3. **Optional handling**: Need to handle None cases explicitly
4. **Generic types**: Still using old-style Dict, List instead of dict, list

### Common Linting Patterns:
1. **Whitespace everywhere**: Almost every function has whitespace issues
2. **Old-style imports**: Still using typing.Dict instead of dict
3. **No formatting applied**: Code hasn't been run through black/autopep8

## Estimated Total Errors Across Codebase

If optimization.py (~800 lines) has 140+ linting errors and 13 type errors:
- Average: ~0.175 linting errors per line
- Average: ~0.016 type errors per line

With ~30,000 lines of Python code:
- Estimated total linting errors: 5,250
- Estimated total type errors: 480

This suggests our initial estimates were conservative!