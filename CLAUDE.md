# Claude Development Guidelines for AlgoStack

## Test Design Reference
For comprehensive test design patterns and implementation guidelines, see:
- **Quick Reference**: `docs/planning/test-design-index.md` (start here)
- **Full Documentation**: `docs/planning/test-design-overhaul-plan.md` (detailed examples)

## CRITICAL: Prevent Module Bloat

**ALWAYS follow these rules when working on AlgoStack:**

1. **SEARCH FIRST**: Before creating ANY new module or file:
   - Use Glob to find similar filenames
   - Use Grep to search for similar functionality
   - Read existing modules to understand current implementations
   - Use Context7 MCP server to search for up-to-date library information. Remember, you were trained on old data, so you need to find new data first.

2. **EXTEND, DON'T DUPLICATE**: 
   - Prefer modifying existing code over creating new files
   - Add functions to existing modules rather than creating new ones
   - Extend existing classes instead of creating similar new ones

3. **VERIFY BEFORE CREATING**:
   - Check if functionality already exists (even partially)
   - Confirm no similar code exists in the codebase
   - Ask user for approval before creating new modules

5. **MODULE CREATION CHECKLIST**:
   - [ ] Searched for similar functionality with Grep
   - [ ] Checked existing modules with Glob
   - [ ] Read related files to understand current structure
   - [ ] Confirmed this is genuinely new functionality
   - [ ] Got explicit approval from user

**Remember**: The codebase is already feature-rich. Your job is to fix, refine, and integrate - not to recreate what already exists.