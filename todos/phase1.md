Issues in order of fixing:

- issue 1 (multiple validation pathways):
  code refactor for population/. make workflow.md for population/ current flow. seems like there are multiple validation pathways - quick_validate.py, validator/\*.py, and hydrator_utils has its own validation functions. also validator/**init** has content in it (functions etc - not consistent with the rest of the codebase). it also uses @dataclass when we use pydantic. first propose a consolidated method to validate. there are semantic, syntactic errors at a broader level. validation should ensure that the spec syntax is consistent, formulas, modifiers, etc are correct - both at sub-phase and end-phase stage of population creation. a consolidated approach must be explored. keep code as simple and readable as possible.

  sub-issue-a: refer docs/phase1-currentworkflow.md step 3
  if there is an issue with bind_constraints() entire thing fails.

  Currently: If bind_constraints()
  raises CircularDependencyError, the CLI: Prints the error message
  Exits with code 1
  All progress is lost — no YAML saved, no recovery
  The problem: By this point you've already:

  Completed sufficiency check
  Waited for attribute selection (LLM call)
  Waited for hydration Steps 2a-2d (4+ LLM calls with web search)
  All that work is wasted if there's a circular dependency.

  Options for improvement:

  Detect earlier — Check for cycles after Step 1 (selection) before doing expensive hydration
  Save partial progress — Dump intermediate state to allow resumption
  Auto-fix — Try to break the cycle by removing one dependency edge
  Would you like me to implement one of these? Detecting cycles earlier (option 1) would be the quickest win.

  sub-issue-b: validate_spec() failure loses work

  After build_spec() completes, the spec exists in memory. If validate_spec() finds structural errors, the CLI displays them and exits without saving. The user has waited through 5+ LLM calls, but the resulting spec is discarded.

  Proposed fix: Save the spec with an .invalid.yaml suffix when validation fails. This allows the user to inspect and manually fix issues in the YAML rather than re-running the entire pipeline. The distinct suffix signals the file is not production-ready. NOTE that most plausible cases to cause failure are prevented by llm-fast-fail-checks leave edge cases.

  sub-issue-c: Move condition value validation to LLM fail-fast, delete fixer.py

  Currently, validate_modifiers_response() during Step 2d only checks syntax, not whether string literals in when conditions match valid categorical options. This semantic check only happens during final validate_spec() — too late for LLM retry.

  Proposed fix:

  Add condition value validation to validate_modifiers_response() pass the categorical options map and verify each string literal in when clauses matches a valid option. On mismatch, return an ERROR with suggestion listing valid options, triggering LLM retry. Delete fixer.py and the entropy fix CLI command — they become redundant once the LLM can fix typos via retry feedback.

- issue 2 (schema inconsistency use and prompts):
  what is the difference between binder build spec vs build schema() functions in hydrator_utils(). also, selector, sufficiency.py and others have schema files.

- issue 3 (context section)
  the way context section is built in selector.py, hydrator.py is un-readable and does not follow proper conventions.
