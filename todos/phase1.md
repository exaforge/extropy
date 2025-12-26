Issues in order of fixing:

- issue 1 (multiple validation pathways):
  code refactor for population/. make workflow.md for population/ current flow. seems like there are multiple validation pathways - quick_validate.py, validator/\*.py, and hydrator_utils has its own validation functions. also validator/**init** has content in it (functions etc - not consistent with the rest of the codebase). it also uses @dataclass when we use pydantic. first propose a consolidated method to validate. there are semantic, syntactic errors at a broader level. validation should ensure that the spec syntax is consistent, formulas, modifiers, etc are correct - both at sub-phase and end-phase stage of population creation. a consolidated approach must be explored. keep code as simple and readable as possible.

- issue 2 (schema inconsistency use and prompts):
  what is the difference between binder build spec vs build schema() functions in hydrator_utils(). also, selector, sufficiency.py and others have schema files.

- issue 3 (context section)
  the way context section is built in selector.py, hydrator.py is un-readable and does not follow proper conventions.
