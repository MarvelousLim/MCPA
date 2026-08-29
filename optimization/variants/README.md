# Variant policy

The benchmark exposes `reference` (a frozen, readable oracle) and `candidate`
(the production `equilibrate` wrapper). Optimization names live in
`policies.tsv`; their implementation is a commit or a small policy/template
switch in production code, never a copied source tree.

Use one branch or commit series per experiment. Put the commit hash and preset in
the JSONL ledger through `../run.sh`. If two policies must coexist temporarily,
prefer a narrowly scoped compile definition selected by CMake and remove it after
the decision.
