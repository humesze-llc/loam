# Agent guidance

For AI agents and humans reviewing their output. Review gates enforce
these items; violations are defects, not style preferences.

## Audience

Contributors are fluent in Rust, geometry, and graphics. Write for
them. Do not explain standard idioms, textbook math, or what a line
plainly does.

## Comments

Comment only when absolutely necessary. A comment earns its place by
stating something the code cannot: a non-obvious invariant, a numerical
conditioning rationale, a citation, a cross-module contract, or why the
plausible alternative loses.

- Identifiers carry WHAT; comments carry WHY. If a name can say it,
  say it with the name.
- No narration, no restating signatures or types, no comments naming
  the current task, milestone, or PR.
- Public doc comments: one tight sentence unless the contract needs
  more.
- Wrap near 80 columns. No em-dashes, decorative arrows, or emoji.
- History lives in git, never in comments.

## Prose

Applies to README, public docs, and error messages. State facts. No
hype, no idiom that costs precision, no self-congratulation. A claim
carries its evidence (test, measurement, citation) or gets cut.

## Scope

One purpose per branch. No ridealong refactors, no goldplating, no
defensive abstraction. Fix root causes, not symptoms.
