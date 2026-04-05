Update the CLAUDE.md documentation files to reflect any recent code changes in this repository.

## Instructions

1. **Identify what changed**: Review git status and recent commits on the current branch (`git log main..HEAD --name-only`) to understand what was added or modified.

2. **Update affected CLAUDE.md files**:
   - `/CLAUDE.md` — top-level architecture overview, service list, frontend section, access URLs, dependencies
   - `/src/history_book/api/CLAUDE.md` — API endpoints and DTOs
   - `/src/history_book/services/CLAUDE.md` — service-layer classes and their responsibilities
   - `/frontend/CLAUDE.md` — frontend structure, pages, components, state management
   - Subsystem CLAUDE.md files (chains, database, evals, agents) if their code changed

3. **What to document**:
   - New services, components, pages, API endpoints
   - Architecture patterns and data flow
   - Non-obvious implementation decisions (e.g., why MultiDiGraph instead of DiGraph)
   - Practical usage: CLI commands, curl examples, access URLs
   - Known gotchas or constraints relevant to future work

4. **Standards**:
   - Every line must add value — omit what can be inferred from reading the code
   - Prefer tables and short bullet points over prose
   - Don't duplicate content across files — link between them instead
   - Update existing sections in place rather than appending redundant new sections
   - Remove or correct stale content if the implementation changed
