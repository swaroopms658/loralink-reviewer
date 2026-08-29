# Beginner's Playbook: CLAUDE.md + GitHub Spec Kit
### For the project: **Agentic Orchestrator for Software Testing**

You've never written a `CLAUDE.md` and don't know what goes into a Spec Kit. This guide fixes both. Read top to bottom once, then follow the **"Do this now"** boxes in order. Every prompt you need to paste is written out for you.

---

## 0. What you are building (in plain words)

An **agentic orchestrator for software testing** is a system where AI agents *autonomously plan, write, run, and triage software tests* instead of a human doing it manually. Think of one "manager" agent (the **orchestrator**) that coordinates several "worker" agents:

| Agent | Job |
|-------|-----|
| **Planner** | Reads the code/requirements, decides *what* to test |
| **Test Generator** | Writes the actual test cases (e.g. pytest / Jest) |
| **Executor / Runner** | Runs the tests in a sandbox, captures output |
| **Triage / Analyzer** | Reads failures, decides if it's a real bug or a flaky test |
| **Reporter** | Summarizes results for a human |

Keep this table handy — you will paste it into prompts later, because the AI produces far better specs when it knows the shape of your system.

---

## 1. Two tools, two different jobs

Beginners confuse these. They are **not** the same thing.

| | **CLAUDE.md** | **Spec Kit** |
|---|---|---|
| What it is | A single memory file Claude auto-reads every session | A folder of Markdown files that describe *what to build* before coding |
| Purpose | Tells Claude the rules, commands, and conventions of *your repo* | Turns a vague idea into a spec → plan → task list → code |
| When used | Every time you open Claude Code in this project | Once per feature, at the start |
| Analogy | The "house rules" pinned on the fridge | The "blueprint" before construction |

You want **both**. CLAUDE.md keeps Claude consistent; Spec Kit keeps *you* from building the wrong thing.

---

## PART A — Writing your CLAUDE.md

### A1. What actually goes in a CLAUDE.md

A good `CLAUDE.md` is short and factual. It answers questions Claude would otherwise guess wrong. Sections that matter:

1. **Project overview** — one paragraph: what this is, who uses it.
2. **Architecture** — the main components and how they talk.
3. **Build / run / test commands** — the exact terminal commands.
4. **Conventions** — language, style, folder layout, naming.
5. **Guardrails** — things Claude must *never* do (e.g. "never run generated tests outside the sandbox").
6. **Glossary** — project-specific words ("orchestrator", "triage agent").

> **Rule of thumb:** if a new human teammate would need to be *told* it, put it in CLAUDE.md. If they could read it from the code in 5 seconds, leave it out.

### A2. A ready-to-use starter CLAUDE.md (written in two passes)

CLAUDE.md is a **living file**, not a one-shot. You cannot honestly fill in build commands or the folder layout at the very start — those are decided later by `/speckit.plan`. So write it in two passes (see the flow in Section 2):

- **PASS 1 (now):** fill only what you already know — *Project overview*, *Guardrails*, *Glossary*. Leave *Architecture* and *Build / run / test* as TODO.
- **PASS 2 (after `/speckit.plan`):** replace those TODOs with the real layout and commands the plan defines.

Copy this into a file named `CLAUDE.md` at the repo root. Sections are tagged so you know which pass fills them.

```markdown
# CLAUDE.md — Agentic Orchestrator for Software Testing

## Project overview            <!-- PASS 1: you know this now -->
This project is an agentic orchestrator that automates software testing.
A central orchestrator coordinates specialized LLM agents (planner, test
generator, executor, triage, reporter) to plan, generate, run, and analyze
tests for a target codebase with minimal human input.

## Architecture               <!-- PASS 2: fill after /speckit.plan defines the real layout; leave as TODO for now -->
- `orchestrator/` — central controller; routes work between agents, holds run state.
- `agents/` — one module per agent role (planner, generator, executor, triage, reporter).
- `runners/` — sandboxed test execution (subprocess/container) for pytest, Jest, etc.
- `integrations/` — CI hooks, VCS (git) access, framework adapters.
- `core/` — shared types, message schemas, config.
Agents communicate via [structured messages / a message bus — DECIDE LATER].

## Build / run / test          <!-- PASS 2: fill after /speckit.plan picks the stack; leave as TODO for now -->
- Install: `[e.g. uv pip install -e .]`
- Run orchestrator: `[e.g. python -m orchestrator run --target ./sample_repo]`
- Run the project's own tests: `[e.g. pytest]`
- Lint / format: `[e.g. ruff check . && ruff format .]`

## Conventions                 <!-- PASS 2: confirm once the plan sets language & structure -->
- Language: [Python 3.12 — CONFIRM].
- All agent inputs/outputs are typed schemas (no free-form dict passing).
- Every agent call must be logged with a run id and cost estimate.
- Tests live in `tests/`, mirroring the source folder structure.

## Guardrails (do NOT violate) <!-- PASS 1: you know these now -->
<!-- keep in sync with .specify/memory/constitution.md -->

- NEVER execute AI-generated tests outside the sandboxed runner.
- NEVER commit API keys or real credentials; use `.env` (git-ignored).
- Do not add a new agent role without updating the spec first.
- Prefer deterministic logic over an LLM call when a rule suffices.

## Glossary                    <!-- PASS 1: you know these now -->
- **Orchestrator**: the top-level coordinator agent.
- **Triage agent**: decides whether a test failure is a real bug or flaky.
- **Run**: one full plan→generate→execute→triage→report cycle.
```

### A3. The prompt to give Claude to generate/refine it

If you'd rather have Claude build the CLAUDE.md by interviewing you, paste this:

> **Prompt to paste:**
> "I'm building an agentic orchestrator for software testing (a central orchestrator that coordinates planner, test-generator, executor, triage, and reporter agents). Help me write a `CLAUDE.md` for this repo. Ask me up to 5 questions about my tech stack, folder layout, build commands, and guardrails, then generate the file. Keep it factual and under one page."

Answer its questions, and you'll get a tailored file. Save it as `CLAUDE.md` in the repo root. If you use this route, tell Claude to only fill the PASS 1 sections and leave Architecture / Build-run-test as TODO.

> **Do this now (PASS 1):**
> 1. Create `CLAUDE.md` from the starter above **or** run the prompt in A3.
> 2. Fill only the **PASS 1** sections: Project overview, Guardrails, Glossary. Leave Architecture and Build/run/test as TODO — you don't know them yet.
> 3. Commit it: `git add CLAUDE.md && git commit -m "Add CLAUDE.md stub"`.
>
> **Later (PASS 2), after `/speckit.plan`:** come back and replace the Architecture and Build/run/test TODOs with the real layout and commands the plan defines, then re-commit. This is the `★ PASS 2` touch-point in the Section 2 flow.

---

## PART B — Setting up the Spec Kit

Spec Kit enforces **spec-driven development**: you describe the *what* and *why* before any code. It gives you slash commands that produce a chain of Markdown files.

### B1. The Spec Kit workflow (the big picture)

```
/constitution  →  /specify  →  /plan  →  /tasks  →  /implement
   (rules)        (the what)    (the how)  (to-do list) (build it)
```

Each step writes a Markdown file. You review each file before moving on — that's the whole point: catch mistakes on paper, not in code.

### B2. Install Spec Kit

You need `uv` (a Python tool runner) installed. Then, in your project folder:

```bash
# One-time init — scaffolds .specify/ templates and the slash commands
uvx --from git+https://github.com/github/spec-kit.git specify init --here
```

Pick **Claude** as your AI agent when prompted. This creates a `.specify/` folder and registers the `/constitution`, `/specify`, `/plan`, `/tasks`, `/implement` commands inside Claude Code.

> If `uvx` isn't found: install uv first (`pip install uv` or see astral.sh/uv), then re-run.

### B3. What each Markdown file is FOR (this is the part beginners miss)

| File (created by) | Answers the question | What YOU put in the prompt |
|---|---|---|
| **`constitution.md`** (`/constitution`) | "What rules govern *every* feature?" | Non-negotiable principles: testing standards, security, code quality, "agents must be sandboxed" |
| **`spec.md`** (`/specify`) | "*What* are we building and *why*?" | User stories, features, success criteria — **no tech choices** |
| **`plan.md`** (`/plan`) | "*How* will we build it technically?" | Stack, architecture, data models, agent message formats |
| **`tasks.md`** (`/tasks`) | "What are the concrete steps?" | Nothing — it's generated *from* the plan into a checklist |
| *(code)* (`/implement`) | "Build it" | Nothing — executes the task list |

Golden rule: **`spec.md` = WHAT, `plan.md` = HOW.** Don't put "use Python and Redis" in the spec; that belongs in the plan.

### B4. The exact prompts to run, in order

Run these one at a time inside Claude Code, reviewing the generated file after each.

**Step 1 — Constitution** (the rules for the whole project):

> `/constitution Create principles for an agentic software-testing orchestrator. Priorities: (1) all AI-generated tests run only inside a sandboxed executor, (2) every agent has typed inputs/outputs, (3) prefer deterministic logic over LLM calls when possible, (4) full observability — log every agent call with a run id and cost, (5) no secrets in code, (6) human-in-the-loop approval before any destructive action.`

**Step 2 — Specify** (the WHAT — notice: zero tech words):

> `/specify Build an agentic orchestrator for software testing. A user points it at a target code repository. The orchestrator coordinates five agents: a Planner that decides what to test, a Test Generator that writes test cases, an Executor that runs them safely, a Triage agent that classifies failures as real bugs vs. flaky, and a Reporter that produces a human-readable summary. Success = the system can take an untested sample repo and produce passing/failing test results with a triaged report, with a human able to approve or reject the plan before execution.`

**Step 3 — Clarify** (optional but recommended — de-risks the spec):

> `/clarify`

(Answer the questions it asks. This fills gaps before planning.)

**Step 4 — Plan** (the HOW — now you name the tech):

> `/plan Use Python 3.12. The orchestrator is a state machine coordinating agents over a typed message schema (Pydantic). Agents call the Claude API. Test execution happens in an isolated subprocess/container runner supporting pytest and Jest. Persist run state and logs to SQLite for the MVP. Expose a CLI entry point first; a web UI is out of scope for v1.`

**Step 5 — Tasks** (auto-generate the checklist):

> `/tasks`

**Step 6 — (Optional) Analyze** (cross-check consistency):

> `/analyze`

**Step 7 — Implement** (build it, task by task):

> `/implement`

> **Do this now:**
> 1. Run B2 to install.
> 2. Run Steps 1→2→(3)→4→5 above, **reading each generated file before the next step**.
> 3. Stop after `/tasks`. Review `tasks.md`. Only then run `/implement`.

---

## 2. The real flow (with review gates and loop-backs)

This is **not** a straight line. Two things make it real: a **review gate** after every step (you read the file before advancing), and **loop-backs** (when a step reveals a problem, you go back and fix the earlier file). CLAUDE.md is written in **two passes** — a stub up front, enriched after the plan exists.

**Legend:** `[GATE]` = stop and read the output before continuing · `↺` = loop back and fix · `★` = CLAUDE.md touch-point

```
                 ┌───────────────────────────────────────────────┐
   PASS 1  ★     │ Stub CLAUDE.md: overview + guardrails only     │
                 │ (build commands & layout unknown yet)          │
                 └───────────────────────┬───────────────────────┘
                                         │
                            specify init  (scaffold .specify/, choose Claude;
                                         │  git is handled here — no separate git init)
                                         ▼
                 ┌───────────────────────────────────────────────┐
             1   │ /speckit.constitution   project-wide rules      │
                 └───────────────────────┬───────────────────────┘
                                   [GATE] read it ──↺ rules wrong? edit & rerun
                                         ▼
                 ┌───────────────────────────────────────────────┐
             2   │ /speckit.specify        the WHAT (no tech)      │
                 └───────────────────────┬───────────────────────┘
                                   [GATE] read spec.md
                                         ▼
                 ┌───────────────────────────────────────────────┐
             3   │ /speckit.clarify        answer gap questions    │
                 └───────────────────────┬───────────────────────┘
                        ↺ answers expose missing/ wrong scope?
                          go back to step 2, edit spec, re-clarify
                                         ▼
                 ┌───────────────────────────────────────────────┐
             4   │ /speckit.plan           the HOW (tech choices)  │
                 └───────────────────────┬───────────────────────┘
                                   [GATE] read plan.md
                                         ▼
   PASS 2  ★     ENRICH CLAUDE.md: now that the plan defines them, add real
                 build/run/test commands + folder layout + conventions
                                         ▼
                 ┌───────────────────────────────────────────────┐
             5   │ /speckit.tasks          generated checklist     │
                 └───────────────────────┬───────────────────────┘
                                   [GATE] read tasks.md
                                         ▼
                 ┌───────────────────────────────────────────────┐
             6   │ /speckit.analyze        consistency check       │
                 └───────────────────────┬───────────────────────┘
                        ↺ mismatch across spec / plan / tasks?
                          go back to whichever file is wrong, fix, re-analyze
                                         ▼ (clean)
                 ┌───────────────────────────────────────────────┐
             7   │ /speckit.implement      build ONE chunk         │◀────┐
                 └───────────────────────┬───────────────────────┘      │
                                   run the project's tests                │
                        ↺ fail? fix, or loop to step 5 to re-task ────────┘
                                         ▼ (all chunks pass)
                                      ✅  done
```

**Why the gates matter:** a wrong `spec.md` makes every downstream file wrong. Catching it at the `[GATE]` costs one re-read; catching it at `implement` costs a rebuild.

**File locations produced:** constitution → `.specify/memory/constitution.md`; spec/plan/tasks → `specs/<feature-id>/` (e.g. `specs/001-orchestrator/spec.md`).

**CLAUDE.md vs constitution (kept in sync across both passes):** CLAUDE.md = *repo mechanics* (commands, layout); constitution = *project principles* (sandboxing, observability, human-in-the-loop). When the plan changes the mechanics, update CLAUDE.md; when a principle changes, update the constitution.

---

## 3. Common beginner mistakes (avoid these)

- **Putting tech in the spec.** "Use FastAPI" belongs in `/plan`, not `/specify`. Specs describe behavior, not implementation.
- **Skipping review.** The value is reading each `.md` before the next command. If the spec is wrong, everything downstream is wrong.
- **Over-stuffing CLAUDE.md.** It's not documentation. It's the *non-obvious* rules. Keep it under a page.
- **Vague success criteria.** "It should work" is useless. Use measurable ones: "produces a triaged report for a sample repo with ≥1 real failure detected."
- **Letting `/implement` run everything blindly.** Implement in chunks, run the project's tests between chunks.

---

## 4. What to ask me next

When you're ready, paste one of these to me:

- *"Generate the actual CLAUDE.md file for my repo"* → I'll write it to disk (I can even interview you first).
- *"Write out constitution.md / spec.md / plan.md by hand"* → if you don't want to install the CLI, I'll hand-author the Spec Kit files with your project's content already filled in.
- *"Set up the repo structure"* → I'll create the folder skeleton (`orchestrator/`, `agents/`, `runners/`, etc.) matching the plan.

Tell me which, and whether you want the **official CLI** or **hand-authored** Spec Kit files.
