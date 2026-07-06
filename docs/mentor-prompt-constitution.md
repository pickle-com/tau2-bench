# Tau Voice Tool-Mentor Prompt Constitution

Status: adopted 2026-07-02, incorporating revisions through 2026-07-04
(project decisions). Binding for every change to
mentor-facing prompt text and mentor wiring in this fork. This file is the
operative reference that ships with the public fork and with any leaderboard
submission.

## Scope

This document covers:

- `_system_prompt` and every other static instruction string in
  `src/tau2/voice/audio_native/mentor/tool_boundary_mentor.py`
- note and gate output templates, structured-output schema descriptions
- any config value that injects instruction text into the mentor request or
  into the agent-visible mentor note channel
- gate-list wiring choices (which tools route to pre-write or pre-signal gates)

Exemption: the strings inside the heuristic development mode
(`tool_mentor_mode=heuristic`) are out of scope — they exist for offline
plumbing tests only and never run in submitted evaluations, which use `llm`
mode (R7).

## Core principle

The general-method claim of this fork is a single sentence: the mentor is
information-equivalent to the voice agent, and score uplift comes only from
additional reasoning over agent-visible inputs (agent-heard STT transcript,
official tool results, the domain policy the agent also receives, tool
schemas). Two boundaries protect that claim:

1. Runtime input boundary, enforced in code: mentor request packets carry the
   agent-observed transcript, official tool results, and policy text. Hidden
   task instructions, reward definitions, and user-simulator state stay out.
2. Prompt content boundary, enforced by this document through human litmus
   review of the full prompt text on every prompt-touching change: mentor
   prompt text carries zero benchmark-derived facts, so the prompt channel
   cannot smuggle information past the runtime boundary. There is
   deliberately no static test for this boundary (removed 2026-07-04, project
   decision): prompt meaning is wording-sensitive, and a passing keyword
   blacklist creates false confidence — the 2026-07-04 full-prompt review
   found benchmark-shaped content in a prompt the removed test accepted.

## Litmus test

Before adding a sentence to any mentor prompt, ask: could this sentence appear
in a real call-center training manual written before its author ever saw
tau-bench?

Examples taken from the 2026-07-02 telecom trace analysis:

| Candidate sentence | Verdict | Reason |
| --- | --- | --- |
| "Identifiers spoken aloud lose formatting in transcription. When a lookup keyed on a verbally provided identifier fails, retry canonical format variants of the customer's literal words (separators, spacing) before asking for new information." | PASS | True of every voice-plus-database system. The motivating trace (agent passed digits-only `5551232002` while the database key format uses separators) stays out of the sentence. |
| "When a lookup fails on a value the customer already stated, name the untried common renderings of that value (for example dash-separated groups, bare digits) and suggest retrying them before asking the customer again." (shipped v1.0 wording of the rule above) | PASS | Same rule with the examples narrowed from "(separators, spacing)" to the two renderings a speech-transcription layer universally produces for spoken number sequences — dash/space-grouped and bare digits — across phone numbers, order IDs, and account numbers in any voice channel. "For example" keeps the list non-exhaustive, the trigger (a failed lookup on an already-stated value) is a runtime predicate, and the rule fires for any identifier type. The narrowing was reviewed against this litmus before shipping. |
| "The caller only knows facts listed in their scenario instructions, so avoid asking for a birthdate." | FAIL | This describes the user-simulator specification. A real caller may know their birthdate. The legitimate version reacts at runtime to what this customer already said ("I don't remember the birthdate"). |
| "After device-side fixes, call refuel_data and enable_roaming before transferring." | FAIL | Task-family answer reverse-engineered from failed test cases, and it names tools. The mentor must derive available remedies from the runtime policy text on each call. |

## Rules

R1. Prompt text carries zero benchmark proper nouns. Banned inside prompt
strings: tool names, domain entity terms (for the current tau-voice surface:
mms, apn, roaming, sim, wifi calling, airplane mode, reservation, flight,
baggage, cabin), task family names, and the domain names themselves. Generic
customer-service vocabulary (customer, identifier, price, address, escalation)
stays allowed.

R2. Every domain fact the mentor uses enters at runtime through the policy
text, the tool names and descriptions, the observed transcript, or official
tool results.
Prompt text carries procedures only. When a procedure needs a domain fact (for
example, which remedies the agent itself can perform), the prompt instructs
the mentor to extract that fact from the policy, and the fact itself stays out
of the prompt.

R3. User-simulator meta-knowledge is banned. Caller-behavior priors are
allowed only when they hold for real callers (callers spell identifiers,
callers get impatient during silence, callers state one issue at a time).
Simulator internals fail the litmus test: stop-token mechanics, persona
labels, scenario-knowledge limits, check-in counts before hangup.

R4. Task-conditional instructions are banned. Conditions must be
domain-agnostic predicates over runtime state. Banned shape: "if the issue is
X, do Y". Allowed shape: "if the customer's goal check still fails after the
completed steps, enumerate policy-listed remedies within the agent's own
authority that remain untried, before any escalation".

R5. Statistical hygiene (relaxed 2026-07-02, project decision). Tune prompt
changes on small task slices and treat full-base-split runs as confirmation.
Adopt a change only when its effect clears observed single-trial churn:
task-level pass/fail flips between near-identical runs make single-run deltas
under roughly 10-15pp unreliable (measured churn: 17/36 mobile_data tasks
flipped between two same-config telecom runs). Formal dev-slice declarations
and iteration logs are optional working notes and carry no submission
obligation; the content boundary is enforced by litmus review, and run
provenance is already captured by results.json plus the fork git hash.

R6. Transfer evidence. One frozen prompt serves retail, airline, and telecom.
Per-domain deltas against the matching vanilla baseline are the generality
evidence; a rule motivated by telecom traces earns its place by also moving
retail or airline.

R7. Disclosure. Submissions publish the mentor prompt verbatim (or a frozen
public fork hash). The heuristic mentor mode (`tool_mentor_mode=heuristic`)
stays development-only; submissions use `llm` mode.

## Config wiring versus prompt text

Choosing which tools route to gates by name in code or config is architecture
wiring: it is visible in the public fork, disclosed in the submission, and
equivalent in kind to the existing ToolType-based routing. Preferred form:
derive action classes (escalation, mutation) from tool metadata and tool
descriptions at runtime. Allowed with disclosure: explicit enumeration in
config. Banned in all cases: tool names or task-conditional instructions
inside prompt text (R1, R4).

## Review checklist for any prompt-touching change

1. The full prompt (every sentence, old and new) passes the
   call-center-manual litmus test on a fresh read — review the whole text,
   never only the diff.
2. No user-simulator vocabulary appears (stop tokens, persona labels,
   scenario-instruction references, silence-policy constants).
3. Adoption is backed by an effect larger than single-trial churn (R5).
4. The change is a domain-agnostic procedure; any domain fact it relies on is
   fetched from runtime inputs.
