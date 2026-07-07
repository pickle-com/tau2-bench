# Voice tau²-bench: grok-voice + Tool-Boundary Mentor (custom scaffold)

This branch is the public, curated form of the code that produced the results
below. Every code path exercised by the reported runs is byte-identical to
the experimental branch that ran them. The curation removed: internal
experiment launchers and analysis/log-viewer tooling, an internal API-key
alias, an unrelated in-development provider integration, and two undocumented
xAI debug env overrides (`TAU2_XAI_CONNECT_URL_OVERRIDE`,
`TAU2_XAI_SESSION_EXTRA_JSON` — verified unset in every reported run: all
connection logs show the standard query-built URL). Additionally, two
handshake log lines moved from info to debug level and one import block was
re-sorted. The
trajectory files record the internal run commit (`ae28e64`); this branch is
its curated public equivalent, and offline rescoring of the published
trajectories with this code reproduces the reported numbers exactly. This
document is the reproduction contract for the leaderboard submission that
references this branch.

## Results

Full base task splits, single trial (pass^1), tau² standard evaluators
(DB-hash + gold-action matching), one frozen configuration across all three
voice domains — no per-domain tuning:

| Domain (tasks) | This system | Official grok-voice entry (2026-04) |
|---|---|---|
| retail (114) | **0.7632** | 0.623 |
| airline (50) | **0.7000** | 0.660 |
| telecom (114) | **0.7807** | 0.737 |
| **macro** | **0.7480** | 0.673 |

Comparability caveats, in the spirit of existing custom leaderboard entries:

- **User simulator**: `gpt-5.5-2026-04-23` with `reasoning_effort=xhigh`.
  Not comparable to standard-user-simulator (v1.0, gpt-4.1) leaderboard
  entries.
- **Voices**: local ElevenLabs voice personas (`TAU2_VOICE_ID_*`), not the
  Sierra-managed official submission voices.
- **Serving stack**: the base model is xAI `grok-voice-think-fast-1.0` as
  served in July 2026. In our environment we were unable to reproduce the
  April 2026 official score on the current endpoint with default settings
  (server-default VAD frequently missed user speech entirely); the explicit
  VAD/gain settings below restored a working audio pipeline. In the same
  environment, runs without the mentor on the current serving stack scored
  substantially lower than both columns of the table (partial ablations; a
  full three-domain mentor-off suite under the final audio configuration was
  not run).
- **Single trial**: task-level pass^1 rerun churn is large on these domains;
  treat per-domain deltas of a few points as within noise.

## System

Two layers on top of the upstream full-duplex voice harness (200 ms ticks,
8 kHz G.711 mu-law telephony simulation):

### 1. Audio pipeline settings (xAI serving-stack compat)

The xAI realtime endpoint now requires the model as an explicit URL
parameter and emits `session.created` (previously `conversation.created`).
With server-default VAD, user utterances are frequently not detected at all.
The run configuration pins:

- `--xai-vad-threshold 0.1` — detect low-energy telephony speech
- `--xai-vad-silence-duration-ms 1200` — keep spelled-out readings (IDs,
  emails) in one turn instead of fragmenting them
- `--xai-vad-prefix-padding-ms 600` — preserve first syllables
- `TAU2_XAI_INPUT_GAIN=2.0` — linear gain on outbound mu-law audio;
  improves detection of quiet utterances and resumption after agent replies

### 2. Tool-boundary mentor (`--tool-mentor`)

A separate LLM (`gemini/gemini-3.5-flash`, reasoning effort **high**)
supervises the voice agent's tool calls in real time:

- **pre_write gate**: state-mutating and escalation tool calls (default:
  `transfer_to_human_agents`) are reviewed before execution. `allow`
  executes normally; `block` skips execution and returns the mentor's note
  into the agent context instead.
- **post_read note**: read-tool results get a short guidance note appended
  before the agent sees them.
- **stop-interception**: when the agent's stop tool is pre-gated,
  conversation termination defers to the gate outcome — a blocked hand-off
  proposal continues the conversation instead of silently terminating it.
- **fail-open**: mentor timeouts (read 10 s / write 15 s in the submitted
  runs) or errors never block the run; the original tool call proceeds.

**Information boundary.** The mentor sees only agent-visible inputs: the
agent-heard STT transcript, official tool results, the same domain policy
text the agent receives, and the agent tool names and descriptions. Hidden task
instructions, reward definitions, and user-simulator state stay out. The
mentor prompt itself carries zero benchmark-derived facts; the binding
content policy, litmus test, and review checklist are in
[mentor-prompt-constitution.md](mentor-prompt-constitution.md). The
heuristic mentor mode (`--tool-mentor-mode heuristic`) exists for offline
tests only; submitted runs use `llm` mode.

**Stop-interception fairness rationale.** Composite-agent semantics: the
base model's emission becomes final only after the composite system's
verifier clears it — exactly the treatment blocked writes already receive.
The un-emitted transfer is unobservable to the user simulator (the tool path
produces no audio). Mentor latency stays charged as simulated silence under
`tool_mentor_realtime_wait`; the global tick budget is unchanged; with the
mentor disabled, official stop semantics are untouched.

Measured gate activity in the submitted runs — retail: 229 pre-write gates
(127 allow / 58 block / 44 fail-open timeouts) and 893 post-read notes;
airline: 72 (49 / 10 / 13) and 442; telecom: 186 (111 / 47 / 28) and 625. A
block is not a win by itself: the continued conversation must still pass the
standard evaluators within the unchanged tick budget, and failed tasks in
every domain include conversations where the mentor blocked a hand-off
(retail 10 of 27 failures, airline 5 of 15, telecom 7 of 25). Every gate
decision is auditable per simulation via
`info.voice_timing_trace.mentor_events`.

## Contribution ablation

With the mentor prompt frozen, raising the mentor's reasoning effort from
low to high moved telecom 0.649 → 0.781 (paired same-tasks: 24 recovered /
9 regressed) and the gate block rate from 23% to 30%; by comparison, a
round of prompt-wording iteration was worth a net +1 task. Both figures are
internal same-stack, single-trial comparisons — the official vanilla row
does not reproduce on this stack (see the caveats above), so neither number
is comparable to it. The dominant contribution of this scaffold is the
supervisor's reasoning budget, not its prompt wording.

## Reproduction

### Credentials

`ELEVENLABS_API_KEY` (user-sim TTS), `DEEPGRAM_API_KEY` (user-sim
transcription), `XAI_API_KEY` (agent), `OPENAI_API_KEY` (user-sim LLM),
`GEMINI_API_KEY` (mentor LLM), and one `TAU2_VOICE_ID_<PERSONA>` ElevenLabs
voice ID per persona in `tau2.data_model.voice_personas`. The preflight below
checks the voice-pipeline and mentor credentials (the user-simulator LLM key
is not part of the preflight — export it as well):

```bash
tau2 run --domain telecom --audio-native --audio-native-provider xai \
  --tool-mentor --tool-mentor-model gemini/gemini-3.5-flash --preflight-only
```

(`--skip-voice-id-check` skips the local voice-ID checks.)

### Run command (as submitted, per domain)

```bash
TAU2_XAI_INPUT_GAIN=2.0 \
TAU2_OFFICIAL_PYTHON_HASH_VOICE_SEED=1 \
TAU2_ELEVENLABS_OUT_OF_TURN_TTS_MAX_WORKERS=1 \
python -m tau2.cli run --domain {retail|airline|telecom} \
  --audio-native --audio-native-provider xai \
  --audio-native-model grok-voice-think-fast-1.0 \
  --xai-vad-threshold 0.1 --xai-vad-silence-duration-ms 1200 \
  --xai-vad-prefix-padding-ms 600 \
  --num-trials 1 --max-steps 6000 --max-concurrency 10 \
  --user voice_streaming_user_simulator --user-llm gpt-5.5-2026-04-23 \
  --user-llm-args '{"reasoning_effort": "xhigh"}' \
  --tool-mentor --tool-mentor-model gemini/gemini-3.5-flash \
  --tool-mentor-reasoning-effort high \
  --tool-mentor-read-timeout 10.0 --tool-mentor-write-timeout 15.0 \
  --tool-mentor-realtime-wait --tool-mentor-realtime-workers 5 \
  --speech-complexity regular --skip-voice-id-check --auto-resume
```

Notes:

- `--tool-mentor-model` must be passed explicitly (the CLI default is the
  generic default agent model, not the mentor model used here).
- `TAU2_OFFICIAL_PYTHON_HASH_VOICE_SEED=1` selects the upstream salted-hash
  per-task voice assignment (the submitted runs used it). Without it, this
  branch uses a process-stable blake2b seed — see behavioral differences.
- `TAU2_XAI_INPUT_GAIN` has no CLI flag; the env var is the only switch.

### Auditing mentor decisions

Mentor runs force `--verbose-logs` and default `--llm-log-mode all`. Every
gate/note decision is recorded in each saved simulation under
`info.voice_timing_trace.mentor_events` (linked to tool calls by
`tool_call_id`), and every mentor LLM request/response lands under each
task's `llm_debug/` artifact directory as raw JSON.

## Behavioral differences from upstream (mentor off)

Disclosed changes that affect voice runs even without `--tool-mentor`:

1. The voice user simulator's `get_init_state()` double-initialization is
   removed (upstream initializes twice per simulation).
2. `Tick.get_all_messages` no longer emits empty message chunks (no content
   and no tool calls). Derived flat message streams shrink; the env-replay
   scoring path is unaffected.
3. Per-task voice-config seeds default to a process-stable blake2b
   derivation instead of Python's salted `hash()`;
   `TAU2_OFFICIAL_PYTHON_HASH_VOICE_SEED=1` restores upstream behavior.
4. The xAI model is passed explicitly as a URL query parameter, the default
   model name is `grok-voice-think-fast-1.0`, and the connect handshake
   accepts `session.created` in addition to `conversation.created` —
   required by the current xAI serving stack.
5. Voice-run results include `voice_timing_trace` telemetry, and
   assistant-message `raw_data` carries per-turn input audio transcripts
   (larger result files; no scoring effect).
6. Out-of-turn TTS pre-generation workers default to 2 (upstream: 10) to
   avoid ElevenLabs throttling; setup speed only.
7. `tau2.scripts.evaluate_trajectories` auto-detects full-duplex runs when
   rescoring saved trajectories (upstream rescored them as half-duplex,
   which corrupts voice rescoring).
