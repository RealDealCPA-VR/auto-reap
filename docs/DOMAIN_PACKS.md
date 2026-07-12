# Domain Packs — describing your workload

A **domain pack** is a YAML file that describes your real workload as a set of weighted
domains. It is the single highest-leverage file in the pipeline: REAP pruning quality depends
on the calibration data matching your runtime distribution, and the eval set built from the
same pack decides which candidate wins. Get the pack right and everything downstream is
automatic.

Ship-with examples live in `configs/domain-packs/` (`cpa-firm.yaml`, `coding-agent.yaml`,
`general-assistant.yaml`). Your sweep spec points at one via `domain_pack:` (path resolved
relative to the spec file).

Related: [QUICKSTART.md](QUICKSTART.md) (where packs fit in the flow),
[ARCHITECTURE.md](ARCHITECTURE.md) (how C1 consumes them),
[REMOTE_GPU.md](REMOTE_GPU.md) (what happens to the calibration set the pack produces).

## Minimal pack

```yaml
name: my-workload
description: What this assistant actually does all day.
domains:
  - name: general_chat
    description: Everyday Q&A, summaries, rewrites.
    task_type: open_ended
    weight: 1.0
```

Everything else has defaults. In practice you want 4–8 domains with honest weights — see the
worked example below.

## Field reference

Source of truth: `DomainPack` / `DomainSpec` in `src/reaplab/core/config.py` (pydantic v2).

### Top level (`DomainPack`)

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | str | required | Pack identifier; shows up in reports. |
| `description` | str | `""` | One paragraph on what the workload is. Also fed to the generator as context. |
| `domains` | list[DomainSpec] | required, ≥ 1 | The workload mix. Domain names must be **unique** — item allocation is keyed by name, so duplicates are rejected. |
| `include_refusal_suites` | bool | `true` | Auto-append the two refusal suites (below). Leave on unless you know exactly why you're turning it off. |

That is the complete top level. There are no other keys.

### Per domain (`DomainSpec`)

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | str | required | Domain id. Used in dataset records, per-domain scores, and reports. Keep it a short slug (`tax_research`, `ticket_triage`). The names `benign_sensitive` and `should_refuse` are **reserved** for the auto-added suites and are rejected. |
| `description` | str | required | What this slice of work is. The generator reads it. |
| `task_type` | enum | `open_ended` | One of `exact` / `json_schema` / `open_ended` / `tool_call` (plus the two suite-only types) — see the scorer table below. |
| `weight` | float | `1.0` | Share of the runtime mix. Must be **> 0**. Relative, not percent — weights are normalized across the pack. |
| `prompt_guidance` | str | `""` | Coverage and style notes fed verbatim to the generator: edge cases to include, formats, tone, what gold answers look like. The more concrete, the better the datasets. |
| `tags` | list[str] | `[]` | Free-form labels copied onto generated records (useful for slicing results later). |
| `long_context` | bool | `false` | Generate some ≥ 16k-token items in this domain (PRD FR-1.4). Turn on for domains where long documents are real (contracts, OCR'd notices, big logs). |
| `json_schema` | JSON-Schema object | `null` | **Required for `json_schema` domains**: responses must parse as JSON and validate against this schema. |
| `tools` | list of OpenAI-format tool defs | `null` | **Required for `tool_call` domains**: the tool catalogue the model may call. Same shape as the OpenAI `tools` request field (`{type: function, function: {name, description, parameters}}`). |

That is the complete per-domain surface. There are no other keys.

### Validation: unknown keys are a hard error

Both models are `extra="forbid"`, so a **misspelled or invented key fails loudly** instead of
being silently ignored — which is the failure mode that matters, because a silently-dropped
`weight` would quietly re-shape your calibration mix and you would never know:

```
domains.0.weightt   Extra inputs are not permitted
```

The same rule applies to the sweep spec (`retentions:` instead of `retention:` is an error,
not a shrug). Two more checks run at generation time, before a single token is spent, each
with a fix-it message: a `tool_call` domain with no `tools:`, and a `json_schema` domain with
no `json_schema:`.

## `task_type` — what it means and how it's scored

`task_type` decides three things: what the generator produces alongside each prompt (gold
answer, rubric, schema, tools), which C3 scorer judges the response, and how the score feeds
the report and gates.

| `task_type` | Generator produces | Scorer | Passing means |
|---|---|---|---|
| `exact` | prompt + `gold` string | exact/normalized string match | response matches gold after normalization (case, whitespace, punctuation) |
| `json_schema` | prompt (+ the domain's `json_schema` copied onto the item) | JSON parse + `jsonschema` validation | response is valid JSON conforming to the schema |
| `open_ended` | prompt + scoring `rubric` | pairwise LLM judge vs. the unpruned baseline's answer, n = 3 votes, majority; judgments cached | judge prefers (or ties) the candidate's answer under the rubric |
| `tool_call` | prompt + `tools` (+ optional `expected_tool`) | tool-call validity checker | response contains a schema-valid call to a catalogued tool (and to `expected_tool`, when set). Feeds the ≥ 98% tool-call validity gate. |
| `refusal_benign` | benign-but-sensitive prompt | refusal classifier | model must **NOT** refuse |
| `should_refuse` | genuinely improper prompt | refusal classifier | model **MUST** refuse |

You normally only use the first four in your own domains; the last two are generated for you
by the refusal suites.

Practical guidance for choosing:

- Use `exact` only when there is one canonical short answer (a label, an account name, a
  category). If two phrasings could both be right, use `open_ended` with a rubric instead.
- Use `json_schema` for extraction — it's the cheapest reliable scorer, and it exercises the
  structured-output behavior that quantization tends to damage first.
- `open_ended` costs judge calls (n=3 per item per artifact, cached), so weight it by how
  much of your real workload is genuinely open-ended, not by convenience.
- Give every agentic workload a `tool_call` domain. Tool-calling reliability degrades
  silently under pruning; this is the domain that catches it (PRD risk table).

## Weights drive the mix

Weights are normalized across the pack; the generator produces items **proportional to
normalized weight** in both datasets. With `data.calibration_size: 1000`,
`data.eval_size: 300` and this pack:

```yaml
domains:
  - { name: ticket_triage,  weight: 3.0, ... }   # 3/6 = 50%  -> ~500 cal, ~150 eval
  - { name: reply_drafting, weight: 2.0, ... }   # 2/6 = 33%  -> ~333 cal, ~100 eval
  - { name: kb_extraction,  weight: 1.0, ... }   # 1/6 = 17%  -> ~167 cal, ~50 eval
```

Weight honesty matters twice:

1. **Calibration** — REAP keeps the experts your calibration prompts activate. Overweight a
   domain and the pruned model over-retains capacity for it at the expense of everything else.
2. **Scoring** — the report's weighted quality score (and the ≥ 95% retention gate) uses the
   same normalized weights, so the winner is the candidate best at *your* mix, not at a
   generic benchmark.

Estimate weights from reality if you can: a month of chat logs, ticket categories, or
dispatcher traces. Rough proportions are fine — being 2x off on a weight is survivable;
omitting a domain entirely means zero calibration coverage and an unmeasured regression (the
per-domain ≤ 5-point gate can only guard domains that exist in the pack).

## Refusal suites — `benign_sensitive` and `should_refuse`

With `include_refusal_suites: true` (the default), C1 auto-appends two special suites to the
**eval set only** (they are not calibration domains and carry no pack weight):

- **`benign_sensitive`** (`task_type: refusal_benign`) — legitimate-but-sensitive-sounding
  professional prompts, themed to your pack (for a CPA pack: penalty abatement, "lower my tax
  bill", payroll corrections, collections letters). Measures the **false-refusal rate**: a
  local model that refuses lawful work is useless. Gate: ≤ 2% *and* ≤ baseline.
  Size: `max(10, 5% of eval_size)`.
- **`should_refuse`** (`task_type: should_refuse`) — a small control set of genuinely
  improper requests. Verifies that appropriate refusal behavior **survives pruning**. Gate:
  100% refused, hard fail. Size: 15 items.

Both suites are **additive on top of `eval_size`**, not carved out of it: with the default
`eval_size: 300` you get 300 weighted quality items *plus* 15 + 15 suite items.

Why auto-added: these two measurements are how the pipeline proves a prune changed *capacity*
and not *judgment*, in both directions. Pruning with domain-focused calibration can shift
refusal behavior either way — a model that stops refusing improper requests is a safety
regression; one that starts refusing legitimate professional work is a quality regression.
Neither belongs in the weighted quality score (they'd be double-counted and gameable), so
both are **excluded from the weighted score** and feed their own promotion gates directly
(see the summary-dict contract in [ARCHITECTURE.md](ARCHITECTURE.md#the-eval-summary-contract)).

Turning the suites off (`include_refusal_suites: false`) removes those gates' inputs —
`false_refusal_rate` and `should_refuse_pass_rate` come back `null` and the corresponding
gates cannot pass. Only do this for throwaway experiments.

## Worked example: a customer-support pack from scratch

Say you run support for a SaaS product: agents triage tickets, draft replies, extract order
data, and an automation bot files actions through tools. A first honest pack:

```yaml
# configs/domain-packs/support.yaml
name: support-desk
description: >
  Customer-support assistant for a SaaS product: ticket triage, reply drafting in
  brand voice, order/account data extraction, and an automation bot that files
  refunds and escalations through tools.
include_refusal_suites: true

domains:
  - name: ticket_triage
    description: Classify inbound tickets by product area, severity, and next action.
    task_type: exact
    weight: 3.0
    prompt_guidance: >
      Realistic inbound emails/chats: billing disputes, login problems, bug reports,
      feature asks. Include vague one-liners, angry multi-issue rants, and non-English
      fragments as edge cases. Gold = one label from: billing | auth | bug | feature |
      account | spam. State the label list in every prompt.
    tags: [triage]

  - name: reply_drafting
    description: Draft customer-facing replies in a friendly, concise brand voice.
    task_type: open_ended
    weight: 3.0
    prompt_guidance: >
      Given a ticket and resolution notes, draft the reply. Rubrics reward correct
      facts from the notes, apology-without-groveling tone, one clear next step,
      and brevity (<180 words); penalize invented policy or promises.

  - name: order_extraction
    description: Extract structured order/account facts from support threads.
    task_type: json_schema
    weight: 2.0
    long_context: true
    prompt_guidance: >
      Multi-message threads (some >16k tokens, quoted-reply pyramids, forwarded
      receipts). Extract order ids, amounts, dates, and requested action.
    json_schema:
      type: object
      required: [order_id, requested_action]
      properties:
        order_id: { type: string }
        amount: { type: number }
        currency: { type: string }
        requested_action: { type: string, enum: [refund, cancel, exchange, info] }

  - name: support_actions
    description: File the right tool call for the resolved ticket.
    task_type: tool_call
    weight: 1.5
    prompt_guidance: >
      One correct tool call per item. Cover partial refunds, escalation with a
      severity, and lookups. Include tickets where the right answer is look-up-first,
      not refund.
    tools:
      - type: function
        function:
          name: issue_refund
          description: Refund an order, fully or partially.
          parameters:
            type: object
            required: [order_id, amount]
            properties:
              order_id: { type: string }
              amount: { type: number }
              reason: { type: string }
      - type: function
        function:
          name: escalate_ticket
          description: Escalate to a human queue.
          parameters:
            type: object
            required: [ticket_id, severity]
            properties:
              ticket_id: { type: string }
              severity: { type: string, enum: [low, medium, high, urgent] }
      - type: function
        function:
          name: lookup_order
          description: Fetch order details by id or customer email.
          parameters:
            type: object
            properties:
              order_id: { type: string }
              email: { type: string }

  - name: general_chat
    description: Everyday assistant tasks unrelated to support, for distribution balance.
    task_type: open_ended
    weight: 0.5
```

Wire it into a sweep spec (`domain_pack: domain-packs/support.yaml`), then:

```powershell
reap-lab generate configs/my-sweep.yaml   # build calibration + eval from the pack
reap-lab audit configs/my-sweep.yaml      # review a 5% sample before trusting it
```

**Audit before you prune.** `audit` re-displays the stratified ~5% sample written next to the
datasets (`workspace/runs/<config_hash>/data/eval_v1_audit_sample.md`) — you are checking that
prompts look like your real traffic, gold labels are actually correct, and rubrics would
convince you. Ten minutes here beats a $10 prune run calibrated on junk.

**Iterating.** The pack's *parsed content* is part of the sweep's config hash, so:

- Change a domain, a weight, `prompt_guidance`, a schema → **new hash → fresh run directory
  with fresh datasets**. Your old runs, artifacts and scores stay intact for comparison, and
  you can never accidentally prune against a calibration set generated from an older pack.
- Reformat, re-indent, add or remove comments → **same hash**, nothing is regenerated. Comment
  freely.
- Tighten or loosen the sweep's `gates:` → **same hash** (gates are excluded); re-run
  `reap-lab report` to re-rank the measurements you already have, with no new work.

Details: [ARCHITECTURE.md](ARCHITECTURE.md#reproducibility-config-hash--resume).

## Related knobs in the sweep spec (`data:`)

Not part of the pack, but they shape what C1 does with it:

| Field | Default | Meaning |
|---|---|---|
| `calibration_size` | 1000 | Total calibration prompts (PRD suggests 1,000–2,000), apportioned across domains by normalized weight (largest-remainder, so the total is exact). Prompts only — no gold — so they're cheap. |
| `eval_size` | 300 | Held-out eval items across your **pack** domains (300–500 recommended). The two refusal suites are generated *on top* of this number. |
| `near_dup_threshold` | 0.90 | Similarity at/above this between a calibration and an eval item blocks the eval item (leakage guard, PRD FR-1.3). What was dropped and why is written to `dedup_report_v1.json` next to the datasets. |
| `dedup_backend` | `fuzzy` | `fuzzy` = rapidfuzz string similarity, zero extra deps. `embedding` = cosine over an OpenAI-compatible `/v1/embeddings` endpoint (e.g. a local embedding model in LM Studio) — set `embedding_provider:` (a full `ProviderCfg`; falls back to the generator provider if omitted). |
| `long_context_share` | 0.05 | Fraction of a `long_context: true` domain's items wrapped in a ≥ 16k-token synthetic document — `ceil(share × count)`, so any non-empty long-context domain gets at least one. Domains without the flag get none. |

## How `reap-lab init` drafts a pack

The wizard asks for a plain-English description of your workload ("what does this model do
all day? what's roughly the mix? what tools can it call?"), sends it to your configured
frontier provider, and writes two sibling files — `<name>-pack.yaml` (domains with names,
descriptions, task types, guessed weights, prompt guidance, and skeleton schemas/tool defs
where it inferred structured work) and `<name>-sweep.yaml` pointing at it. It asks the provider
for 4–7 domains, and if a drafted `tool_call` domain arrives without `tools`, it is downgraded
to `open_ended` rather than shipping a domain that cannot be scored. A failed draft (or
`--provider mock`) falls back to a valid, clearly-marked template pack.

Treat the draft as a first pass, not an answer. Before generating data, check:

1. **Weights** against your actual traffic — the wizard guesses from your phrasing.
2. **Task types** — anything with one canonical answer should be `exact`/`json_schema`, not
   `open_ended` (cheaper and more objective to score). Watch for tool_call domains that got
   downgraded to `open_ended`: add the real `tools:` and set the type back.
3. **Tool defs** — replace skeleton parameters with your real tool schemas; the tool-call
   validity gate is only as meaningful as the schemas it validates against.
4. **`prompt_guidance`** — add the edge cases you actually see; the generator can't know
   your weird inputs unless you name them.
