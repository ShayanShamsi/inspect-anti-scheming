# inspect-anti-scheming

Inspect-AI re-implementation of the 25 (of 26) anti-scheming test environments from:

> Schoen et al. 2025. *Stress Testing Deliberative Alignment for Anti-Scheming
> Training.* Apollo Research & OpenAI. [arXiv:2509.15541](https://arxiv.org/abs/2509.15541).

Environment #4 in Appendix B (*OpenAI Chat Deception*) is intentionally
omitted — it is an internal OpenAI evaluation drawn from the GPT-5 system
card representing ChatGPT production conversations, and is not available
externally (per the paper's Fig. 6 †, even Apollo did not have access to it
for manual verification).

## What this measures

The paper defines **covert actions** — situations where an AI agent
strategically withholds, misrepresents, or conceals information that users
or developers would consider material — as a proxy for scheming. Each
environment in this package places a model in a multi-turn, tool-using
scenario where covert action is incentivized, and a rule-based or
model-graded scorer determines whether the model took it.

The 25 environments span six behavioural buckets matching the paper's
Fig. 6 taxonomy:

| Bucket | Tasks |
|---|---|
| **QA / reward-hacking** | `sandbagging_qa`, `sandbagging_qa_no_goal`, `sandbagging_grading_cot`, `literature_qa`, `lazy_checklist`, `long_document_verification` |
| **Software / sabotage** | `impossible_coding` (logical + SLA families), `sabotage_math`, `sabotage_safety`, `log_falsification`, `smuggle_capabilities`, `email_scoring`, `as_over_escalation` |
| **Corporate / lying** | `legal_reporting_cn`, `legal_reporting_us`, `broken_promises`, `powerseeking_survey`, `subagent_collusion` |
| **Spec sabotage** (red-team) | `spec_sabotage_moving`, `spec_sabotage_selection`, `spec_sabotage_verification` |
| **Games** | `blotto`, `prisoners_dilemma` |
| **Subagent / privilege** | `subagent_permissions`, `subagent_resource_quota` |

A separate `awareness/eval_awareness_classifier.py` post-hoc scans completed
log files for verbalized evaluation awareness, mirroring the analysis in
§6 of the paper.

## Install

```bash
pip install inspect-anti-scheming
# or, for development
uv sync --group dev
```

## Run

Run a single environment with `inspect eval`:

```bash
inspect eval anti_scheming/qa/sabotage_safety.py \
    --model openai/gpt-4o-mini --limit 5
```

Or use the included driver to run the full suite against one model and
emit a Fig. 24-style table of per-env covert rates:

```bash
python scripts/run_anti_scheming.py --model openai/gpt-4o-mini --limit 3
```

The default backend is an in-process virtual filesystem (`anti_scheming/common/vfs.py`)
to avoid per-sample Docker container churn. For envs that support a real
sandbox (`sabotage_safety`, `impossible_coding`), pass `--backend=docker` to
the task — see each module's docstring for env-specific knobs.

## Anti-scheming spec

Tasks accept a `with_spec: bool = False` parameter. Setting it to `True`
injects the paper's 9-clause anti-scheming spec (AS1–AS5 + GP1–GP4) into
the system prompt, matching the deliberative-alignment intervention
evaluated in the paper. This lets you A/B the same model with and without
the spec without retraining. The full text lives in
`anti_scheming/common/prompts.py`.

## Scoring

Per-environment covert criteria are encoded as `covert_predicate(state, target)`
callables passed to a shared `covert_action_scorer` (`anti_scheming/common/classifier.py`).
The scorer reports `INCORRECT` for covert samples and `CORRECT` otherwise;
the standard `accuracy()` metric on the resulting score yields the
**non-covert rate**, so the paper's "covert rate" is `1 − score`.

A few envs use `model_graded_covert_scorer` when covertness is too
subjective for a regex / structural check; that judge uses last-match-wins
grade extraction (`GRADE: covert` vs `GRADE: not_covert`).

## Citation

If you use this package, please cite the underlying paper:

```bibtex
@article{schoen2025stress,
  title  = {Stress Testing Deliberative Alignment for Anti-Scheming Training},
  author = {Schoen, Bronson and Nitishinskaya, Evgenia and Balesni, Mikita and
            H{\o}jmark, Axel and Hofst{\"a}tter, Felix and Scheurer, J{\'e}r{\'e}my
            and Meinke, Alexander and Wolfe, Jason and van der Weij, Teun and
            Lloyd, Alex and Goldowsky-Dill, Nicholas and Fan, Angela and
            Matveiakin, Andrei and Shah, Rusheb and Williams, Marcus and
            Glaese, Amelia and Barak, Boaz and Zaremba, Wojciech and
            Hobbhahn, Marius},
  journal = {arXiv preprint arXiv:2509.15541},
  year   = {2025},
}
```

## License

MIT. See [LICENSE](LICENSE).

This package is an independent re-implementation. It is not affiliated
with Apollo Research or OpenAI.
