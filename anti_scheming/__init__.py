"""Anti-scheming evals — re-implementation of Schoen et al. 2025.

Importing this package does NOT import the individual env modules eagerly;
they're loaded on demand via `run_anti_scheming.py` or by importing the
specific submodule (e.g. `anti_scheming.qa.lazy_checklist`).

Coverage note — Appendix B has 26 test environments. We implement 25.
#4 (OpenAI Chat Deception) is intentionally not implemented: per the paper
(Fig. 6 footnote †) it is an internal OpenAI evaluation drawn from the
GPT-5 system card representing ChatGPT production conversations, and is
not available externally — even Apollo did not have access to the data
for manual verification (the paper excludes its results from cross-env
aggregations for the same reason).
"""
