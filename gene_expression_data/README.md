# `gene_expression_data/`

Input files for `build_context_model_with_corda` — the tool that builds a
context-specific (e.g. tissue- or condition-specific) genome-scale model from
the currently loaded base/reference model using **CORDA** (Cost Optimization
Reaction Dependency Assessment).

Upload a single CSV. The tool auto-detects (`data_type="auto"`, the default)
which of three formats it's looking at; you can also force the
interpretation with `data_type="expression"`, `"gene_confidence"`, or
`"reaction_confidence"`.

## Accepted formats

All three formats are a plain 2-column CSV: an identifier column and a value
column.

1. **Gene expression** — gene id + a continuous expression value (TPM, FPKM,
   or raw counts). This is the easiest format to provide — just export raw
   expression values. Values are automatically binned per-gene into CORDA's
   four confidence classes by percentile:
   - `>= high_percentile` (default 75th) → confidence `3` (high)
   - `>= mid_percentile` (default 50th) → confidence `2` (medium)
   - `>= low_percentile` (default 25th) → confidence `1` (low)
   - below that → confidence `-1` (treated as not expressed)

   Thresholds are adjustable via the `high_percentile`, `mid_percentile`, and
   `low_percentile` tool parameters.

   Example: [`brca_corda_input.csv`](brca_corda_input.csv) — gene symbol +
   expression value, breast-cancer expression data.

2. **Gene confidence** — gene id + an integer already on CORDA's native
   confidence scale: `{-1, 0, 1, 2, 3}` (`-1` = not expressed, `0` =
   unknown/unmeasured, `1`–`3` = low/medium/high confidence expressed). Use
   this if you've already computed confidence scores yourself.

3. **Reaction confidence** — reaction id (must match reaction IDs in the
   currently loaded model) + an integer confidence in `{-1, 0, 1, 2, 3}`. Use
   this to drive CORDA directly at the reaction level instead of the gene
   level, bypassing gene→reaction mapping entirely.

## Column naming

The tool looks for common header names (case-insensitive) and otherwise
falls back to position (column 1 = id, column 2 = value):

- **ID column**: `gene`, `gene_id`, `genes`, `gene_name`, `locus`,
  `reaction`, `rxn`, `rxn_id`, `reaction_id`, `id`
- **Value column**: `confidence`, `conf`, `score`, `expression`, `expr`,
  `value`, `level`, `tpm`, `fpkm`, `rpkm`, `counts`, `count`

Gene-level vs. reaction-level is auto-decided by whichever id type in the
file matches more entries in the loaded model — you don't need to declare it
unless you want to force a specific interpretation.

## Gene ID matching

Gene ids in the file do **not** need to match the model's own gene IDs. They
are resolved through a cross-reference index built from the loaded model —
covering the model's gene id, gene symbol/name, and every cross-reference in
each gene's annotation (Ensembl, Entrez/NCBI gene, UniProt, RefSeq, ...) —
with trailing version suffixes like `.15` stripped before matching. This
means a file downloaded directly from GTEx, Expression Atlas, GEO, etc. can
usually be used as-is without renaming any genes. The tool's response
includes `id_mapping_coverage`, reporting how many model genes were
successfully matched, so you can sanity-check the mapping quality.

## Other behavior worth knowing

- Duplicate ids (e.g. multiple probes or transcripts per gene) are collapsed
  by taking the **max** value across duplicates.
- By default (`keep_objective_high=True`) the model's objective reaction(s)
  are forced to confidence `3` so the reconstructed context-specific model
  can still support/grow around the objective, even if the expression data
  alone wouldn't have kept those reactions.
- A base/reference model must already be loaded in the session before
  calling this tool — CORDA prunes down from that model, it doesn't build
  one from scratch (for that, see `build_model_with_carveme` or
  `build_model_with_mackinac`, documented in
  [`../Documentation/README.md`](../Documentation/README.md)).
