# Rule for the default merge dtype (written before the export comparison was run)

`spill export --merge-dtype float32|bf16` needs a default. The default is chosen by this rule, fixed
here, before any result of the controlled comparison (docs/reports/017-export-diagnosis.md) exists.

Candidates: a float32 merge and a bf16 merge of the same adapter into the same base. Evidence: the
verification rows of both task types (8 validation rows each, so 16 rows per artifact kind), for two
artifact kinds per candidate: the safetensors merge loaded by Transformers on the CPU, and the GGUF
q8_0 derived from that merge run by llama.cpp on the CPU. All inputs, rendered prompts, decoding
settings and rows are identical across the comparison.

1. Task quality first. For each candidate, count the verification rows scored correct under the
   task's primary metric (classification accuracy; JSON whole-record accuracy under metric version
   2), summed over both task types and both artifact kinds (32 rows per candidate). If the
   candidates differ by 2 rows or more, the one with more correct rows is the default.
2. If task quality is within 1 row, fidelity to the source engine: count the rows whose output text
   differs from the source engine's output on the same row (same 32 rows). If the candidates differ
   by 2 or more, the one with fewer differing rows is the default. Text agreement is never the only
   criterion: it applies only when step 1 did not decide.
3. If still undecided, resource cost: the smaller safetensors artifact on disk (and, if equal, the
   lower peak process memory in the Transformers verification). The bf16 merge is about half the
   size of the float32 merge, so on this step bf16 wins.
4. Tie-break: if every step ties, the default stays bf16 (the current behavior).

What this can and cannot show: 8 rows per task type can support an implementation decision, such as
which dtype a default should be. They cannot establish that one merge dtype universally preserves
quality better, and the report says so.
