# Windows backend qualification cell producer

This runner executes one exact `(family, model, quant, provider, target,
backend_id)` cell inside an already prepared open-core qualification scope. It
derives provider identity from the projected matrix, compares GPU cold/warm
process rows against CPU family-oracle traces, and delegates evidence binding to
`gpu_correctness_gate.py bind-cell`.

It does not install, activate, sign, or deploy a provider. Its four receipts and
four traces (two token traces and two complete logits traces) are inputs to the existing correctness gate only. A trusted workflow
must attest those exact output bytes before catalog activation can consume them.

Run with PowerShell 7 and Python 3. `OPENASR_HOME` and the qualification scope
must identify an isolated home already prepared by the native CLI. The runner
discovers the actual device through a native request, then uses `qualify-family`
for cold and warm requests in one process. It binds both against independent
CPU oracles. The projected graph mode and evidence revision come from the
shared runtime inventory; they do not authorize capture or product activation.

All raw inputs, receipts and failed evidence remain in a unique `raw-*` child of
the output directory. The caller owns deliberate cleanup after inspection.
