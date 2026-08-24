# Windows backend qualification cell producer

This runner executes one exact `(family, model, quant, provider, target,
backend_id)` cell inside an already prepared open-core qualification scope. It
derives provider identity from the projected matrix, compares GPU cold/warm
process rows against CPU family-oracle traces, and delegates evidence binding to
`gpu_correctness_gate.py bind-cell`.

It does not install, activate, sign, or deploy a provider. Its four receipts and
two traces are inputs to the existing correctness gate only. A trusted workflow
must attest those exact output bytes before catalog activation can consume them.
