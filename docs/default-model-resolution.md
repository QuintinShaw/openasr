# Default Model Resolution

Normative contract for what "the default model" means across OpenASR, and for
the single resolver every caller must go through to read or change it.

## Single authority

`openasr_core::default_selection` (`crates/openasr-core/src/default_selection.rs`)
is the **only** place that resolves or persists the default model. It is used
by:

- `openasr-server`'s `GET/POST/PUT /v1/models/default` and `DELETE
  /v1/models/{id}` routes (`crates/openasr-server/src/routes/models_api.rs`,
  thin delegates only).
- `openasr-cli`'s `serve`/`transcribe`/`live` pack resolution
  (`crates/openasr-cli/src/native_segment_cli.rs`) for the "no `--model`
  passed" case.
- `openasr pull`'s auto-set-default-on-install behavior
  (`crates/openasr-cli/src/pull_cli.rs`).

No shell, frontend, or new server/CLI code may reimplement this resolution or
fabricate a default from in-memory state. If a new surface needs "what is the
default model," it calls `default_selection::resolve` (or the server's `GET
/v1/models/default`, which is backed by the same resolver) -- it does not read
`config.json`'s `default_model` field directly and assume that is the whole
answer, because it isn't (see "Two files" below).

## Two files, one state machine

The default model spans two files under `OPENASR_HOME`:

- `config.json`'s `default_model` field: the user's explicit choice (a bare
  model id, no quant tag). `None` means "the user has not chosen one yet."
  This field carries **no implicit value** -- a fresh config (missing key, or
  `OpenAsrConfig::default()`) deserializes to `None`, never to a hardcoded
  model id.
- `default.json` (`default_pack_pointer_path`): a pointer recording the
  quant-tagged pack that the last default-setting write actually installed
  against. It is a fallback, read only when `config.default_model` is unset,
  and it is also how a `QuantPreference::Pinned` preference recovers which
  quant was pinned.

`default_selection::persist` writes both files together (config first, then
the pointer) so they never drift; `default_selection::clear` resets both
(clearing `default_model` and `quant_preference`, removing the pointer file).
Callers must never write one without the other.

## Fail-closed: never invent a default

`resolve` never substitutes a different installed pack when the configured
default is missing, and never picks "some" installed model when nothing is
configured. A configured-but-uninstalled default model resolves to
`NotInstalled`, not to whatever else happens to be on disk -- silently
substituting a different model/quant than the one the user chose would defeat
the point of "default" and could route audio through unexpected weights.

## Priority

1. `config.default_model`, if set.
2. Otherwise, `default.json`'s pointer model id, if a pointer file exists.
3. Otherwise: unset.

If `preferences.quant_preference` is `Pinned` and a pointer file exists, the
pointer's *quant* is tried first for whichever bare model id priority (1) or
(2) selects, falling back to the best installed quant for that model if the
exact pinned quant was removed.

## Three-state result

A bare `Option<InstalledPack>` cannot distinguish "nothing configured" from
"configured but not installed" -- both collapse to `None`. Callers that need
to show the right prompt ("choose a model" vs. "reinstall your default") use
`DefaultModelResolution`:

- `Installed(InstalledPack)` -- a default is configured (directly or via the
  pointer) and a matching pack is installed.
- `NotInstalled(String)` -- a default is configured but nothing installed
  matches it (removed, never pulled, or an unrecoverable quant).
- `Unset` -- neither `config.default_model` nor the `default.json` pointer is
  set.

`GET /v1/models/default` exposes this as `default_model_status`: `"installed"`
| `"not_installed"` | `"unset"`. See the JSON shape below.

```jsonc
{
  "object": "model.default",
  // Bare model id. Present for "installed" and "not_installed"; absent for "unset".
  "default_model": "whisper-small",
  "default_model_status": "not_installed",
  // Quant-tagged pull id + full pack metadata, only when status is "installed".
  "default_pull": null,
  "pack": null
}
```

## What this module is not

`DEFAULT_MODEL_ID` (`crates/openasr-core/src/config.rs`) is a **separate,
CLI-only** concept: the bare-invocation convention `transcribe`/`live`/`pull`
fall back to when the caller passes neither `--model` nor has a persisted
`config.default_model`. It feeds the CLI's last-resort reference and the
consent-pull prompt copy; it is never written into `config.json` and never
flows through `default_selection`. Conflating the two is the bug class this
module was written to close: a fresh install used to get an implicit
`config.default_model` pointing at a model nobody had installed, so the
default-model status looked "configured" everywhere except in reality.
