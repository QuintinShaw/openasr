# Distributing apps that embed OpenASR.xcframework

This note is for developers who link `OpenASR.xcframework` (from
`crates/openasr-ffi`) into iOS, iPadOS, or macOS applications. Build and API
details live in [docs/SDK_IOS_MACOS.md](docs/SDK_IOS_MACOS.md). Brand and
trademark rules live in [TRADEMARKS.md](TRADEMARKS.md) and
[BRANDING.md](BRANDING.md).

## What you may ship

Under Apache-2.0 you may:

- build the xcframework from this repository (or use a binary build you
  obtained lawfully);
- statically or dynamically link it into your app;
- redistribute the linked binary as part of your app, with the required
  license notices (`LICENSE`, `NOTICE`, and third-party notices as applicable);
- call the C ABI for local transcription, streaming, and consent-based model
  install as documented in the SDK guide.

The xcframework artifact does **not** carry a separate proprietary SDK license.
Trademark rules still apply to the **name and branding** of the shipping app.

## What you must not do

1. **Primary app name.** Do not name the App Store / TestFlight / Mac App Store
   product "OpenASR" or a confusingly similar title.
2. **Official icons.** Do not use the official OpenASR app icon or logo as your
   app icon or store artwork.
3. **Imply Apple or OpenASR endorsement.** Do not claim that Apple, the OpenASR
   project, or the project operators certified, authorized, or co-publish your
   app unless that is true under a written agreement.
4. **Pass off as the official app.** Do not reuse official bundle identifiers,
   App Store product pages, screenshots, or update funnels in a way that makes
   users think they are installing or updating the official OpenASR app.
5. **Support deflection.** Do not point end users at official OpenASR support
   channels for issues in your app's UI, accounts, billing, or packaging. You
   own support for your product.

## Required independent product surface

Any App Store (or sideloaded) app that embeds the xcframework must provide:

| Surface | Requirement |
| --- | --- |
| App display name | Independent of "OpenASR" |
| Bundle ID | Under **your** team / reverse-DNS |
| Signing & notarization | **Your** Apple Developer team |
| UI | Your screens and interaction design |
| Privacy nutrition / policies | Your privacy policy and data practices |
| User support | Your support URL or email in the store listing |

Attribution such as **“Powered by OpenASR”** on an About screen is encouraged.
It must remain secondary to your product name.

## Official apps vs ecosystem apps

| | Official OpenASR apps | Third-party apps embedding the xcframework |
| --- | --- | --- |
| Publisher | Project operators only | Any third party under this policy |
| Apple Developer account | Project-controlled accounts only | Your account |
| Product name | OpenASR (and official localized names) | Your brand |
| Engine | Same open-core / FFI | Same open-core / FFI allowed |
| Model packs | Signed catalog + user consent | Same trust rules in-engine; your UX for consent |

This split exists so users can tell who ships and supports the app they
installed, and so the project can show a clear chain of custody for official
store listings when needed (including evidence for platform review).

## Store review and “who publishes OpenASR?”

If a platform reviewer asks whether your binary is the official OpenASR app:

- **Official app:** published by the project operators’ developer accounts,
  under the OpenASR name and icon, listed from [https://openasr.org](https://openasr.org).
- **Your app:** a separate product that embeds Apache-2.0 OpenASR engine code;
  you are the sole publisher and support contact; engine attribution is
  “Powered by OpenASR” (or equivalent), not a claim of official status.

Keep a short statement to that effect in your review notes if you embed the
framework. This repository’s trademark and branding docs are the public
reference you can link.

## Technical obligations (unchanged)

Embedding the framework does not relax engine invariants:

- **No silent downloads.** Catalog fetch and model pull remain explicit,
  consent-driven calls; transcription must not trigger network install.
- **Fail closed.** Invalid packs, failed verification, and missing models must
  surface errors, not fabricated transcripts.
- **Trust boundary in-core.** Signature and path validation stay in
  `openasr-core`; do not bypass them from app code.

See [AGENTS.md](AGENTS.md) and [docs/SDK_IOS_MACOS.md](docs/SDK_IOS_MACOS.md).

## License notices in shipping apps

Ship Apache-2.0 attribution for OpenASR and the third-party components listed in
[NOTICE](NOTICE) (for example in Settings → Acknowledgments or a `Licenses`
screen). That is a license condition; it is not permission to brand the app as
OpenASR.

## Related docs

- [TRADEMARKS.md](TRADEMARKS.md) — name, logo, official-app reservation
- [BRANDING.md](BRANDING.md) — practical do / don’t for product identity
- [docs/SDK_IOS_MACOS.md](docs/SDK_IOS_MACOS.md) — build, C API, CPU-only v1 posture
- [LICENSE](LICENSE) / [NOTICE](NOTICE) — Apache-2.0 and third-party notices
