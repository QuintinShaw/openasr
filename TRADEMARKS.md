# OpenASR Trademarks

This policy covers the **OpenASR** name, word marks, logos, official app icons,
and other brand assets of the OpenASR project. It is separate from the
Apache-2.0 license that covers the open-core source code.

Apache-2.0 **does not** grant trademark rights. Section 6 of the
[LICENSE](LICENSE) already states that the license does not grant permission to
use the trade names, trademarks, service marks, or product names of the
licensor, except as required for reasonable and customary use in describing the
origin of the work. This document states how the project applies that rule in
practice.

## What is reserved

The project operators retain all rights in:

- the name **OpenASR** and confusingly similar variants (including spacing,
  capitalization, and transliterations used as a product name);
- the official OpenASR logos and wordmarks;
- the official OpenASR Desktop / mobile **app icons** and store listing artwork
  published by the project;
- domain names and URLs operated by the project (including `openasr.org` and
  `dl.openasr.org`), when used to imply official status.

## Software license vs trademarks

| Asset | Covered by Apache-2.0? | Trademark / brand rules |
| --- | --- | --- |
| Source code, docs text, build scripts | Yes | You may use, modify, and redistribute under Apache-2.0 |
| Compiled binaries / libraries you build from source | Yes (as Object form) | You may ship them, but **not** under the OpenASR product name as your primary app identity (see below) |
| Name "OpenASR", logos, official icons | **No** | This policy |

Using the open-core code does **not** make a third-party product an official
OpenASR product.

## Allowed uses (no extra permission required)

You may:

1. **Factual reference.** State that your product uses, embeds, or is compatible
   with OpenASR open-core software (for example in documentation, academic
   papers, or a dependency list).
2. **"Powered by OpenASR".** Use that exact phrase (or a clear equivalent such as
   "Uses OpenASR open-core") in a secondary attribution line, provided it does
   not look like the primary product name or store title.
3. **Describe origin.** As Apache-2.0 allows, note that portions of the work are
   derived from the OpenASR project and point to the upstream repository.
4. **Compatibility claims that are true.** For example "imports OpenASR `.oasr`
   packs" or "speaks the OpenASR local HTTP API", when accurate.

## Not allowed without written permission

You may **not**:

1. Use **OpenASR** (or a confusingly similar name) as the **primary product
   name**, app name, store listing title, or home-screen label of a third-party
   product.
2. Use the official OpenASR **logo** or **app icon** as your product's logo,
   icon, or store artwork.
3. Imply **official status**: endorsement, certification, partnership,
   "authorized", "official client", "official build", or joint offering with the
   OpenASR project or its operators, unless you have a written agreement that
   says so.
4. Present modified or re-skinned builds as **the** OpenASR Desktop / mobile app
   or as updates to the official app.
5. Register company names, trademarks, app IDs, or domains that are likely to
   confuse users into thinking they are dealing with the official project
   (for example `openasr-pro.example`, `get-openasr`, or look-alike icons).

## Official applications

Official OpenASR end-user applications (desktop and mobile store builds) are
published **only** by the project operators through their own developer
accounts (including the official Apple Developer and other store accounts they
control). Store listings, signing identities, and bundle identifiers for those
official apps are not licensed for third-party reuse as a way to appear
official.

Third parties who ship apps that embed OpenASR open-core or the xcframework
must use **their own** product name, UI, branding, signing identity, store
listing, and user-support channels. See [BRANDING.md](BRANDING.md) and
[XCFRAMEWORK-DISTRIBUTION.md](XCFRAMEWORK-DISTRIBUTION.md).

## Fair use and nominative use

Nothing in this policy is intended to limit non-confusing nominative fair use
under applicable law (for example, "we converted weights for use with OpenASR"
in a technical write-up). When in doubt, prefer a factual sentence plus a link
to [https://openasr.org](https://openasr.org) over logo use.

## Reporting confusion

If you believe a product is misusing the OpenASR name or marks in a way that
confuses users about official status, contact the maintainers through the
channels listed on [https://openasr.org](https://openasr.org) or the GitHub
organization that publishes this repository.

## Changes

This policy may be updated as the project and its official products evolve.
The copy in the default branch of this repository is the current version.
