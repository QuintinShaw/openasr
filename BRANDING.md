# OpenASR Branding Guidelines

How to refer to OpenASR in products, documentation, and store listings without
creating the impression of an official or endorsed app. Trademark rules are
normative in [TRADEMARKS.md](TRADEMARKS.md); this file is the practical checklist.

## Primary product identity

Third-party products that use OpenASR open-core, the CLI, the local API, or the
iOS/macOS xcframework **must** ship under an independent product identity:

| Element | Third-party requirement |
| --- | --- |
| Product / app name | Not "OpenASR" and not confusingly similar |
| Logo and app icon | Your own artwork; do not copy official icons |
| UI chrome | Your own design system and screens |
| Store listing | Your developer account, screenshots, and description |
| Support | Your own support email, site, or channel |
| Signing | Your certificates, bundle IDs, and package names |

Official OpenASR Desktop and mobile apps are published only by the project
operators. See [TRADEMARKS.md](TRADEMARKS.md#official-applications).

## Attribution that is welcome

Preferred secondary attribution (footer, About screen, documentation):

```text
Powered by OpenASR
https://openasr.org
```

Also fine:

- "Speech recognition uses OpenASR open-core (Apache-2.0)."
- "Compatible with OpenASR `.oasr` model packs."
- "Local API compatible with the OpenASR HTTP subset."

Keep attribution **secondary**: smaller than the product name, not the store
title, not the home-screen label.

## Phrasing to avoid

Do not use copy that suggests official status, for example:

- "Official OpenASR client" / "Official OpenASR for iOS"
- "OpenASR Certified" / "Authorized OpenASR partner" (unless you have a written
  agreement)
- "The OpenASR app" when referring to a third-party build
- App names like "OpenASR Pro", "OpenASR+", "OpenASR Mobile", "OpenASR Cloud"
- Reusing official marketing screenshots or store preview GIFs as if they were
  your product

## Visual marks

- Prefer text attribution ("Powered by OpenASR") over pasting the logo.
- If you need a logo for a dependency wall or conference slide, use only assets
  the project explicitly publishes for that purpose on
  [https://openasr.org](https://openasr.org), and do not alter them in a way that
  suggests a different product.
- Do not place the OpenASR logo more prominently than your own product mark on
  an app icon, splash screen, or store icon.

## Relationship to the open-core license

Branding rules do **not** shrink Apache-2.0 rights in the code. You may still
fork, modify, embed, and redistribute the software under [LICENSE](LICENSE).
You may not use the OpenASR brand as if the fork or embed were the official
product. When you redistribute, keep license notices (`LICENSE`, `NOTICE`) as
Apache-2.0 requires.

## Desktop and web wrappers

Electron, Tauri, SwiftUI, or other shells around the open-core binary or
xcframework are welcome. Each wrapper is a **separate product** for branding
purposes: independent name, UI, and support. Shipping a thin re-skin of the
official OpenASR Desktop UI under another name is fine; shipping it under the
OpenASR name or icon is not.

## Questions

For partnership, co-marketing, or permission to use logos beyond "Powered by
OpenASR", contact the maintainers via [https://openasr.org](https://openasr.org)
or the publishing GitHub organization. Until you have written permission, stay
within the allowed uses in [TRADEMARKS.md](TRADEMARKS.md).
