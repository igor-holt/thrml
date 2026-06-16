## 2025-02-18 - Accessible Data Tables
**Learning:** Converting the first cell of a table row to `<th scope="row">` is a powerful way to provide context for screen readers. To maintain visual consistency with other data cells, apply `font-weight: normal` and ensuring text alignment matches.
**Action:** Always evaluate if the first column of a data table acts as a key identifier and upgrade it to a row header.

## 2025-02-18 - Next.js Accessibility Utilities
**Learning:** `sr-only` is not built-in to standard CSS or basic Next.js setups. Adding a reusable `.sr-only` utility class in `globals.css` is essential for adding hidden descriptions (like table captions) without impacting design.
**Action:** Check for and add `.sr-only` utility in new projects if missing.
