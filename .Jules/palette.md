## 2026-04-18 - Data Table Accessibility
**Learning:** React data tables in this project often lack accessibility attributes like `<caption>` and `scope` because the app avoids external CSS frameworks and utilities like `.sr-only`, leading developers to skip visually hidden screen reader texts.
**Action:** Always add a visually hidden `<caption>` with explicit inline styling to tables to provide context for screen readers, and add `scope="col"` to `<th>` elements to associate headers with column data.
