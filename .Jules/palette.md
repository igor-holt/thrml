## 2024-04-26 - Accessible Data Tables
**Learning:** Data tables in this app's components (like TelemetryTimeline) lack proper semantic structure, specifically `scope="col"` on headers and visually hidden `<caption>` elements, which is critical for screen reader users to understand the table's context.
**Action:** Always include a `<caption>` (using standard visually-hidden inline styles since `.sr-only` is unavailable) and `scope="col"` attributes on `<th>` elements to ensure robust screen reader navigation.
