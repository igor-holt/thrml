## 2023-10-24 - Accessible Data Tables
**Learning:** Data tables in this app's components (like TelemetryTimeline) lack basic accessibility semantics like `<caption>` and `scope="col"`. This makes screen reader navigation and context understanding difficult.
**Action:** Always verify that tabular data structures include `<caption>` to describe their contents and `scope="col"`/`scope="row"` on header cells for proper column/row association.
