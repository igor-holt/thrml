## 2024-03-18 - Missing Table Semantics in Data Components
**Learning:** Data tables in this application often lack essential semantic HTML for screen readers, such as `scope="col"` on header cells and `<caption>` elements, which makes navigating raw telemetry data difficult for non-visual users.
**Action:** When working with data visualization or log tables, always add `scope="col"` to `<th>` elements and include a visually-hidden `<caption>` describing the table structure to ensure screen reader accessibility.
