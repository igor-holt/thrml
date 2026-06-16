## 2026-02-04 - Rich Data Hidden in Charts
**Learning:** The application's data models (`TelemetryEntry`) contain rich contextual fields (`event`, `mode`) that are often stripped away before reaching visualization components, forcing users to cross-reference tables.
**Action:** When enhancing visualizations in this app, first audit the raw data source to see if valuable context is being discarded during data transformation, and restore it for use in tooltips.
