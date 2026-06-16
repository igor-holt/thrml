## 2026-02-25 - Contextual Telemetry Visualization

**Learning:** When visualizing complex telemetry data, relying on a simple line chart with only numerical values in tooltips forces users to constantly cross-reference the raw data table below to understand the *cause* (Event) and *context* (Subjective notes) of spikes or dips. Adding these qualitative fields directly into the chart tooltip significantly reduces cognitive load and makes the visualization self-contained. Furthermore, simple HTML data tables often lack basic accessibility features; adding `scope="col"` to `<th>` elements and a descriptive `<caption>` provides critical structure for screen reader users with minimal effort.

**Action:** Always enrich chart tooltips with relevant metadata from the dataset (like event names, outcomes, or notes), not just the X and Y coordinates. Ensure every data table has a `<caption>` and proper `scope` attributes on headers to improve accessibility for screen readers.
