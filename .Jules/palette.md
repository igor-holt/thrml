## 2026-05-04 - Handling Visually Hidden Elements in THRML
**Learning:** The thrml web app lacks utility classes for visually hiding elements (like `.sr-only`) since it doesn't use Tailwind or similar frameworks. Simply adding an element for screen readers can break layouts without proper styling.
**Action:** Use a standardized inline CSS string to visually hide elements for screen readers without breaking the layout: `style={{ position: 'absolute', width: '1px', height: '1px', padding: 0, margin: '-1px', overflow: 'hidden', clip: 'rect(0, 0, 0, 0)', whiteSpace: 'nowrap', borderWidth: 0 }}`.
