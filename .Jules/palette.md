## 2024-03-30 - Telemetry Table Accessibility
## 2024-03-30 - Visually Hidden Accessibility Elements
**Learning:** The `app/globals.css` file does not define utility classes like `.sr-only` for visually hiding elements intended for screen readers (like table captions).
**Action:** Used explicit inline CSS styles `style={{ position: 'absolute', width: '1px', height: '1px', padding: 0, margin: '-1px', overflow: 'hidden', clip: 'rect(0, 0, 0, 0)', whiteSpace: 'nowrap', borderWidth: 0 }}` to visually hide accessibility elements without breaking the layout.
