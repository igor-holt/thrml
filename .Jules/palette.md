## 2026-03-28 - Visually Hidden Elements for Accessibility
**Learning:** The `app/globals.css` file does not define utility classes like `.sr-only` for visually hiding content for screen readers.
**Action:** When elements (e.g., screen-reader captions) need to be visually hidden for accessibility without breaking layout, use explicit inline CSS styles: `style={{ position: 'absolute', width: '1px', height: '1px', padding: 0, margin: '-1px', overflow: 'hidden', clip: 'rect(0, 0, 0, 0)', whiteSpace: 'nowrap', borderWidth: 0 }}`.
