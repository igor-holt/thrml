## 2024-03-24 - Screen Reader Only Text Implementation

**Learning:** When implementing visually hidden screen reader descriptions (e.g. for `<caption>` or other a11y context) in this web application, standard utility classes like `.sr-only` are not available via `app/globals.css`.

**Action:** Continue using an explicit inline styles object for visual hiding when standard UI libraries/utility classes are not available. Use the explicit style: `style={{ position: 'absolute', width: '1px', height: '1px', padding: 0, margin: '-1px', overflow: 'hidden', clip: 'rect(0, 0, 0, 0)', whiteSpace: 'nowrap', borderWidth: 0 }}`.
