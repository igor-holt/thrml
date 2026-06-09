## 2025-02-18 - Visually hiding elements in thrml web UI
**Learning:** The application lacks a standard `.sr-only` utility class (like in Tailwind) in its global CSS. Visually hiding semantic elements for screen readers (e.g. `<caption>` on tables) without breaking layout requires explicit inline CSS strings.
**Action:** Use an inline style object for `.sr-only` behavior: `style={{ position: 'absolute', width: '1px', height: '1px', padding: 0, margin: '-1px', overflow: 'hidden', clip: 'rect(0, 0, 0, 0)', whiteSpace: 'nowrap', borderWidth: 0 }}`.
