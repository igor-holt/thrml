## 2024-05-24 - Visually Hidden Elements in THRML
**Learning:** This repository's global CSS (`app/globals.css`) does not define utility classes like `.sr-only` for visually hiding elements while maintaining accessibility.
**Action:** When elements (e.g., screen-reader captions) need to be visually hidden for accessibility without breaking layout, use explicit inline CSS styles: `style={{ position: 'absolute', width: '1px', height: '1px', padding: 0, margin: '-1px', overflow: 'hidden', clip: 'rect(0, 0, 0, 0)', whiteSpace: 'nowrap', borderWidth: 0 }}`.
