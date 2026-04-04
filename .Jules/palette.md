## 2024-04-04 - Visually Hidden Elements Pattern
**Learning:** The application lacks standard `.sr-only` utility classes for screen-reader-only text, requiring a specific inline style pattern to visually hide elements like table captions without breaking layout.
**Action:** Use explicit inline styles (`position: 'absolute', width: '1px', height: '1px', padding: 0, margin: '-1px', overflow: 'hidden', clip: 'rect(0, 0, 0, 0)', whiteSpace: 'nowrap', borderWidth: 0`) for screen-reader-only text instead of relying on non-existent CSS classes.
