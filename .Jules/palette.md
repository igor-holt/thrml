## 2026-03-25 - Standardizing visually hidden styles for screen readers
**Learning:** This app's styling configuration doesn't define standard utility classes like `.sr-only` for visually hiding elements without breaking layout (e.g. for table captions).
**Action:** Use the exact inline style string (`{{ position: 'absolute', width: '1px', height: '1px', padding: 0, margin: '-1px', overflow: 'hidden', clip: 'rect(0, 0, 0, 0)', whiteSpace: 'nowrap', borderWidth: 0 }}`) for screen reader-only text instead of adding custom CSS.
