## 2026-02-04 - Screen-reader only styles without utility classes
**Learning:** This Next.js repository doesn't include common utility classes like `.sr-only` out of the box in `app/globals.css`.
**Action:** When making elements visually hidden for screen readers without breaking layout (like table captions), use the explicit inline style: `style={{ position: 'absolute', width: '1px', height: '1px', padding: 0, margin: '-1px', overflow: 'hidden', clip: 'rect(0, 0, 0, 0)', whiteSpace: 'nowrap', borderWidth: 0 }}`.
