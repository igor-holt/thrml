## 2026-04-07 - Visually Hidden Accessibility Elements
**Learning:** This codebase (specifically `app/globals.css`) lacks standard accessibility utility classes like `.sr-only`.
**Action:** When creating visually hidden elements for screen readers (like table captions), apply explicit inline CSS styles: `style={{ position: 'absolute', width: '1px', height: '1px', padding: 0, margin: '-1px', overflow: 'hidden', clip: 'rect(0, 0, 0, 0)', whiteSpace: 'nowrap', borderWidth: 0 }}`.
