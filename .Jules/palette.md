## 2026-04-05 - Visually Hidden Accessibility Elements
**Learning:** This codebase does not include a `.sr-only` or similar utility class in its `globals.css` for visually hiding elements intended only for screen readers.
**Action:** When adding visually hidden elements (like table captions), use inline styles (`style={{ position: 'absolute', width: '1px', height: '1px', padding: 0, margin: '-1px', overflow: 'hidden', clip: 'rect(0, 0, 0, 0)', whiteSpace: 'nowrap', borderWidth: 0 }}`) to ensure they are accessible without breaking layout.
