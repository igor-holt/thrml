## 2026-04-21 - Visually Hidden Elements for Accessibility
**Learning:** This application lacks utility classes like `.sr-only` for visually hiding elements intended only for screen readers (like table captions).
**Action:** When an element needs to be visually hidden but remain accessible to screen readers, use explicit inline styles: `style={{ position: 'absolute', width: '1px', height: '1px', padding: 0, margin: '-1px', overflow: 'hidden', clip: 'rect(0, 0, 0, 0)', whiteSpace: 'nowrap', borderWidth: 0 }}`.
