## 2024-05-24 - Visually Hidden Elements for Accessibility
**Learning:** The application lacks utility CSS classes like `.sr-only` for visually hiding elements intended for screen readers (e.g., table captions).
**Action:** When an element must be hidden visually but available for screen readers, use explicit inline styles: `style={{ position: 'absolute', width: '1px', height: '1px', padding: 0, margin: '-1px', overflow: 'hidden', clip: 'rect(0, 0, 0, 0)', whiteSpace: 'nowrap', borderWidth: 0 }}`.
