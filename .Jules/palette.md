## 2024-04-16 - Inline styling required for visually hidden elements
**Learning:** This specific application's design system (defined primarily in `globals.css`) does not provide utility classes like `.sr-only` for visually hiding elements intended for screen readers (like table captions).
**Action:** When elements need to be visually hidden for accessibility without breaking layout, use explicit inline CSS styles: `style={{ position: 'absolute', width: '1px', height: '1px', padding: 0, margin: '-1px', overflow: 'hidden', clip: 'rect(0, 0, 0, 0)', whiteSpace: 'nowrap', borderWidth: 0 }}`.
