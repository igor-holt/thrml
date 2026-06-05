## 2024-06-25 - Visually Hidden Elements in App
**Learning:** The `app/globals.css` file lacks common utility classes like `.sr-only` for visually hiding elements while maintaining screen reader accessibility.
**Action:** When elements (e.g., table captions) need to be visually hidden for accessibility without breaking the visual layout, explicitly apply an inline CSS style dictionary: `style={{ position: 'absolute', width: '1px', height: '1px', padding: 0, margin: '-1px', overflow: 'hidden', clip: 'rect(0, 0, 0, 0)', whiteSpace: 'nowrap', borderWidth: 0 }}`.
