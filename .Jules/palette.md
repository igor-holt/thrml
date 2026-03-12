## 2024-03-24 - Initial setup\n**Learning:** Creating the palette journal\n**Action:** Starting UX review

## 2024-03-24 - Visually Hidden Elements Without Utility Classes
**Learning:** The global CSS does not include utility classes like `.sr-only`, making visually hidden screen reader elements slightly harder to write. Since I should use existing global CSS classes, I need to use inline styles for these screen-reader-only elements to avoid adding custom CSS and breaking the layout.
**Action:** Use specific inline CSS styles: `style={{ position: 'absolute', width: '1px', height: '1px', padding: 0, margin: '-1px', overflow: 'hidden', clip: 'rect(0, 0, 0, 0)', whiteSpace: 'nowrap', borderWidth: 0 }}` for visually hidden screen-reader elements in React components when `.sr-only` is unavailable.
