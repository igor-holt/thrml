## 2024-07-25 - Data Table Accessibility
**Learning:** Data tables require scope attributes on headers and captions for screen readers, but global utility classes like .sr-only might not exist in every Next.js project.
**Action:** Use inline styles (position: 'absolute', width: '1px', overflow: 'hidden') to visually hide captions when .sr-only is missing, and explicitly add scope="col" to table headers.
