## 2024-04-06 - Visually Hidden Elements for Accessibility
**Learning:** In codebases without a utility framework (like Tailwind's `sr-only`), a complex set of inline styles (`position: absolute`, `width: 1px`, `clip: rect(0,0,0,0)`, etc.) is required to visually hide an element while keeping it accessible to screen readers.
**Action:** Use this exact inline style block when needing to visually hide screen reader elements (like table captions) to maintain accessibility without breaking visual layout.
