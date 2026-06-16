## 2026-04-20 - Data Table Accessibility and Visually Hidden Text
**Learning:** Tables in this project require a visually hidden `<caption>` for context and `scope="col"` on header cells for accessibility, but the global CSS lacks utility classes like `.sr-only`.
**Action:** When making elements visually accessible to screen readers without breaking layout, apply explicit inline CSS properties for clipping (`clip: rect(0, 0, 0, 0)` and absolute positioning with `1px` dimensions) directly.
