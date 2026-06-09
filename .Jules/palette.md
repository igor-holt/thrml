## 2024-05-05 - Data Table Accessibility with CSS

**Learning:** When adding `<caption>` elements to tables for accessibility, if utility classes like `.sr-only` are not available in the design system, using explicit inline CSS `clip` and `position: absolute` is an effective pattern to provide screen reader context without disrupting the visual layout.
**Action:** Use this inline CSS block (`style={{ position: 'absolute', width: '1px', height: '1px', padding: 0, margin: '-1px', overflow: 'hidden', clip: 'rect(0, 0, 0, 0)', whiteSpace: 'nowrap', borderWidth: 0 }}`) for visually hiding elements when global utility classes are missing, while ensuring semantic HTML structures like `<caption>` and `scope="col"` are implemented.
