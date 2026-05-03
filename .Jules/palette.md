## 2024-05-24 - Screen Reader Only Text Pattern
**Learning:** The application does not define utility classes like `.sr-only` for visually hiding text for screen readers without breaking layout.
**Action:** Use an explicit inline CSS object with styles like `position: absolute`, `width: 1px`, `height: 1px`, `padding: 0`, `margin: -1px`, `overflow: hidden`, `clip: rect(0, 0, 0, 0)`, `whiteSpace: nowrap`, and `borderWidth: 0` when adding accessibility captions or text that needs to be visually hidden.
