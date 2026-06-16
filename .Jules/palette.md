## 2026-03-24 - Visually hiding elements without utility classes
**Learning:** This app's global CSS does not define utility classes like `.sr-only`. Elements like screen-reader captions that need to be visually hidden must use explicit inline CSS styles (`position: absolute`, `width: 1px`, etc.) to hide them without breaking layout.
**Action:** Use inline styles to visually hide elements when accessibility requires hidden text in this codebase, instead of relying on common framework utility classes.
