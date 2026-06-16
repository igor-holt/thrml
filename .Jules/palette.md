## 2024-03-31 - Visually Hidden Accessibility Text
**Learning:** The `app/globals.css` file does not define utility classes like `.sr-only` for screen readers.
**Action:** When adding screen-reader captions or other accessible text that needs to be visually hidden without breaking layout, use explicit inline CSS styles: `style={{ position: "absolute", width: "1px", height: "1px", padding: 0, margin: "-1px", overflow: "hidden", clip: "rect(0, 0, 0, 0)", whiteSpace: "nowrap", borderWidth: 0 }}`.
