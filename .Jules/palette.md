## 2024-05-18 - Missing Utility Classes
**Learning:** The `app/globals.css` file does not provide utility classes like `.sr-only`.
**Action:** Use explicit inline CSS styles for visually hidden elements (like screen-reader captions) to ensure accessibility without breaking layout (e.g., `style={{ position: "absolute", width: "1px", height: "1px", padding: 0, margin: "-1px", overflow: "hidden", clip: "rect(0, 0, 0, 0)", whiteSpace: "nowrap", borderWidth: 0 }}`).
