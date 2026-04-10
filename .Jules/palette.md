## 2024-05-15 - Missing Utility Classes for Visually Hidden Elements
**Learning:** The `app/globals.css` in this application lacks an `.sr-only` utility class, and it is a strict boundary to not add custom CSS classes.
**Action:** Use an explicit inline style payload (`{{ position: "absolute", width: "1px", height: "1px", padding: 0, margin: "-1px", overflow: "hidden", clip: "rect(0, 0, 0, 0)", whiteSpace: "nowrap", borderWidth: 0 }}`) for components like `<caption>` to ensure they are visually hidden but still available to screen readers.
