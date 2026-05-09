## 2024-05-15 - Inline CSS for screen-reader text
**Learning:** The Next.js app lacks standard utility classes like `.sr-only` for visually hiding content (e.g. `<caption>` elements) without breaking layout.
**Action:** When hiding text for screen readers, apply explicit inline styling (`{{ position: "absolute", width: "1px", height: "1px", padding: 0, margin: "-1px", overflow: "hidden", clip: "rect(0, 0, 0, 0)", whiteSpace: "nowrap", borderWidth: 0 }}`) directly to the element.
