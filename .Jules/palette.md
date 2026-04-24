## 2024-05-18 - Missing global utility classes for accessibility
**Learning:** This repository lacks global utility CSS classes like `.sr-only` that are often standard in other environments. Using external or assumed classes will break styling silently.
**Action:** Always use explicit inline CSS styles (`style={{ position: "absolute", width: "1px", height: "1px", padding: 0, margin: "-1px", overflow: "hidden", clip: "rect(0, 0, 0, 0)", whiteSpace: "nowrap", borderWidth: 0 }}`) for visually hidden but screen-reader accessible elements to ensure proper rendering.
