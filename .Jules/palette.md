
## 2024-05-18 - Visually Hidden Text in Next.js Custom Styles
**Learning:** This application lacks utility classes like Tailwind's `.sr-only` in `globals.css` for visually hiding text for screen readers. Standard accessibility improvements (like adding table captions) require inline styling to avoid breaking layouts.
**Action:** Always provide explicit inline CSS strings (e.g., `clip: 'rect(0, 0, 0, 0)'`, `position: 'absolute'`, etc.) for screen reader-only text and ensure elements like `<th>` include the correct `scope="col"` attribute to explicitly define table layout.
