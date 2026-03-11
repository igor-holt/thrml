## 2024-03-10 - Add semantic HTML tags and visually hidden captions for screen-reader accessibility
**Learning:** Tables can easily omit captions for visual reasons, resulting in poor screen-reader accessibility. Simple `scope="col"` attributes on `<th>` elements and visually hidden `<caption>` using inline styles can significantly improve accessibility without breaking the layout.
**Action:** Always consider screen-reader visibility for tables by ensuring `scope="col"` on headers and visually hidden captions using the inline CSS pattern when layout constraints prevent visible captions.
