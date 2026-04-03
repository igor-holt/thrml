## 2025-02-18 - Accessibility improvements to data tables
**Learning:** Recharts components use specific roles and standard HTML tables lack built-in accessible names without captions. Visually hidden styles are needed for captions when global utility classes aren't available.
**Action:** Always add scope="col" to table headers and visually hidden captions using inline styles to improve screen reader experience when displaying data tables.
