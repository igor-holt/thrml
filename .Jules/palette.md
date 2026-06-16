## 2024-03-29 - Initial findings

## 2024-03-29 - Recharts Tooltip Customization & Table A11y
**Learning:** When customizing Recharts `<Tooltip>` components, TypeScript errors around `TooltipProps` (specifically missing `payload` properties in older versions) can be mitigated by typing the custom tooltip props as `any`. Visually hiding elements (like `<caption>` for tables) without global utility classes like `.sr-only` requires explicit inline CSS styling to maintain layout flow while remaining accessible to screen readers.
**Action:** Use inline styles `position: 'absolute', width: '1px', height: '1px', padding: 0, margin: '-1px', overflow: 'hidden', clip: 'rect(0, 0, 0, 0)', whiteSpace: 'nowrap', borderWidth: 0` for visually hidden elements in repositories lacking utility classes. Leverage existing global CSS classes (e.g., `.card`) within Recharts custom components to maintain visual consistency.
