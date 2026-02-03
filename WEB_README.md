# THRML Telemetry Visualization

This directory contains a Next.js web application for visualizing telemetry data from the THRML library.

## Features

- 📊 Interactive Recharts line chart showing probability trends over time
- 📅 Time-series visualization with formatted UTC timestamps
- 📋 Detailed data table with all telemetry entries
- 🎨 Responsive design with clean UI components
- ⚡ Built with Next.js 16 App Router and TypeScript

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

Install dependencies:

```bash
npm install
```

### Development

Run the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the home page, then navigate to [http://localhost:3000/telemetry](http://localhost:3000/telemetry) to see the telemetry visualization.

### Build

Build for production:

```bash
npm run build
npm start
```

## Project Structure

```
├── app/
│   ├── globals.css          # Global styles
│   ├── layout.tsx           # Root layout component
│   ├── page.tsx             # Home page
│   └── telemetry/
│       └── page.tsx         # Telemetry visualization page
├── components/
│   ├── Card.tsx             # Reusable card component
│   └── TelemetryTimeline.tsx # Main telemetry chart component
├── data/
│   ├── telemetry.ts         # Telemetry data and utilities
│   └── types.ts             # TypeScript type definitions
├── next.config.js           # Next.js configuration
├── tsconfig.json            # TypeScript configuration
└── package.json             # Project dependencies
```

## Technologies

- **Next.js 16**: React framework with App Router
- **React 19**: UI library
- **TypeScript**: Type-safe JavaScript
- **Recharts**: Charting library for data visualization
- **CSS**: Custom styling for components

## Data Model

The telemetry data includes:
- **logId**: Unique identifier for each entry
- **utc**: UTC timestamp
- **event**: Type of event detected
- **prediction**: Object containing mode and probability (p)
- **outcome**: Result of the event
- **subjective**: Descriptive notes

## Extending

To add more telemetry data, edit `data/telemetry.ts` and add new entries to the `telemetryData` array following the `TelemetryEntry` interface defined in `data/types.ts`.

To customize the chart, modify the Recharts components in `components/TelemetryTimeline.tsx`.

## License

See the main repository LICENSE file.
