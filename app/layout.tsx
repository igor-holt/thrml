import './globals.css';

export const metadata = {
  title: 'THRML Telemetry',
  description: 'Telemetry visualization for THRML',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
