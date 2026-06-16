import Link from 'next/link';

export default function Home() {
  return (
    <div style={{ padding: '40px', maxWidth: '800px', margin: '0 auto' }}>
      <h1>THRML - Thermodynamic HypergRaphical Model Library</h1>
      <p style={{ marginTop: '20px', lineHeight: '1.6' }}>
        THRML is a JAX library for building and sampling probabilistic graphical models,
        with a focus on efficient block Gibbs sampling and energy-based models.
      </p>
      
      <div style={{ marginTop: '40px' }}>
        <h2>Telemetry Dashboard</h2>
        <p style={{ marginTop: '10px' }}>
          View real-time telemetry data and visualizations:
        </p>
        <Link href="/telemetry" style={{ 
          display: 'inline-block',
          marginTop: '20px',
          padding: '10px 20px',
          backgroundColor: '#0070f3',
          color: 'white',
          textDecoration: 'none',
          borderRadius: '5px'
        }}>
          Go to Telemetry Timeline
        </Link>
      </div>
    </div>
  );
}
