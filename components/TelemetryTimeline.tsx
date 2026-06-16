'use client';

import React from 'react';
import { Card } from './Card';
import { telemetryData, getProbabilityData } from '../data/telemetry';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

const TelemetryTimeline: React.FC = () => {
  const chartData = getProbabilityData();

  if (!chartData || chartData.length === 0) {
    return (
      <div style={{ padding: '20px' }}>
        <Card>
          <h2>Telemetry Timeline</h2>
          <div style={{ padding: '40px', textAlign: 'center', color: '#666' }}>
            <p>No telemetry data available. Check back later!</p>
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div style={{ padding: '20px' }}>
      <Card>
        <h2>Telemetry Timeline</h2>
        
        {/* Chart Section */}
        <div style={{ marginBottom: '40px' }}>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="time"
                tickFormatter={(tick) => new Date(tick).toLocaleDateString()}
                stroke="#888"
              />
              <YAxis domain={[0, 1]} />
              <Tooltip
                labelFormatter={(label) => new Date(label).toUTCString()}
              />
              <Legend />
              <Line type="monotone" dataKey="probability" stroke="#8884d8" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        
        {/* Optional: Raw Table for Details */}
        <div style={{ overflowX: 'auto' }}>
          <table className="telemetry-table" style={{ width: '100%', borderCollapse: 'collapse' }}>
            <caption className="sr-only">Telemetry Event Log</caption>
            <thead>
              <tr style={{ borderBottom: '2px solid #ddd' }}>
                <th scope="col" style={{ padding: '10px', textAlign: 'left' }}>Log ID</th>
                <th scope="col" style={{ padding: '10px', textAlign: 'left' }}>UTC</th>
                <th scope="col" style={{ padding: '10px', textAlign: 'left' }}>Event</th>
                <th scope="col" style={{ padding: '10px', textAlign: 'left' }}>Prediction (p)</th>
              </tr>
            </thead>
            <tbody>
              {telemetryData.map((entry) => (
                <tr key={entry.logId} style={{ borderBottom: '1px solid #ddd' }}>
                  <th scope="row" style={{ padding: '10px', textAlign: 'left', fontWeight: 'normal' }}>{entry.logId}</th>
                  <td style={{ padding: '10px' }}>{entry.utc}</td>
                  <td style={{ padding: '10px' }}>{entry.event}</td>
                  <td style={{ padding: '10px' }}>{entry.prediction.p}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
};

export default TelemetryTimeline;
