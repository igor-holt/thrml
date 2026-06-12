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
                content={({ active, payload, label }) => {
                  if (active && payload && payload.length) {
                    return (
                      <div style={{ backgroundColor: '#fff', padding: '10px', border: '1px solid #e0e0e0', borderRadius: '4px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
                        <p style={{ margin: 0, fontWeight: 'bold', fontSize: '0.9rem' }}>{label ? new Date(label).toLocaleString() : ''}</p>
                        <p style={{ margin: '4px 0 0', color: '#666', fontSize: '0.85rem' }}>{`Event: ${payload[0].payload.event}`}</p>
                        <p style={{ margin: 0, color: '#8884d8', fontSize: '0.85rem' }}>{`Probability: ${(Number(payload[0].value) * 100).toFixed(0)}%`}</p>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Legend />
              <Line type="monotone" dataKey="probability" stroke="#8884d8" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        
        {/* Optional: Raw Table for Details */}
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
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
                  <td style={{ padding: '10px', fontFamily: 'monospace' }}>{entry.logId}</td>
                  <td style={{ padding: '10px' }}>{new Date(entry.utc).toLocaleString()}</td>
                  <td style={{ padding: '10px' }}>{entry.event}</td>
                  <td style={{ padding: '10px' }}>{(entry.prediction.p * 100).toFixed(0)}%</td>
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
