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

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div style={{
        backgroundColor: '#fff',
        border: '1px solid #e0e0e0',
        padding: '12px',
        borderRadius: '8px',
        boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
        minWidth: '200px'
      }}>
        <p style={{ fontWeight: 'bold', marginBottom: '8px', fontSize: '0.9rem' }}>
          {label ? new Date(label).toUTCString() : ''}
        </p>
        <p style={{ margin: '4px 0', fontSize: '0.9rem' }}>
          <span style={{ color: '#666' }}>Event:</span> {data.event}
        </p>
        <p style={{ margin: '4px 0', fontSize: '0.9rem' }}>
          <span style={{ color: '#666' }}>Probability:</span> {(data.probability * 100).toFixed(1)}%
        </p>
        {data.subjective && (
          <p style={{
            marginTop: '8px',
            fontStyle: 'italic',
            color: '#666',
            fontSize: '0.85em',
            borderTop: '1px solid #eee',
            paddingTop: '8px'
          }}>
            "{data.subjective}"
          </p>
        )}
      </div>
    );
  }
  return null;
};

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
              <Tooltip content={<CustomTooltip />} />
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
                <th style={{ padding: '10px', textAlign: 'left' }}>Log ID</th>
                <th style={{ padding: '10px', textAlign: 'left' }}>UTC</th>
                <th style={{ padding: '10px', textAlign: 'left' }}>Event</th>
                <th style={{ padding: '10px', textAlign: 'left' }}>Prediction (p)</th>
              </tr>
            </thead>
            <tbody>
              {telemetryData.map((entry) => (
                <tr key={entry.logId} style={{ borderBottom: '1px solid #ddd' }}>
                  <td style={{ padding: '10px' }}>{entry.logId}</td>
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
