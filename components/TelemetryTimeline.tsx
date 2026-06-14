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
  TooltipProps,
} from 'recharts';

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="card" style={{
        padding: '12px',
        fontFamily: 'sans-serif',
        maxWidth: '300px'
      }}>
        <p style={{ margin: '0 0 8px', fontWeight: 'bold', fontSize: '14px', color: '#333' }}>
          {label ? new Date(label).toUTCString() : ''}
        </p>
        <p style={{ margin: '0 0 4px', fontSize: '13px', color: '#0066cc', fontWeight: 600 }}>
          {data.event}
        </p>
        <p style={{ margin: '0 0 8px', fontSize: '12px', color: '#666' }}>
          Mode: {data.mode} | Outcome: {data.outcome}
        </p>
        <p style={{ margin: '0 0 8px', fontSize: '13px', fontWeight: 'bold', color: '#333' }}>
          Probability: {data.probability}
        </p>
        <div style={{ borderTop: '1px solid #eee', paddingTop: '8px', marginTop: '8px' }}>
          <p style={{ margin: 0, fontSize: '12px', fontStyle: 'italic', color: '#555' }}>
            "{data.subjective}"
          </p>
        </div>
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
              <Tooltip content={CustomTooltip} />
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
