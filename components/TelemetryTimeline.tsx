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

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{
    payload: {
      time: number;
      probability: number;
      event: string;
      outcome: string;
      subjective: string;
    };
  }>;
  label?: number;
}

const CustomTooltip = ({ active, payload, label }: CustomTooltipProps) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    // Helper to format label safely
    const dateLabel = label ? new Date(label).toUTCString() : '';

    return (
      <div className="card" style={{ padding: '12px', fontSize: '0.9rem' }}>
        <p style={{ fontWeight: 'bold', marginBottom: '8px' }}>
          {dateLabel}
        </p>
        <p style={{ color: '#8884d8', marginBottom: '4px' }}>
          <strong>Probability:</strong> {data.probability}
        </p>
        <p style={{ marginBottom: '4px' }}>
          <strong>Event:</strong> {data.event}
        </p>
        <p style={{ marginBottom: '4px' }}>
          <strong>Outcome:</strong> {data.outcome}
        </p>
        {data.subjective && (
          <p style={{ fontStyle: 'italic', marginTop: '8px', color: '#666', borderTop: '1px solid #eee', paddingTop: '4px' }}>
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
            <caption style={{ captionSide: 'top', textAlign: 'left', marginBottom: '10px', fontWeight: 'bold' }}>
              Telemetry Event Log
            </caption>
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
