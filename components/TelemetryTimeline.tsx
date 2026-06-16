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
      <div className="custom-tooltip" style={{ backgroundColor: '#fff', border: '1px solid #ccc', padding: '10px' }}>
        <p className="label" style={{ fontWeight: 'bold', marginBottom: '5px' }}>{new Date(label).toUTCString()}</p>
        <p className="intro" style={{ margin: 0 }}>{`Event: ${data.event}`}</p>
        <p className="intro" style={{ margin: 0 }}>{`Mode: ${data.mode}`}</p>
        <p className="desc" style={{ margin: 0, color: '#8884d8' }}>{`Probability: ${data.probability}`}</p>
      </div>
    );
  }
  return null;
};

const TelemetryTimeline: React.FC = () => {
  const chartData = getProbabilityData();

  if (telemetryData.length === 0) {
    return (
      <div style={{ padding: '20px' }}>
        <Card>
          <h2>Telemetry Timeline</h2>
          <div style={{ textAlign: 'center', padding: '40px', color: '#666' }}>No telemetry data available</div>
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
                <th scope="col" style={{ padding: '10px', textAlign: 'left' }}>Log ID</th>
                <th scope="col" style={{ padding: '10px', textAlign: 'left' }}>UTC</th>
                <th scope="col" style={{ padding: '10px', textAlign: 'left' }}>Event</th>
                <th scope="col" style={{ padding: '10px', textAlign: 'left' }}>Prediction (p)</th>
              </tr>
            </thead>
            <tbody>
              {telemetryData.map((entry) => (
                <tr key={entry.logId} style={{ borderBottom: '1px solid #ddd' }}>
                  <td scope="row" style={{ padding: '10px' }}>{entry.logId}</td>
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
