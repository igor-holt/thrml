import { TelemetryEntry } from './types';

export const telemetryData: TelemetryEntry[] = [
  {
    logId: '0x004F',
    utc: '2026-02-01T03:14:15Z',
    event: 'HighRadiationLikeSpike',
    prediction: { mode: 'Shear-2', p: 0.61 },
    outcome: 'TBD',
    subjective: 'Cold. The static tastes like blue geometry.',
  },
  {
    logId: '0x0050',
    utc: '2026-02-02T14:22:30Z',
    event: 'LedgerBurnAlert',
    prediction: { mode: 'Nominal', p: 0.85 },
    outcome: 'Mitigated',
    subjective: 'Warmth fading; credits conserved.',
  },
  {
    logId: '0x0051',
    utc: '2026-02-02T18:45:00Z',
    event: 'SubstrateDensityAnomaly',
    prediction: { mode: 'Resonant-A', p: 0.73 },
    outcome: 'Monitored',
    subjective: 'Patterns echo in the void.',
  },
  {
    logId: '0x0052',
    utc: '2026-02-03T02:10:22Z',
    event: 'QuantumFluxSpike',
    prediction: { mode: 'Cascade-3', p: 0.92 },
    outcome: 'TBD',
    subjective: 'The flux sings in harmonics.',
  },
  {
    logId: '0x0053',
    utc: '2026-02-03T08:33:15Z',
    event: 'EnergyWellDepletion',
    prediction: { mode: 'Critical', p: 0.45 },
    outcome: 'Recharging',
    subjective: 'Dim corridors, faint hum.',
  },
  {
    logId: '0x0054',
    utc: '2026-02-03T12:15:47Z',
    event: 'SeismicEventDetected',
    prediction: { mode: 'Shear-1', p: 0.78 },
    outcome: 'Logged',
    subjective: 'Ground trembles with ancient memory.',
  },
];

// Utility to extract chart data
export function getProbabilityData() {
  return telemetryData.map(entry => ({
    time: new Date(entry.utc).getTime(), // For x-axis
    probability: entry.prediction.p,
    event: entry.event,
    subjective: entry.subjective,
  }));
}
