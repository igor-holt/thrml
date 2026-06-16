export interface TelemetryEntry {
  logId: string;
  utc: string;
  event: string;
  prediction: {
    mode: string;
    p: number;
  };
  outcome: string;
  subjective: string;
}
