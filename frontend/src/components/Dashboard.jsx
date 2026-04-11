import StatsPanel      from './StatsPanel';
import NetworkTopology from './NetworkTopology';
import BerVsSinrChart  from './BerVsSinrChart';
import ThroughputChart from './ThroughputChart';
import OutageProbChart from './OutageProbChart';

export default function Dashboard({ data }) {
  return (
    <div style={{
      display: 'grid',
      gridTemplateRows: 'auto 1fr',
      gridTemplateColumns: '1fr',
      gap: 10,
      padding: 10,
      height: 'calc(100vh - 56px)',
      boxSizing: 'border-box',
    }}>
      {/* ── Metrics bar ── */}
      <StatsPanel data={data} />

      {/* ── Main body: topology left, charts right ── */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '480px 1fr',
        gap: 10,
        minHeight: 0,
        overflow: 'hidden',
      }}>
        {/* CRN topology — fills full height */}
        <div style={{ overflow: 'hidden' }}>
          <NetworkTopology data={data} />
        </div>

        {/* Three charts stacked */}
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: 10,
          minHeight: 0,
          overflow: 'auto',
        }}>
          <BerVsSinrChart  scatter={data?.scatter} />
          <ThroughputChart data={data} />
          <OutageProbChart data={data} />
        </div>
      </div>
    </div>
  );
}
