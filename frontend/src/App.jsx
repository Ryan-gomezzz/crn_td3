import { useWebSocket } from './hooks/useWebSocket';
import Dashboard from './components/Dashboard';

const WS_URL = `ws://${window.location.host}/ws`;

export default function App() {
  const { data, connected } = useWebSocket(WS_URL);

  return (
    <div style={{
      background: '#0a0e1a',
      minHeight: '100vh',
      color: '#d4ddf7',
      fontFamily: "'Inter','Segoe UI',system-ui,sans-serif",
      display: 'flex',
      flexDirection: 'column',
    }}>

      {/* ── Header ── */}
      <header style={{
        borderBottom: '1px solid #1c2847',
        padding: '0 24px',
        height: 56,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexShrink: 0,
        background: '#080c18',
      }}>
        <div>
          <div style={{ fontWeight: 700, fontSize: 15, color: '#eaf0ff', letterSpacing: 0.2 }}>
            Cognitive Radio Network — TD3 Power Allocation
          </div>
          <div style={{ fontSize: 11, color: '#5a6e9a', marginTop: 1 }}>
            Ramaiah Institute of Technology &nbsp;·&nbsp; Nakagami-m Fading Channel &nbsp;·&nbsp; BPSK Modulation
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <div style={{ textAlign: 'right', fontSize: 11, color: '#5a6e9a' }}>
            <div>Episode <span style={{ color: '#d4ddf7', fontWeight: 600 }}>{data?.episode_count ?? '—'}</span></div>
            <div>Step <span style={{ color: '#d4ddf7', fontWeight: 600 }}>{data?.step ?? '—'}</span></div>
          </div>
          <div style={{
            fontSize: 11,
            fontWeight: 600,
            letterSpacing: 1.2,
            padding: '4px 10px',
            borderRadius: 3,
            border: `1px solid ${connected ? '#16a34a' : '#dc2626'}`,
            color: connected ? '#4ade80' : '#f87171',
            background: connected ? 'rgba(22,163,74,0.08)' : 'rgba(220,38,38,0.08)',
          }}>
            {connected ? '● LIVE' : '○ OFFLINE'}
          </div>
        </div>
      </header>

      <Dashboard data={data} />
    </div>
  );
}
