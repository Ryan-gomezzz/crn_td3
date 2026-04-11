/* Compact academic metrics bar — one row of labelled values */

const ROW = {
  display: 'flex',
  gap: 0,
  borderRadius: 6,
  border: '1px solid #1c2847',
  overflow: 'hidden',
  background: '#0c1120',
};

const CELL = {
  flex: 1,
  padding: '8px 14px',
  borderRight: '1px solid #1c2847',
};

const CLABEL = { fontSize: 10, color: '#4a5e88', textTransform: 'uppercase', letterSpacing: 0.8 };
const CVAL   = { fontSize: 18, fontWeight: 700, marginTop: 1, lineHeight: 1.2 };
const CSUB   = { fontSize: 10, color: '#3a4e70', marginTop: 2 };

function Cell({ label, value, color = '#d4ddf7', sub }) {
  return (
    <div style={CELL}>
      <div style={CLABEL}>{label}</div>
      <div style={{ ...CVAL, color }}>{value ?? '—'}</div>
      {sub && <div style={CSUB}>{sub}</div>}
    </div>
  );
}

export default function StatsPanel({ data }) {
  const cr = data?.constraint_rate;
  const crColor = cr == null ? '#d4ddf7' : cr >= 0.85 ? '#4ade80' : cr >= 0.6 ? '#fbbf24' : '#f87171';

  return (
    <div style={ROW}>
      <Cell
        label="Avg100 Reward"
        value={data?.avg100_reward?.toFixed(3)}
        color="#60a5fa"
        sub={data?.reward_trend ?? 'collecting...'}
      />
      <Cell
        label="PU Constraint"
        value={cr != null ? (cr * 100).toFixed(1) + '%' : null}
        color={crColor}
        sub={`SINR_p ≥ γ_th  (${data?.sinr_p_db?.toFixed(1) ?? '—'} dB)`}
      />
      <Cell
        label="SU Throughput  R_s"
        value={data?.throughput?.toFixed(3)}
        color="#34d399"
        sub="log₂(1 + SINR_s)  [bits/s/Hz]"
      />
      <Cell
        label="Instantaneous BER"
        value={data?.ber != null ? data.ber.toExponential(2) : null}
        color="#a78bfa"
        sub="½ · erfc(√SINR_s)"
      />
      <Cell
        label="Outage Probability"
        value={data?.outage_prob?.toFixed(4)}
        color="#f59e0b"
        sub="P(SINR_s < γ_th)  500-step window"
      />
      <Cell
        label="ST Power  P_s"
        value={data?.p_s?.toFixed(3) + ' W'}
        color="#fb923c"
        sub={`Max P_max = 1.0 W  |  Stage: ${data?.training_stage ?? '—'}`}
      />
      <Cell
        label="SINR_s"
        value={data?.sinr_s_db?.toFixed(1) + ' dB'}
        color="#e879f9"
        sub="SU received SINR"
      />
    </div>
  );
}
