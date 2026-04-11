/**
 * BER vs SINR_s — scatter of live BPSK measurements + theoretical Nakagami-m=1 curve.
 * Y-axis is log₁₀(BER), represented as the exponent (e.g. -3 means BER = 10⁻³).
 */
import { useMemo } from 'react';
import {
  ComposedChart, Scatter, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine,
} from 'recharts';
import { nakagamiBerCurve } from '../lib/nakagami';

const THEORY = nakagamiBerCurve(1, -5, 25, 80).map(p => ({
  x: parseFloat(p.x.toFixed(2)),
  y: parseFloat(Math.log10(Math.max(p.y, 1e-7)).toFixed(3)),
}));

const TICK  = { fill: '#5a6e9a', fontSize: 10, fontFamily: "'Inter',system-ui" };
const TITLE = { fill: '#5a6e9a', fontSize: 10, fontFamily: "'Inter',system-ui" };

function BerTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  if (!d) return null;
  return (
    <div style={{ background: '#0c1322', border: '1px solid #1c2847', padding: '6px 10px',
                  borderRadius: 4, fontSize: 11, fontFamily: "'Inter',system-ui" }}>
      <div style={{ color: '#5a6e9a' }}>SINR_s = <b style={{ color: '#d4ddf7' }}>{d.x?.toFixed(2)} dB</b></div>
      <div style={{ color: '#5a6e9a' }}>BER ≈ <b style={{ color: '#d4ddf7' }}>10<sup>{d.y?.toFixed(2)}</sup></b></div>
    </div>
  );
}

export default function BerVsSinrChart({ scatter }) {
  const live = useMemo(() =>
    (scatter ?? []).map(p => ({
      x: p.x,
      y: parseFloat(Math.log10(Math.max(p.y, 1e-7)).toFixed(3)),
    })),
    [scatter]
  );

  return (
    <div style={{ background: '#080c18', border: '1px solid #1c2847', borderRadius: 8 }}>
      <div style={{ padding: '8px 14px', borderBottom: '1px solid #1c2847',
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: 11, fontWeight: 600, color: '#8ea4cc',
                       textTransform: 'uppercase', letterSpacing: 0.8 }}>
          BER Performance  —  P_b vs SINR_s
        </span>
        <span style={{ fontSize: 10, color: '#3a4e70' }}>
          BPSK  ·  P_b = ½·erfc(√γ_s)  ·  Nakagami-m (m=1)
        </span>
      </div>
      <div style={{ padding: '8px 8px 4px' }}>
        <ResponsiveContainer width="100%" height={210}>
          <ComposedChart margin={{ top: 8, right: 24, bottom: 28, left: 14 }}>
            <CartesianGrid stroke="#0f1830" strokeDasharray="3 3" />
            <XAxis dataKey="x" type="number" domain={[-5, 25]}
                   tick={TICK}
                   label={{ value: 'SINR_s  [dB]', position: 'insideBottom', offset: -14, style: TITLE }} />
            <YAxis dataKey="y" type="number" domain={[-7, 0]}
                   tickFormatter={v => `10${v > -1 ? '' : String.fromCharCode(0x207B)}${Math.abs(v)}`}
                   tick={TICK}
                   label={{ value: 'BER  [log₁₀]', angle: -90, position: 'insideLeft',
                            offset: 10, style: TITLE }} />
            <ReferenceLine x={10 * Math.log10(2.0)} stroke="#fbbf24" strokeDasharray="4 3"
                           strokeWidth={1} opacity={0.6}
                           label={{ value: 'γ_th', fill: '#fbbf24', fontSize: 9, position: 'top' }} />
            <Tooltip content={<BerTooltip />} />
            <Legend iconSize={8}
                    wrapperStyle={{ fontSize: 10, color: '#5a6e9a', paddingTop: 4,
                                    fontFamily: "'Inter',system-ui" }} />
            <Line name="Theoretical (Nakagami m=1)" data={THEORY} dataKey="y"
                  dot={false} stroke="#f59e0b" strokeWidth={2} legendType="line" />
            <Scatter name="Measured BER" data={live} dataKey="y"
                     fill="#60a5fa" opacity={0.5} r={2.5} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
