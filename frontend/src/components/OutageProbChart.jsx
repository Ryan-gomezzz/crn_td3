import { useState, useEffect, useRef } from 'react';
import {
  ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts';

const MAX_PTS = 600;
const TICK  = { fill: '#5a6e9a', fontSize: 10, fontFamily: "'Inter',system-ui" };
const TITLE = { fill: '#5a6e9a', fontSize: 10, fontFamily: "'Inter',system-ui" };

export default function OutageProbChart({ data }) {
  const histRef = useRef([]);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    if (!data) return;
    histRef.current = [
      ...histRef.current.slice(-(MAX_PTS - 1)),
      { step: data.step, pout: data.outage_prob, cr: 1 - data.outage_prob },
    ];
    setHistory([...histRef.current]);
  }, [data]);

  const latest = data?.outage_prob?.toFixed(4);

  return (
    <div style={{ background: '#080c18', border: '1px solid #1c2847', borderRadius: 8 }}>
      <div style={{ padding: '8px 14px', borderBottom: '1px solid #1c2847',
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: 11, fontWeight: 600, color: '#8ea4cc',
                       textTransform: 'uppercase', letterSpacing: 0.8 }}>
          Outage Probability  —  P_out
        </span>
        <span style={{ fontSize: 10, color: '#3a4e70' }}>
          P(SINR_s &lt; γ_th)  ·  500-step window &nbsp;·&nbsp;
          current: <span style={{ color: '#a78bfa', fontWeight: 600 }}>{latest ?? '—'}</span>
        </span>
      </div>
      <div style={{ padding: '8px 8px 4px' }}>
        <ResponsiveContainer width="100%" height={185}>
          <ComposedChart data={history} margin={{ top: 8, right: 24, bottom: 28, left: 14 }}>
            <defs>
              <linearGradient id="poutGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor="#a78bfa" stopOpacity={0.2} />
                <stop offset="95%" stopColor="#a78bfa" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="#0f1830" strokeDasharray="3 3" />
            <XAxis dataKey="step" tick={TICK}
                   label={{ value: 'Environment Step', position: 'insideBottom',
                            offset: -14, style: TITLE }} />
            <YAxis domain={[0, 1]} tick={TICK}
                   label={{ value: 'P_out', angle: -90,
                            position: 'insideLeft', offset: 10, style: TITLE }} />
            {/* Common target outage threshold */}
            <ReferenceLine y={0.1} stroke="#f59e0b" strokeDasharray="4 3" strokeWidth={1} opacity={0.7}
                           label={{ value: 'P_out = 0.1', fill: '#f59e0b', fontSize: 8,
                                    position: 'insideTopRight' }} />
            <Tooltip
              contentStyle={{ background: '#0c1322', border: '1px solid #1c2847',
                              fontSize: 11, fontFamily: "'Inter',system-ui" }}
              labelStyle={{ color: '#5a6e9a' }}
              formatter={(v, name) => [v.toFixed(4), name === 'pout' ? 'P_out' : 'P_success']}
            />
            <Area type="monotone" dataKey="pout" stroke="none"
                  fill="url(#poutGrad)" isAnimationActive={false} />
            <Line type="monotone" dataKey="pout" stroke="#a78bfa"
                  strokeWidth={1.5} dot={false} isAnimationActive={false} name="pout" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
