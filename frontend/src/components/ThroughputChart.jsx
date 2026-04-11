import { useState, useEffect, useRef } from 'react';
import {
  ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts';

const MAX_PTS = 600;
const TICK  = { fill: '#5a6e9a', fontSize: 10, fontFamily: "'Inter',system-ui" };
const TITLE = { fill: '#5a6e9a', fontSize: 10, fontFamily: "'Inter',system-ui" };

export default function ThroughputChart({ data }) {
  const histRef = useRef([]);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    if (!data) return;
    histRef.current = [
      ...histRef.current.slice(-(MAX_PTS - 1)),
      { step: data.step, rs: data.throughput },
    ];
    setHistory([...histRef.current]);
  }, [data]);

  const latest = data?.throughput?.toFixed(3);

  return (
    <div style={{ background: '#080c18', border: '1px solid #1c2847', borderRadius: 8 }}>
      <div style={{ padding: '8px 14px', borderBottom: '1px solid #1c2847',
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: 11, fontWeight: 600, color: '#8ea4cc',
                       textTransform: 'uppercase', letterSpacing: 0.8 }}>
          Secondary User Throughput  —  R_s
        </span>
        <span style={{ fontSize: 10, color: '#3a4e70' }}>
          R_s = log₂(1 + SINR_s) &nbsp;·&nbsp;
          current: <span style={{ color: '#34d399', fontWeight: 600 }}>{latest ?? '—'} b/s/Hz</span>
        </span>
      </div>
      <div style={{ padding: '8px 8px 4px' }}>
        <ResponsiveContainer width="100%" height={185}>
          <ComposedChart data={history} margin={{ top: 8, right: 24, bottom: 28, left: 14 }}>
            <defs>
              <linearGradient id="rsGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor="#34d399" stopOpacity={0.2} />
                <stop offset="95%" stopColor="#34d399" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="#0f1830" strokeDasharray="3 3" />
            <XAxis dataKey="step" tick={TICK}
                   label={{ value: 'Environment Step', position: 'insideBottom',
                            offset: -14, style: TITLE }} />
            <YAxis domain={[0, 'auto']} tick={TICK}
                   label={{ value: 'R_s  [bits/s/Hz]', angle: -90,
                            position: 'insideLeft', offset: 10, style: TITLE }} />
            <ReferenceLine y={0} stroke="#1c2847" />
            <Tooltip
              contentStyle={{ background: '#0c1322', border: '1px solid #1c2847',
                              fontSize: 11, fontFamily: "'Inter',system-ui" }}
              labelStyle={{ color: '#5a6e9a' }}
              itemStyle={{ color: '#34d399' }}
              formatter={(v) => [v.toFixed(4), 'R_s (b/s/Hz)']}
            />
            <Area type="monotone" dataKey="rs" stroke="none"
                  fill="url(#rsGrad)" isAnimationActive={false} />
            <Line type="monotone" dataKey="rs" stroke="#34d399"
                  strokeWidth={1.5} dot={false} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
