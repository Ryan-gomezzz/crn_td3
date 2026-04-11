/**
 * CRN Network Topology — animated signal transmission and interference.
 *
 * Desired links  (PT→PR, ST→SR):  animated packets travel along the path.
 *   - PT→PR packet speed and color reflect PU SINR health.
 *   - ST→SR packet speed, count, and link width reflect P_s and R_s.
 *
 * Interference links (ST→PR, PT→SR): animated "burst" discs, distinct style.
 *
 * Each link is labelled with its channel gain (h_xy) and current dB value.
 */

import { useEffect, useRef, useState } from 'react';

/* ── Layout constants ─────────────────────────────────────────────────────── */
const W = 540, H = 430;

const N = {
  PT: { x: 105, y: 100 },
  PR: { x: 435, y: 100 },
  ST: { x: 105, y: 330 },
  SR: { x: 435, y: 330 },
};

const SINR_THRESH_DB = 10 * Math.log10(2.0);   // ≈ 3.01 dB

/* ── Animated travelling packets along a straight link ────────────────────── */
function TravellingPackets({ x1, y1, x2, y2, color, count, durationS, r = 4 }) {
  return (
    <>
      {Array.from({ length: count }, (_, i) => {
        const begin = `${-(i * durationS / count).toFixed(2)}s`;
        const path  = `M${x1},${y1} L${x2},${y2}`;
        return (
          <circle key={i} r={r} fill={color} opacity={0.9}>
            <animateMotion dur={`${durationS}s`} repeatCount="indefinite"
                           begin={begin} path={path} />
          </circle>
        );
      })}
    </>
  );
}

/* ── Interference "burst" packets — smaller, transparent, faster ──────────── */
function InterferencePackets({ x1, y1, x2, y2, color, count = 2, durationS = 1.2 }) {
  return (
    <>
      {Array.from({ length: count }, (_, i) => {
        const begin = `${-(i * durationS / count).toFixed(2)}s`;
        const path  = `M${x1},${y1} L${x2},${y2}`;
        return (
          <g key={i}>
            <circle r={3} fill={color} opacity={0.55}>
              <animateMotion dur={`${durationS}s`} repeatCount="indefinite"
                             begin={begin} path={path} />
            </circle>
          </g>
        );
      })}
    </>
  );
}

/* ── Animated dashed stroke — gives "flow" appearance on a link ───────────── */
function FlowingDash({ x1, y1, x2, y2, stroke, strokeWidth, opacity,
                       dashArray, animDuration, reverse = false }) {
  return (
    <line x1={x1} y1={y1} x2={x2} y2={y2}
          stroke={stroke}
          strokeWidth={strokeWidth}
          strokeOpacity={opacity}
          strokeDasharray={dashArray}
          style={{
            animation: `${reverse ? 'dash-flow-interf' : 'dash-flow'} ${animDuration}s linear infinite`,
          }} />
  );
}

/* ── Node component ── */
function Node({ x, y, color, label, fullName, metric, metricColor, metricLabel }) {
  return (
    <g>
      {/* Outer ring */}
      <circle cx={x} cy={y} r={34} fill="none" stroke={color} strokeWidth={1.5} opacity={0.3} />
      {/* Body */}
      <circle cx={x} cy={y} r={26} fill="#0c1322" stroke={color} strokeWidth={2} />
      {/* Label */}
      <text x={x} y={y + 1} textAnchor="middle" dominantBaseline="middle"
            fill={color} fontSize={13} fontWeight="700" fontFamily="'Inter',system-ui">
        {label}
      </text>
      {/* Full name below */}
      <text x={x} y={y + 52} textAnchor="middle" fill="#4a5e88" fontSize={9.5}
            fontFamily="'Inter',system-ui">
        {fullName}
      </text>
      {/* Live metric */}
      {metric != null && (
        <text x={x} y={y + 65} textAnchor="middle" fill={metricColor || '#7a8fb8'} fontSize={9}
              fontFamily="'Inter',system-ui">
          {metricLabel}: <tspan fontWeight="600">{metric}</tspan>
        </text>
      )}
    </g>
  );
}

/* ── Link label sitting along the midpoint ── */
function LinkLabel({ x1, y1, x2, y2, offsetX = 0, offsetY = -10,
                     text, subtext, color = '#7a8fb8' }) {
  const mx = (x1 + x2) / 2 + offsetX;
  const my = (y1 + y2) / 2 + offsetY;
  return (
    <g>
      <text x={mx} y={my} textAnchor="middle" fill={color}
            fontSize={9} fontFamily="'Inter',system-ui" fontStyle="italic">
        {text}
      </text>
      {subtext && (
        <text x={mx} y={my + 11} textAnchor="middle" fill={color} opacity={0.75}
              fontSize={8.5} fontFamily="'Inter',system-ui">
          {subtext}
        </text>
      )}
    </g>
  );
}

/* ── Signal quality bar at a receiver ── */
function SINRBar({ cx, cy, sinrDb, label }) {
  const clamped  = Math.max(0, Math.min(1, (sinrDb + 5) / 35));  // map [-5,30] → [0,1]
  const barColor = sinrDb >= SINR_THRESH_DB
    ? `hsl(${120 * clamped}, 70%, 55%)`
    : '#ef4444';

  return (
    <g>
      <rect x={cx - 22} y={cy - 5} width={44} height={7} rx={2}
            fill="#0d1528" stroke="#1c2847" strokeWidth={1} />
      <rect x={cx - 22} y={cy - 5} width={44 * clamped} height={7} rx={2}
            fill={barColor} opacity={0.85} />
      <text x={cx} y={cy + 14} textAnchor="middle" fill="#4a5e88"
            fontSize={8} fontFamily="'Inter',system-ui">
        {label}: {sinrDb?.toFixed(1) ?? '—'} dB
      </text>
    </g>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════ */
export default function NetworkTopology({ data }) {
  const sinr_p_db   = data?.sinr_p_db ?? 0;
  const sinr_s_db   = data?.sinr_s_db ?? 0;
  const p_s         = data?.p_s       ?? 0;
  const throughput  = data?.throughput ?? 0;
  const puOk        = sinr_p_db >= SINR_THRESH_DB;

  /* Packet animation params derived from live metrics */
  const puSpeed   = Math.max(0.8, 3.5 - sinr_p_db * 0.08);   // faster = better SINR
  const suSpeed   = Math.max(0.8, 3.5 - throughput * 0.4);
  const suCount   = Math.max(1, Math.round(1 + p_s * 4));     // more packets = more power
  const interfSpd = 0.9 + (1 - p_s) * 0.6;                   // faster = less interference

  const puColor   = puOk ? '#4ade80' : '#f87171';
  const suColor   = '#60a5fa';
  const interf1   = '#fb923c';   // ST→PR
  const interf2   = '#f472b6';   // PT→SR

  return (
    <div style={{
      background: '#080c18',
      border: '1px solid #1c2847',
      borderRadius: 8,
      overflow: 'hidden',
    }}>

      {/* Section header */}
      <div style={{
        padding: '8px 14px',
        borderBottom: '1px solid #1c2847',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
      }}>
        <span style={{ fontSize: 11, fontWeight: 600, color: '#8ea4cc',
                       textTransform: 'uppercase', letterSpacing: 0.8 }}>
          CRN System Model  —  Signal Propagation
        </span>
        <span style={{ fontSize: 10, color: '#3a4e70' }}>
          4-node topology  ·  block-fading
        </span>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block' }}>

        {/* ── Background grid (subtle) ── */}
        <defs>
          <pattern id="grid" width="30" height="30" patternUnits="userSpaceOnUse">
            <path d="M 30 0 L 0 0 0 30" fill="none" stroke="#0f1830" strokeWidth="0.5" />
          </pattern>
        </defs>
        <rect width={W} height={H} fill="url(#grid)" />

        {/* ══════════════════════════════════════════════════════
            DESIRED LINK 1 : PT → PR   (Primary system)
        ══════════════════════════════════════════════════════ */}

        {/* Static base line */}
        <line x1={N.PT.x} y1={N.PT.y} x2={N.PR.x} y2={N.PR.y}
              stroke={puColor} strokeWidth={1.5} strokeOpacity={0.2} />

        {/* Animated flowing dashes */}
        <FlowingDash x1={N.PT.x} y1={N.PT.y} x2={N.PR.x} y2={N.PR.y}
                     stroke={puColor} strokeWidth={2.5} opacity={0.6}
                     dashArray="10 8" animDuration={puSpeed * 0.6} />

        {/* Travelling data packets */}
        <TravellingPackets x1={N.PT.x} y1={N.PT.y} x2={N.PR.x} y2={N.PR.y}
                           color={puColor} count={3} durationS={puSpeed} r={4.5} />

        {/* Link label */}
        <LinkLabel x1={N.PT.x} y1={N.PT.y} x2={N.PR.x} y2={N.PR.y}
                   offsetY={-14} text="h_pp  (Primary Desired)"
                   subtext={`SINR_p = ${sinr_p_db.toFixed(1)} dB  ${puOk ? '✓' : '✗'}`}
                   color={puColor} />

        {/* ══════════════════════════════════════════════════════
            DESIRED LINK 2 : ST → SR   (Secondary system)
        ══════════════════════════════════════════════════════ */}

        <line x1={N.ST.x} y1={N.ST.y} x2={N.SR.x} y2={N.SR.y}
              stroke={suColor} strokeWidth={1 + p_s * 3} strokeOpacity={0.2} />

        <FlowingDash x1={N.ST.x} y1={N.ST.y} x2={N.SR.x} y2={N.SR.y}
                     stroke={suColor} strokeWidth={2 + p_s * 2} opacity={0.55}
                     dashArray="10 8" animDuration={suSpeed * 0.6} />

        <TravellingPackets x1={N.ST.x} y1={N.ST.y} x2={N.SR.x} y2={N.SR.y}
                           color={suColor} count={suCount} durationS={suSpeed} r={4} />

        <LinkLabel x1={N.ST.x} y1={N.ST.y} x2={N.SR.x} y2={N.SR.y}
                   offsetY={14} text="h_ss  (Secondary Desired)"
                   subtext={`R_s = ${throughput.toFixed(2)} b/s/Hz  ·  P_s = ${p_s.toFixed(2)} W`}
                   color={suColor} />

        {/* ══════════════════════════════════════════════════════
            INTERFERENCE LINK 1 : ST → PR
            Secondary transmitter interfering with Primary receiver
        ══════════════════════════════════════════════════════ */}

        <line x1={N.ST.x} y1={N.ST.y} x2={N.PR.x} y2={N.PR.y}
              stroke={interf1} strokeWidth={1} strokeOpacity={0.15}
              strokeDasharray="6 5" />

        <FlowingDash x1={N.ST.x} y1={N.ST.y} x2={N.PR.x} y2={N.PR.y}
                     stroke={interf1} strokeWidth={1.5} opacity={0.45}
                     dashArray="5 6" animDuration={interfSpd} reverse={false} />

        <InterferencePackets x1={N.ST.x} y1={N.ST.y} x2={N.PR.x} y2={N.PR.y}
                             color={interf1} count={2} durationS={interfSpd + 0.3} />

        <LinkLabel x1={N.ST.x} y1={N.ST.y} x2={N.PR.x} y2={N.PR.y}
                   offsetX={-28} offsetY={-6}
                   text="h_sp  (Interference)"
                   subtext="ST → PR"
                   color={interf1} />

        {/* ══════════════════════════════════════════════════════
            INTERFERENCE LINK 2 : PT → SR
            Primary transmitter interfering with Secondary receiver
        ══════════════════════════════════════════════════════ */}

        <line x1={N.PT.x} y1={N.PT.y} x2={N.SR.x} y2={N.SR.y}
              stroke={interf2} strokeWidth={1} strokeOpacity={0.15}
              strokeDasharray="6 5" />

        <FlowingDash x1={N.PT.x} y1={N.PT.y} x2={N.SR.x} y2={N.SR.y}
                     stroke={interf2} strokeWidth={1.5} opacity={0.4}
                     dashArray="5 6" animDuration={interfSpd + 0.2} reverse={false} />

        <InterferencePackets x1={N.PT.x} y1={N.PT.y} x2={N.SR.x} y2={N.SR.y}
                             color={interf2} count={2} durationS={interfSpd + 0.5} />

        <LinkLabel x1={N.PT.x} y1={N.PT.y} x2={N.SR.x} y2={N.SR.y}
                   offsetX={28} offsetY={-6}
                   text="h_ps  (Interference)"
                   subtext="PT → SR"
                   color={interf2} />

        {/* ══════════════════════════════════════════════════════
            NODES
        ══════════════════════════════════════════════════════ */}

        <Node x={N.PT.x} y={N.PT.y} color="#2563eb" label="PT"
              fullName="Primary Transmitter"
              metric="1.00 W" metricColor="#7a9bd8" metricLabel="P_p" />

        <Node x={N.PR.x} y={N.PR.y} color={puColor} label="PR"
              fullName="Primary Receiver" />

        <Node x={N.ST.x} y={N.ST.y} color="#dc2626" label="ST"
              fullName="Secondary Transmitter"
              metric={`${p_s.toFixed(3)} W`} metricColor="#fb923c" metricLabel="P_s" />

        <Node x={N.SR.x} y={N.SR.y} color="#7c3aed" label="SR"
              fullName="Secondary Receiver" />

        {/* ── SINR bars at receivers ── */}
        <SINRBar cx={N.PR.x} cy={N.PR.y + 36} sinrDb={sinr_p_db} label="SINR_p" />
        <SINRBar cx={N.SR.x} cy={N.SR.y + 36} sinrDb={sinr_s_db} label="SINR_s" />

        {/* ── PU violation indicator ── */}
        {!puOk && (
          <g>
            <circle cx={N.PR.x + 20} cy={N.PR.y - 22} r={7}
                    fill="#dc2626" opacity={0.9}>
              <animate attributeName="opacity" values="0.9;0.3;0.9" dur="0.8s" repeatCount="indefinite" />
            </circle>
            <text x={N.PR.x + 20} y={N.PR.y - 22} textAnchor="middle"
                  dominantBaseline="middle" fill="#fff" fontSize={8} fontWeight="bold">!</text>
          </g>
        )}

        {/* ── Legend ── */}
        <g transform={`translate(10, ${H - 50})`}>
          <rect width={W - 20} height={42} rx={4}
                fill="#080c18" stroke="#1c2847" strokeWidth={1} />
          <g transform="translate(10, 12)">
            {/* Desired PU */}
            <line x1={0} y1={6} x2={22} y2={6} stroke="#4ade80" strokeWidth={2.5} />
            <circle cx={11} cy={6} r={3.5} fill="#4ade80" />
            <text x={26} y={10} fill="#8ea4cc" fontSize={9} fontFamily="'Inter',system-ui">Primary signal (h_pp)</text>

            {/* Desired SU */}
            <line x1={150} y1={6} x2={172} y2={6} stroke="#60a5fa" strokeWidth={2.5} />
            <circle cx={161} cy={6} r={3.5} fill="#60a5fa" />
            <text x={176} y={10} fill="#8ea4cc" fontSize={9} fontFamily="'Inter',system-ui">Secondary signal (h_ss)</text>

            {/* Interference 1 */}
            <line x1={310} y1={6} x2={332} y2={6} stroke="#fb923c" strokeWidth={1.5} strokeDasharray="5 4" />
            <circle cx={321} cy={6} r={2.5} fill="#fb923c" />
            <text x={336} y={10} fill="#8ea4cc" fontSize={9} fontFamily="'Inter',system-ui">Interference ST→PR (h_sp)</text>

            {/* Interference 2 */}
            <line x1={0} y1={22} x2={22} y2={22} stroke="#f472b6" strokeWidth={1.5} strokeDasharray="5 4" />
            <circle cx={11} cy={22} r={2.5} fill="#f472b6" />
            <text x={26} y={26} fill="#8ea4cc" fontSize={9} fontFamily="'Inter',system-ui">Interference PT→SR (h_ps)</text>

            <text x={310} y={26} fill="#4a5e88" fontSize={9} fontFamily="'Inter',system-ui">
              γ_th = {(SINR_THRESH_DB).toFixed(1)} dB  ·  Nakagami-m fading (m=1)
            </text>
          </g>
        </g>

      </svg>
    </div>
  );
}
