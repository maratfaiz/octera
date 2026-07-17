"use client";

interface TrendPoint {
  date: string;
  value: number; // expected range 0..1
}

interface TrendChartProps {
  points: TrendPoint[];
  color?: string;
}

const WIDTH = 320;
const HEIGHT = 96;
const PADDING = 8;

function y(value: number): number {
  const clamped = Math.max(0, Math.min(1, value));
  return HEIGHT - PADDING - clamped * (HEIGHT - PADDING * 2);
}

export function TrendChart({ points, color = "var(--accent)" }: TrendChartProps) {
  if (points.length === 0) {
    return <p style={{ color: "var(--ink-soft)", fontSize: 13 }}>Недостаточно данных для графика.</p>;
  }

  const gradientId = `trend-gradient-${color.replace(/[^a-zA-Z0-9]/g, "")}`;

  if (points.length === 1) {
    const cy = y(points[0].value);
    return (
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} width="100%" height={HEIGHT} preserveAspectRatio="none">
        {[0.25, 0.5, 0.75].map((f) => (
          <line
            key={f}
            x1={PADDING}
            x2={WIDTH - PADDING}
            y1={PADDING + f * (HEIGHT - PADDING * 2)}
            y2={PADDING + f * (HEIGHT - PADDING * 2)}
            stroke="var(--line)"
            strokeWidth={1}
          />
        ))}
        <circle cx={WIDTH / 2} cy={cy} r={4} fill={color} />
      </svg>
    );
  }

  const stepX = (WIDTH - PADDING * 2) / (points.length - 1);
  const coords = points.map((p, i) => ({ x: PADDING + i * stepX, y: y(p.value) }));

  const linePath = coords.map((c, i) => `${i === 0 ? "M" : "L"}${c.x.toFixed(1)},${c.y.toFixed(1)}`).join(" ");
  const last = coords[coords.length - 1];
  const first = coords[0];
  const areaPath = `${linePath} L${last.x.toFixed(1)},${HEIGHT - PADDING} L${first.x.toFixed(1)},${HEIGHT - PADDING} Z`;

  return (
    <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} width="100%" height={HEIGHT} preserveAspectRatio="none">
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.35" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      {[0.25, 0.5, 0.75].map((f) => (
        <line
          key={f}
          x1={PADDING}
          x2={WIDTH - PADDING}
          y1={PADDING + f * (HEIGHT - PADDING * 2)}
          y2={PADDING + f * (HEIGHT - PADDING * 2)}
          stroke="var(--line)"
          strokeWidth={1}
        />
      ))}
      <path d={areaPath} fill={`url(#${gradientId})`} stroke="none" />
      <path d={linePath} fill="none" stroke={color} strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
      <circle cx={last.x} cy={last.y} r={4} fill={color} />
    </svg>
  );
}
