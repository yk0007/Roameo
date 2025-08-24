import { useMemo } from "react";

type Props = {
  destination?: string | null;
  variant?: "aura" | "stamp" | "topo";
  className?: string;
};

function hashToInt(str: string, mod = 360) {
  let h = 0;
  for (let i = 0; i < str.length; i++) h = (h * 31 + str.charCodeAt(i)) | 0;
  return Math.abs(h) % mod;
}

function huePair(seed: string) {
  const h1 = hashToInt(seed, 360);
  const h2 = (h1 + 60 + (hashToInt(seed + "x", 180))) % 360;
  return [h1, h2];
}

export default function DestinationCardArt({ destination = "Somewhere", variant = "aura", className = "" }: Props) {
  const seed = destination?.trim() || "Somewhere";
  const [h1, h2] = useMemo(() => huePair(seed), [seed]);
  const a = useMemo(() => 35 + (hashToInt(seed + "a", 40)), [seed]);  // angle for gradients
  const r = useMemo(() => 20 + (hashToInt(seed + "r", 40)), [seed]);  // radius-ish for shapes

  if (variant === "stamp") return <Stamp seed={seed} h1={h1} h2={h2} className={className} />;
  if (variant === "topo")  return <Topo seed={seed} h1={h1} h2={h2} className={className} />;

  return <Aura seed={seed} h1={h1} h2={h2} a={a} r={r} className={className} />; // default aura
}

/* ---------- Variant 1: GEO-AURA (soft animated gradient + grain) ---------- */
function Aura({ seed, h1, h2, a, r, className = "" }: any) {
  const id = `aura-${seed.replace(/\s+/g, "-")}`;
  return (
    <div className={`relative w-full h-full ${className}`}>
      <svg viewBox="0 0 800 450" className="w-full h-full">
        <defs>
          <radialGradient id={`${id}-rg1`} cx="30%" cy="30%">
            <stop offset="0%" stopColor={`hsl(${h1} 90% 65%)`} />
            <stop offset="100%" stopColor={`hsl(${h2} 70% 20%)`} />
          </radialGradient>

          <filter id={`${id}-grain`}>
            <feTurbulence baseFrequency="0.8" type="fractalNoise" numOctaves="2" stitchTiles="stitch" />
            <feColorMatrix type="saturate" values="0" />
            <feComponentTransfer>
              <feFuncA type="linear" slope="0.08" />
            </feComponentTransfer>
          </filter>
        </defs>

        <rect width="100%" height="100%" fill={`hsl(${h2} 30% 9%)`} />
        <g style={{ transformOrigin: "400px 225px", animation: "spin 24s linear infinite" }}>
          <circle cx="400" cy="225" r={120 + r} fill={`url(#${id}-rg1)`} opacity="0.85" />
          <circle cx="520" cy="160" r="90" fill={`hsl(${h1} 90% 60% / 0.35)`} />
          <circle cx="300" cy="290" r="70" fill={`hsl(${h2} 80% 55% / 0.30)`} />
        </g>

        <text x="32" y="404" fill="white" opacity="0.9" fontSize="24" fontWeight="700" style={{ letterSpacing: "0.02em" }}>
          {seed}
        </text>
        <rect width="100%" height="100%" filter={`url(#${id}-grain)`} opacity="0.65" />
      </svg>

      <style jsx>{`
        @keyframes spin { from { rotate: 0deg; } to { rotate: ${a}deg; } }
      `}</style>
    </div>
  );
}

/* ---------- Variant 2: TRAVEL STAMP (vintage oval + wavy cancel lines) ---------- */
function Stamp({ seed, h1, h2, className = "" }: any) {
  const id = `stamp-${seed.replace(/\s+/g, "-")}`;
  return (
    <div className={`relative w-full h-full ${className}`}>
      <svg viewBox="0 0 800 450" className="w-full h-full">
        <defs>
          <linearGradient id={`${id}-bg`} x1="0" x2="1" y1="0" y2="1">
            <stop offset="0%" stopColor={`hsl(${h1} 60% 90%)`} />
            <stop offset="100%" stopColor={`hsl(${h2} 60% 85%)`} />
          </linearGradient>
          <filter id={`${id}-paper`}>
            <feTurbulence baseFrequency="0.9" type="fractalNoise" numOctaves="2" stitchTiles="stitch" />
            <feColorMatrix type="saturate" values="0.1" />
            <feComponentTransfer><feFuncA type="linear" slope="0.05" /></feComponentTransfer>
          </filter>
        </defs>

        <rect width="100%" height="100%" fill={`url(#${id}-bg)`} />
        {/* Wavy cancel lines */}
        {[0, 1, 2, 3].map((i) => (
          <path key={i}
            d={`M-50 ${120 + i*22} q 80 -20 160 0 t 160 0 t 160 0 t 160 0 t 160 0`}
            stroke={`hsl(${h2} 40% 40% / 0.4)`} strokeWidth="2" fill="none" />
        ))}

        {/* Oval stamp */}
        <g transform="translate(400,230)">
          <ellipse rx="180" ry="110" fill="none" stroke={`hsl(${h1} 50% 30%)`} strokeWidth="4" />
          <ellipse rx="170" ry="100" fill="none" stroke={`hsl(${h1} 40% 35%)`} strokeDasharray="6 8" />
          <text textAnchor="middle" y="-10" fontFamily="ui-monospace" fontWeight="700" fontSize="28" fill={`hsl(${h1} 50% 25%)`}>
            {seed.toUpperCase()}
          </text>
          <text textAnchor="middle" y="22" fontSize="14" letterSpacing="0.25em" fill={`hsl(${h1} 35% 25%)`}>
            ADMIT • ONE • TRAVEL
          </text>
        </g>

        {/* Tiny plane animation */}
        <g>
          <path id={`${id}-path`} d="M40,360 C200,260 600,340 760,260" fill="none" stroke="none"/>
          <g>
            <text fontSize="20" fill={`hsl(${h1} 40% 25%)`}>
              <textPath href={`#${id}-path`} startOffset="0%">
                ✈️ ✈️ ✈️
              </textPath>
            </text>
            <animateTransform
              attributeName="transform"
              type="translate"
              values="0,0; 720,0"
              dur="10s"
              repeatCount="indefinite"
            />
          </g>
        </g>

        <rect width="100%" height="100%" filter={`url(#${id}-paper)`} opacity="0.5" />
      </svg>
    </div>
  );
}

/* ---------- Variant 3: TOPO GRID (contours + compass) ---------- */
function Topo({ seed, h1, h2, className = "" }: any) {
  const id = `topo-${seed.replace(/\s+/g, "-")}`;
  return (
    <div className={`relative w-full h-full ${className}`}>
      <svg viewBox="0 0 800 450" className="w-full h-full">
        <defs>
          <pattern id={`${id}-grid`} width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M40 0 H0 V40" stroke={`hsl(${h2} 25% 65% / 0.35)`} strokeWidth="1" />
          </pattern>
          <filter id={`${id}-noise`}>
            <feTurbulence baseFrequency="0.015" numOctaves="2" type="fractalNoise" />
            <feColorMatrix type="saturate" values="0" />
            <feComponentTransfer><feFuncA type="linear" slope="0.05"/></feComponentTransfer>
          </filter>
        </defs>

        <rect width="100%" height="100%" fill={`hsl(${h1} 55% 92%)`} />
        <rect width="100%" height="100%" fill={`url(#${id}-grid)`} />

        {[0,1,2,3,4,5].map(i => (
          <path key={i}
            d={`M 20 ${120+i*40} C 160 ${90+i*40} 300 ${150+i*40} 440 ${120+i*40} S 720 ${210+i*40} 780 ${180+i*40}`}
            fill="none"
            stroke={`hsl(${h1} 45% ${30 + i*5}% / 0.5)`}
            strokeWidth={i === 2 ? 2.5 : 1.5}
          />
        ))}

        {/* Compass rose */}
        <g transform="translate(720,80)">
          <circle r="22" fill="white" stroke={`hsl(${h1} 40% 35%)`} />
          <g fill={`hsl(${h1} 60% 40%)`}>
            <polygon points="0,-18 4,0 0,8 -4,0" />
            <polygon points="0,18 -4,0 0,-8 4,0" opacity="0.6" />
          </g>
          <text textAnchor="middle" y="6" fontSize="10" fill={`hsl(${h1} 35% 25%)`}>N</text>
        </g>

        <text x="24" y="420" fontWeight="700" fontSize="20" fill={`hsl(${h2} 40% 25%)`}>{seed}</text>
        <rect width="100%" height="100%" filter={`url(#${id}-noise)`} />
      </svg>
    </div>
  );
}
