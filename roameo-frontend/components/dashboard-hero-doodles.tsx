"use client";

type DashboardHeroDoodlesProps = {
  variant?: "background" | "hero";
};

export function DashboardHeroDoodles({
  variant = "background"
}: DashboardHeroDoodlesProps) {
  const isHero = variant === "hero";

  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden">
      <svg
        viewBox="0 0 1440 1100"
        aria-hidden="true"
        className="absolute inset-0 h-full w-full"
        preserveAspectRatio="none"
      >
        <defs>
          <linearGradient id="hero-doodle-fade" x1="0%" x2="100%" y1="0%" y2="0%">
            <stop
              offset="0%"
              stopColor={isHero ? "rgba(255,255,255,0.28)" : "rgba(15,23,42,0.14)"}
            />
            <stop
              offset="55%"
              stopColor={isHero ? "rgba(255,255,255,0.12)" : "rgba(71,85,105,0.08)"}
            />
            <stop
              offset="100%"
              stopColor={isHero ? "rgba(255,255,255,0.24)" : "rgba(15,23,42,0.14)"}
            />
          </linearGradient>
        </defs>

        <g
          fill="none"
          stroke="url(#hero-doodle-fade)"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={isHero ? "1.6" : "1.8"}
          opacity={isHero ? "0.72" : "0.95"}
        >
          <g className="hero-doodle hero-doodle-slow">
            <path d="M110 812h146" />
            <path d="M152 812 185 642 218 812" />
            <path d="M167 756h36" />
            <path d="M160 718h50" />
            <path d="M158 681h54" />
            <path d="M169 642h32" />
            <path d="M176 625h18" />
          </g>

          <g className="hero-doodle hero-doodle-fast">
            <path d="M1170 320h164" />
            <path d="M1222 320V160" />
            <path d="M1258 320V134" />
            <path d="M1207 184h66" />
            <path d="M1218 160h44" />
            <circle cx="1240" cy="160" r="16" />
            <path d="M1240 160v-8m0 8 6 4" />
            <path d="M1220 134h40" />
          </g>

          <g className="hero-doodle hero-doodle-medium">
            <path d="M972 856h166" />
            <path d="M996 856c16-42 34-64 58-64 23 0 42 22 58 64" />
            <path d="M986 822h146" />
            <path d="M1020 790h78" />
            <path d="M1034 768h50" />
            <path d="M1055 748h8m18 0h8" />
            <path d="M1058 768v-20m24 20v-20" />
            <path d="M1016 856v-34m78 34v-34" />
            <path d="M1042 790v-22m34 22v-22" />
          </g>

          <g className="hero-doodle hero-doodle-slow">
            <path d="M248 238h148" />
            <path d="M270 238V132h102v106" />
            <path d="M284 132c8-26 24-40 37-40 14 0 29 14 36 40" />
            <path d="M270 154h102" />
            <path d="M302 176v62m18-62v62m18-62v62" />
            <path d="M314 120h14" />
          </g>

          <g className="hero-doodle hero-doodle-medium">
            <path d="M102 478c22-10 46-13 73-8 23 4 43 4 60-2" />
            <path d="M96 495c27-11 53-14 79-10 26 4 47 4 66-2" />
            <path d="M90 512c31-12 59-15 87-11 28 5 50 5 71-1" />
            <path d="M164 444c8 6 15 14 21 24" />
            <path d="M186 438c10 7 18 15 24 27" />
            <path d="M212 444c7 5 13 12 18 20" />
          </g>

          <g className="hero-doodle hero-doodle-fast">
            <path d="M1140 676c18 14 30 35 35 62" />
            <path d="M1141 677c-8 24-8 46 0 64" />
            <path d="M1141 687c-14-12-30-16-49-13" />
            <path d="M1141 698c15-9 29-11 44-8" />
            <path d="M1168 747c9-22 22-39 39-50" />
            <path d="M1168 748c-3-18-1-34 6-49" />
            <path d="M1168 706c-12-9-26-11-42-7" />
            <path d="M1168 718c11-5 22-6 33-4" />
          </g>

          <g className="hero-doodle hero-doodle-slow">
            <path d="M972 132c18-7 36-6 54 4 18 11 34 13 49 6" />
            <path d="M968 151c22-8 42-7 60 4 18 10 35 12 54 5" />
            <path d="M964 170c25-10 48-9 67 3 18 11 37 13 59 6" />
          </g>

          <g className="hero-doodle hero-doodle-medium">
            <path d="M612 954h144" />
            <path d="M636 954 664 904 688 954" />
            <path d="M674 954 704 886 736 954" />
            <path d="M658 954v34m46-34v34" />
            <path d="M620 924h126" />
            <path d="M628 894h104" />
          </g>

          <g className="hero-doodle hero-doodle-fast">
            <path d="M388 842c18-20 45-31 82-31 36 0 63 10 81 31" />
            <path d="M406 842c3-34 16-50 39-50 18 0 31 12 39 37" />
            <path d="M466 829c8-24 22-37 42-37 23 0 37 17 42 50" />
            <path d="M430 842c0 17-12 30-28 30-15 0-27-13-27-30 0-16 12-29 27-29 16 0 28 13 28 29Z" />
            <path d="M562 842c0 17-12 30-28 30s-28-13-28-30c0-16 12-29 28-29s28 13 28 29Z" />
            <path d="M430 842h76" />
          </g>

          <g className="hero-doodle hero-doodle-medium">
            <path d="M170 178c14-30 31-45 51-45 19 0 37 14 54 45" />
            <path d="M183 178c-8 22-8 42 0 60" />
            <path d="M216 178c-9 24-9 47 0 68" />
            <path d="M252 178c-8 21-8 41 0 60" />
            <path d="M148 240c33-11 67-11 102 0" />
          </g>

          <g className="hero-doodle hero-doodle-slow">
            <path d="M1240 864c9-10 21-15 36-15 15 0 27 5 36 15" />
            <path d="M1276 860c8-9 19-14 31-14 13 0 24 5 31 14" />
            <path d="M1206 892c20-10 44-13 70-8 23 4 43 4 61-2" />
            <path d="M1200 910c25-11 50-14 76-10 27 4 47 4 67-2" />
          </g>

          <g className="hero-doodle hero-doodle-fast">
            <path d="M760 250c10-9 22-14 35-14 13 0 25 5 35 14" />
            <path d="M806 264c9-9 20-14 32-14 13 0 24 5 34 14" />
            <path d="M776 298c14-6 28-8 42-4 13 4 24 4 35 0" />
          </g>
        </g>
      </svg>

      <style jsx>{`
        .hero-doodle {
          transform-origin: center;
        }

        .hero-doodle-slow {
          animation: doodleFloatSlow 12s ease-in-out infinite;
        }

        .hero-doodle-medium {
          animation: doodleFloatMedium 9s ease-in-out infinite;
        }

        .hero-doodle-fast {
          animation: doodleFloatFast 7s ease-in-out infinite;
        }

        @keyframes doodleFloatSlow {
          0%, 100% { transform: translate3d(0, 0, 0); }
          50% { transform: translate3d(0, -5px, 0); }
        }

        @keyframes doodleFloatMedium {
          0%, 100% { transform: translate3d(0, 0, 0); }
          50% { transform: translate3d(3px, -7px, 0); }
        }

        @keyframes doodleFloatFast {
          0%, 100% { transform: translate3d(0, 0, 0); }
          50% { transform: translate3d(-3px, -6px, 0); }
        }
      `}</style>
    </div>
  );
}
