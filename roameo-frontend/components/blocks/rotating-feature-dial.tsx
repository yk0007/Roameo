"use client"

import { motion } from "framer-motion"
import type { LucideIcon } from "lucide-react"
import { useEffect, useMemo, useState } from "react"

type DialFeature = {
  title: string
  description: string
  icon: LucideIcon
  wrapperClass: string
}

type RotatingFeatureDialProps = {
  features: readonly DialFeature[]
}

export function RotatingFeatureDial({ features }: RotatingFeatureDialProps) {
  const [activeIndex, setActiveIndex] = useState(0)
  const activeFeature = features[activeIndex]

  useEffect(() => {
    const timer = window.setInterval(() => {
      setActiveIndex((current) => (current + 1) % features.length)
    }, 4400)

    return () => window.clearInterval(timer)
  }, [features.length])

  const nodes = useMemo(() => {
    const radius = 188
    return features.map((feature, index) => {
      const angle = ((Math.PI * 2) / features.length) * index - Math.PI / 2
      const x = Math.cos(angle) * radius
      const y = Math.sin(angle) * radius
      return { feature, index, x, y }
    })
  }, [features])

  return (
    <div className="grid items-center gap-14 lg:grid-cols-[1.05fr_0.95fr]">
      <div className="relative mx-auto mt-6 flex h-[520px] w-full max-w-[520px] items-center justify-center">
        <div className="absolute inset-[72px] rounded-full bg-[radial-gradient(circle,rgba(95,120,220,0.14),rgba(95,120,220,0.03)_48%,rgba(255,255,255,0)_70%)] blur-2xl" />
        <motion.div
          animate={{ rotate: -(360 / features.length) * activeIndex }}
          transition={{ duration: 1.1, ease: [0.22, 1, 0.36, 1] }}
          className="absolute inset-0"
        >
          <div className="absolute inset-14 rounded-full border border-slate-200/90" />
          <div className="absolute inset-[5.5rem] rounded-full border border-slate-100" />

          {nodes.map(({ feature, index, x, y }) => {
            const Icon = feature.icon
            const isActive = index === activeIndex

            return (
              <button
                key={feature.title}
                type="button"
                onClick={() => setActiveIndex(index)}
                className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2"
                style={{ transform: `translate(calc(-50% + ${x}px), calc(-50% + ${y}px))` }}
                aria-label={feature.title}
              >
                <div className="relative">
                  {isActive ? (
                    <>
                      <motion.div
                        layoutId="feature-dial-active-ring"
                        className="absolute inset-[-10px] rounded-[28px] border-2 border-slate-900/12 shadow-[0_0_0_8px_rgba(255,255,255,0.86),0_0_0_14px_rgba(111,134,216,0.16)]"
                      />
                      <div className="absolute inset-[-20px] rounded-[34px] bg-[radial-gradient(circle,rgba(111,134,216,0.22),rgba(111,134,216,0)_72%)] blur-md" />
                    </>
                  ) : null}
                  <div
                    className={`relative flex h-20 w-20 items-center justify-center shadow-[0_24px_45px_rgba(15,23,42,0.14)] transition-all duration-500 ${
                      isActive ? "scale-110 shadow-[0_30px_60px_rgba(15,23,42,0.2)]" : "scale-100 opacity-85 hover:opacity-100"
                    } ${feature.wrapperClass}`}
                  >
                    <Icon className="h-8 w-8 text-white" strokeWidth={2.1} />
                  </div>
                </div>
              </button>
            )
          })}
        </motion.div>

        <div className="absolute inset-[110px] rounded-full bg-[radial-gradient(circle,rgba(112,144,238,0.08),rgba(112,144,238,0)_62%)] blur-2xl" />

        <div className="relative z-10 flex h-48 w-48 flex-col items-center justify-center rounded-full border border-slate-200 bg-white px-4 shadow-[0_24px_60px_rgba(15,23,42,0.08)]">
          <p className="text-[11px] uppercase tracking-[0.22em] text-slate-400">Features</p>
          <p className="mt-2.5 text-center text-3xl font-semibold tracking-[-0.05em] text-slate-950">
            {String(activeIndex + 1).padStart(2, "0")}
          </p>
          <p className="mt-1.5 max-w-[8.5rem] text-center text-[0.74rem] leading-[1.02rem] text-slate-500">
            Select a node to inspect
          </p>
        </div>
      </div>

      <div className="max-w-xl">
        <div className="flex items-center gap-3">
          <div className={`flex h-12 w-12 items-center justify-center shadow-[0_18px_36px_rgba(15,23,42,0.14)] ${activeFeature.wrapperClass}`}>
            <activeFeature.icon className="h-5 w-5 text-white" strokeWidth={2.1} />
          </div>
          <div>
            <p className="text-[11px] uppercase tracking-[0.26em] text-[#6f86d8]">Features</p>
          </div>
        </div>
        <motion.h3
          key={activeFeature.title}
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, ease: "easeOut" }}
          className="mt-5 text-5xl font-semibold tracking-[-0.06em] text-slate-950"
        >
          {activeFeature.title}
        </motion.h3>
        <motion.p
          key={`${activeFeature.title}-description`}
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, delay: 0.05, ease: "easeOut" }}
          className="mt-6 max-w-lg text-lg leading-9 text-slate-600"
        >
          {activeFeature.description}
        </motion.p>

        <motion.div
          key={`${activeFeature.title}-note`}
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, delay: 0.08, ease: "easeOut" }}
          className="mt-7 rounded-[24px] border border-slate-200/80 bg-[linear-gradient(180deg,#ffffff_0%,#f8fbff_100%)] px-5 py-4 shadow-[0_18px_45px_rgba(15,23,42,0.04)]"
        >
          <p className="text-[11px] uppercase tracking-[0.22em] text-slate-400">Why it matters</p>
          <p className="mt-2 text-sm leading-6 text-slate-600">
            Each option updates the conversation with travel context you can actually use while refining the trip.
          </p>
        </motion.div>

        <div className="mt-8 h-px w-full bg-[linear-gradient(90deg,rgba(148,163,184,0.12),rgba(148,163,184,0.34),rgba(148,163,184,0.12))]" />

        <div className="mt-8 flex flex-wrap gap-3">
          {features.map((feature, index) => (
            <button
              key={feature.title}
              type="button"
              onClick={() => setActiveIndex(index)}
              className={`rounded-full border px-4 py-2 text-sm transition-colors ${
                index === activeIndex
                  ? "border-slate-950 bg-slate-950 text-white"
                  : "border-slate-200 bg-white text-slate-600 hover:border-slate-300 hover:text-slate-950"
              }`}
            >
              {feature.title}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
