"use client"

const moods = [
  {
    image: "/mountain-lake-reflection.png",
    className: "lg:row-span-2",
  },
  {
    image: "/varanasi-ghats-sunrise.png",
    className: "",
  },
  {
    image: "/udaipur-lake-pichola-sunset.png",
    className: "",
  },
] as const

export function LandingTripMoods() {
  return (
    <section className="bg-white px-6 py-24">
      <div className="mx-auto grid max-w-7xl items-end gap-14 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="max-w-xl">
          <p className="inline-flex rounded-full border border-slate-200 bg-slate-50 px-4 py-2 text-[11px] font-medium uppercase tracking-[0.2em] text-slate-500">
            Trip moods
          </p>
          <h2 className="mt-7 max-w-lg text-5xl font-semibold tracking-[-0.06em] text-slate-950">
            Choose the feeling of the trip before you choose every detail.
          </h2>
          <p className="mt-6 max-w-lg text-lg leading-8 text-slate-600">
            Roameo helps you start with pace, atmosphere, and rhythm, then turns that mood into stays,
            route decisions, and day structure that already feel considered.
          </p>

          <p className="mt-10 max-w-sm text-sm leading-6 text-slate-500">
            Start broad, then let the planner narrow the trip into something that feels coherent.
          </p>
        </div>

        <div className="grid auto-rows-[230px] gap-5 lg:grid-cols-[1.05fr_0.95fr]">
          {moods.map((mood) => (
            <article
              key={mood.image}
              className={`group relative overflow-hidden rounded-[32px] border border-slate-100 bg-slate-950 shadow-[0_34px_72px_rgba(15,23,42,0.12)] ${mood.className}`}
            >
              <img
                src={mood.image}
                alt=""
                className="h-full w-full object-cover transition-transform duration-700 group-hover:scale-[1.06]"
              />
              <div className="absolute inset-0 bg-[linear-gradient(180deg,rgba(15,23,42,0.02)_0%,rgba(15,23,42,0.04)_100%)]" />
            </article>
          ))}
        </div>
      </div>
    </section>
  )
}
