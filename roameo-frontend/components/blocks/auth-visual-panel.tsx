"use client"

export function AuthVisualPanel() {
  return (
    <div className="relative hidden h-full min-h-screen overflow-hidden bg-slate-950 lg:block">
      <img
        src="https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&fm=jpg&q=80&w=1800"
        alt="Green mountain across body of water"
        className="absolute inset-0 h-full w-full object-cover"
      />
      <div className="absolute inset-0 bg-[linear-gradient(180deg,rgba(8,15,36,0.16)_0%,rgba(8,15,36,0.28)_20%,rgba(8,15,36,0.72)_100%)]" />
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(132,204,255,0.22),transparent_34%),radial-gradient(circle_at_bottom_right,rgba(56,189,248,0.12),transparent_28%)]" />

      <div className="relative z-10 flex h-full items-end p-8 xl:p-10">
        <div className="max-w-[34rem] pb-2">
          <h2 className="max-w-xl text-5xl font-semibold leading-[0.95] tracking-[-0.07em] text-white xl:text-[4.5rem]">
            Enter a calmer way to plan every route, stay, and day.
          </h2>
          <p className="mt-6 max-w-lg text-base leading-7 text-white">
            Roameo keeps the conversation, itinerary, and decisions tied together so the trip feels designed,
            not assembled.
          </p>
        </div>
      </div>
    </div>
  )
}
