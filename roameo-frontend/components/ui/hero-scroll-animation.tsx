'use client';

import { motion } from 'framer-motion';
import { ArrowRight, MapPin } from 'lucide-react';
import type { User } from '@supabase/supabase-js';
import { Button } from '@/components/ui/button';

type HeroProps = {
  user?: User | null;
  handleProtectedAction?: (action: string) => void;
};

export default function HeroScrollAnimation({ handleProtectedAction }: HeroProps) {
  return (
    <section className="bg-white pb-6 pt-0">
      <div className="relative min-h-[112svh] overflow-hidden rounded-b-[54px] bg-[#061225] text-white sm:rounded-b-[64px]">
        <video
          className="absolute inset-0 h-full w-full object-cover"
          autoPlay
          muted
          loop
          playsInline
          preload="auto"
          poster="/travelliovideo-poster.jpg"
        >
          <source src="/travelliovideo.mp4" type="video/mp4" />
        </video>

        <div className="absolute inset-0 bg-[linear-gradient(180deg,rgba(4,12,28,0.42)_0%,rgba(5,19,42,0.26)_24%,rgba(7,19,38,0.48)_60%,rgba(4,10,22,0.82)_100%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_26%_44%,rgba(9,17,32,0.56),rgba(9,17,32,0.08)_28%,rgba(9,17,32,0)_52%),radial-gradient(circle_at_72%_28%,rgba(255,255,255,0.1),rgba(255,255,255,0)_22%)]" />
        <div className="absolute inset-x-0 bottom-0 h-40 bg-[linear-gradient(180deg,transparent,rgba(4,10,22,0.7))]" />

        <div className="relative mx-auto flex min-h-[100svh] max-w-7xl items-center px-8 pb-20 pt-32 sm:px-10 lg:px-12">
          <div className="max-w-[42rem]">
          <motion.h1
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.65, delay: 0.08, ease: 'easeOut' }}
            className="max-w-[42rem] font-serif text-[4.4rem] font-semibold leading-[0.92] tracking-[-0.05em] text-white sm:text-[5rem] xl:text-[5.35rem]"
          >
            Travel planning that finally feels composed.
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.16, ease: 'easeOut' }}
            className="mt-5 max-w-lg text-[1.02rem] leading-8 text-white/74 sm:text-[1.08rem]"
          >
            Start in chat, refine on the map, and keep every place, day, and decision tied together in one calmer travel workspace.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.24, ease: 'easeOut' }}
            className="mt-9 flex flex-wrap items-center gap-4"
          >
            <Button
              onClick={() => handleProtectedAction?.('start planning')}
              size="lg"
              className="rounded-full bg-black px-7 text-base text-white shadow-[0_16px_32px_rgba(0,0,0,0.22)] hover:bg-gray-800"
            >
              Start planning
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
            <a
              href="#how-it-works"
              className="inline-flex items-center rounded-full border border-white/18 bg-white/10 px-6 py-3 text-sm font-medium text-white/88 backdrop-blur-md transition-colors hover:bg-white/14"
            >
              See how it works
            </a>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.32, ease: 'easeOut' }}
            className="mt-10 flex flex-wrap items-center gap-x-8 gap-y-3 text-sm text-white/70"
          >
            <div className="flex items-center gap-2">
              <div className="h-1.5 w-1.5 rounded-full bg-white/80" />
              Natural conversation
            </div>
            <div className="flex items-center gap-2">
              <div className="h-1.5 w-1.5 rounded-full bg-white/80" />
              Day-by-day itinerary stays synced
            </div>
            <div className="flex items-center gap-2">
              <MapPin className="h-4 w-4" />
              Real destinations, not template fluff
            </div>
          </motion.div>
        </div>
      </div>
      </div>
    </section>
  );
}
