'use client';

import { useScroll, useTransform, motion, MotionValue } from 'motion/react';
import React, { useRef, forwardRef } from 'react';
import { Button } from "@/components/ui/button"
import { BlurFade } from "@/components/ui/blur-fade"
import Link from "next/link"
import { supabase } from "@/lib/supabase/client"
import type { User } from "@supabase/supabase-js"
import { useRouter } from "next/navigation"

interface SectionProps {
  scrollYProgress: MotionValue<number>;
  user?: User | null;
  handleProtectedAction?: (action: string) => void;
}

const Section1: React.FC<SectionProps> = ({ scrollYProgress, user, handleProtectedAction }) => {
  const scale = useTransform(scrollYProgress, [0, 1], [1, 0.8]);
  const rotate = useTransform(scrollYProgress, [0, 1], [0, -5]);
  
  return (
    <motion.section
      style={{ scale, rotate }}
      className='sticky font-semibold top-0 h-screen bg-gradient-to-br from-sky-300 via-blue-500 to-blue-800 flex flex-col items-center justify-center text-white relative overflow-hidden px-6'
    >
      {/* Grid background */}
      <div className='absolute bottom-0 left-0 right-0 top-0 bg-[linear-gradient(to_right,#4f4f4f2e_1px,transparent_1px),linear-gradient(to_bottom,#4f4f4f2e_1px,transparent_1px)] bg-[size:54px_54px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)]'></div>

      <div className="max-w-7xl mx-auto flex items-center justify-between relative z-10">
        <div className="flex-1 max-w-2xl">
          <BlurFade delay={0.25} inView>
            <h1 className="text-6xl font-bold text-white mb-6 leading-tight">
              Plan your perfect
              <br />
              <span className="italic">adventure.</span>
            </h1>
          </BlurFade>
          <BlurFade delay={0.25 * 2} inView>
            <p className="text-xl text-white/90 mb-8 leading-relaxed">
              Experience AI-powered travel planning that understands your preferences and creates personalized
              itineraries in minutes, not hours.
            </p>
          </BlurFade>
          <BlurFade delay={0.25 * 3} inView>
            <div className="flex items-center gap-4">
              <Button
                onClick={() => handleProtectedAction?.("start planning")}
                size="lg"
                className="bg-black text-white hover:bg-gray-800 rounded-full px-8 py-3 text-lg"
              >
                Start planning
              </Button>
            </div>
          </BlurFade>
        </div>

        <div className="flex-1 relative">
          <motion.div
            initial={{ y: 50, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.5, ease: "easeOut" }}
            className="absolute top-0 right-0 bg-white rounded-2xl p-6 shadow-2xl max-w-sm transform rotate-3"
          >
            <div className="space-y-4">
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 bg-black rounded-full flex items-center justify-center">
                  <div className="w-2 h-2 bg-white rounded-full"></div>
                </div>
                <div className="flex-1">
                  <p className="text-sm text-gray-700">
                    I'd love to help you plan your trip to Araku Valley! Are you interested in the coffee plantations,
                    tribal culture, or the scenic waterfalls?
                  </p>
                </div>
              </div>
              <div className="flex items-start gap-3 justify-end">
                <div className="bg-blue-100 rounded-lg p-3 max-w-xs">
                  <p className="text-sm text-gray-700">
                    I want to experience the coffee plantation tours, visit the tribal museum, and see the Borra
                    Caves!
                  </p>
                </div>
                <div className="w-8 h-8 bg-gray-300 rounded-full"></div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 bg-black rounded-full flex items-center justify-center">
                  <div className="w-2 h-2 bg-white rounded-full"></div>
                </div>
                <div className="flex-1">
                  <p className="text-sm text-gray-700">
                    Perfect! I'm creating a 3-day Araku itinerary with visits to the Ananthagiri Coffee Plantations,
                    Tribal Museum, Borra Caves, and a scenic train journey through the Eastern Ghats...
                  </p>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </motion.section>
  );
};

const Section2: React.FC<SectionProps> = ({ scrollYProgress }) => {
  const scale = useTransform(scrollYProgress, [0, 1], [0.8, 1]);
  const rotate = useTransform(scrollYProgress, [0, 1], [5, 0]);

  return (
    <motion.section
      style={{ scale, rotate }}
      className='relative h-screen bg-gradient-to-t to-[#1a1919] from-[#06060e] text-white'
    >
      <div className='absolute bottom-0 left-0 right-0 top-0 bg-[linear-gradient(to_right,#4f4f4f2e_1px,transparent_1px),linear-gradient(to_bottom,#4f4f4f2e_1px,transparent_1px)] bg-[size:54px_54px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)]'></div>
      <article className='container mx-auto relative z-10 px-6 py-16'>
        <h1 className='text-6xl leading-[100%] py-10 font-semibold tracking-tight text-center mb-8'>
          Explore Popular Destinations
        </h1>
        
        <div className="max-w-5xl mx-auto">
          <div className="grid grid-cols-6 gap-2 h-[476px] relative overflow-hidden">
            {/* Spain - Large card spanning 2 columns and full height */}
            <div className="col-span-2 row-span-2 relative group cursor-pointer overflow-hidden">
              <div className="w-full h-full max-h-[476px] overflow-hidden rounded-2xl">
                <img
                  src="https://images.unsplash.com/photo-1539037116277-4db20889f2d4?q=80&w=2070&auto=format&fit=crop"
                  alt="Sagrada Familia, Barcelona, Spain"
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                />
                <div className="absolute bottom-4 left-4">
                  <span className="bg-white/90 backdrop-blur-sm px-3 py-1 rounded-full text-sm font-medium text-gray-900">
                    Spain
                  </span>
                </div>
              </div>
            </div>

            {/* London - Medium card spanning 2 columns, top half */}
            <div className="col-span-2 relative group cursor-pointer overflow-hidden">
              <div className="w-full h-full max-h-[476px] overflow-hidden rounded-2xl">
                <img
                  src="https://images.unsplash.com/photo-1513635269975-59663e0ac1ad?q=80&w=2070&auto=format&fit=crop"
                  alt="London with red double-decker bus"
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                />
                <div className="absolute bottom-4 left-4">
                  <span className="bg-white/90 backdrop-blur-sm px-3 py-1 rounded-full text-sm font-medium text-gray-900">
                    London
                  </span>
                </div>
              </div>
            </div>

            {/* Croatia - Medium card spanning 2 columns, top half */}
            <div className="col-span-2 relative group cursor-pointer overflow-hidden">
              <div className="w-full h-full max-h-[476px] overflow-hidden rounded-2xl">
                <img
                  src="https://images.unsplash.com/photo-1603551664565-1b3c3dd61075?q=80&w=2070&auto=format&fit=crop"
                  alt="Dubrovnik, Croatia coastal view"
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                />
                <div className="absolute bottom-4 left-4">
                  <span className="bg-white/90 backdrop-blur-sm px-3 py-1 rounded-full text-sm font-medium text-gray-900">
                    Croatia
                  </span>
                </div>
              </div>
            </div>

            {/* Bratislava - Small card, bottom left */}
            <div className="relative group cursor-pointer overflow-hidden">
              <div className="w-full h-full max-h-[476px] overflow-hidden rounded-2xl">
                <img
                  src="https://images.unsplash.com/photo-1567072584703-e445170f9478?q=80&w=987&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
                  alt="Bratislava Castle and Danube River"
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                />
                <div className="absolute bottom-4 left-4">
                  <span className="bg-white/90 backdrop-blur-sm px-3 py-1 rounded-full text-sm font-medium text-gray-900">
                    Bratislava
                  </span>
                </div>
              </div>
            </div>

            {/* Copenhagen - Small card, bottom center */}
            <div className="relative group cursor-pointer overflow-hidden">
              <div className="w-full h-full max-h-[476px] overflow-hidden rounded-2xl">
                <img
                  src="https://images.unsplash.com/photo-1513622470522-26c3c8a854bc?q=80&w=2070&auto=format&fit=crop"
                  alt="Copenhagen colorful buildings and architecture"
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                />
                <div className="absolute bottom-4 left-4">
                  <span className="bg-white/90 backdrop-blur-sm px-3 py-1 rounded-full text-sm font-medium text-gray-900">
                    Copenhagen
                  </span>
                </div>
              </div>
            </div>

            {/* Lisbon - Medium card spanning 2 columns, bottom half */}
            <div className="col-span-2 relative group cursor-pointer overflow-hidden">
              <div className="w-full h-full max-h-[476px] overflow-hidden rounded-2xl">
                <img
                  src="https://images.unsplash.com/photo-1555881400-74d7acaacd8b?q=80&w=2070&auto=format&fit=crop"
                  alt="Lisbon historic tram and architecture"
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                />
                <div className="absolute bottom-4 left-4">
                  <span className="bg-white/90 backdrop-blur-sm px-3 py-1 rounded-full text-sm font-medium text-gray-900">
                    Lisbon
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </article>
    </motion.section>
  );
};

const Component = forwardRef<HTMLElement, { user?: User | null; handleProtectedAction?: (action: string) => void }>((props, ref) => {
  const container = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: container,
    offset: ['start start', 'end end'],
  });

  return (
    <>
      <main ref={container} className='relative h-[200vh] bg-black'>
        <Section1 scrollYProgress={scrollYProgress} user={props.user} handleProtectedAction={props.handleProtectedAction} />
        <Section2 scrollYProgress={scrollYProgress} />
      </main>
    </>
  );
});

Component.displayName = 'HeroScrollAnimation';

export default Component;