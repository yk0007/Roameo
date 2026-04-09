"use client";

import type { ReactNode } from "react";
import { AnimatePresence, MotionConfig, motion, useReducedMotion } from "framer-motion";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const SMOOTH_EASE = [0.22, 1, 0.36, 1] as const;

interface MotionShellProps {
  children: ReactNode;
  className?: string;
  delay?: number;
  disabled?: boolean;
}

export function SiteMotionProvider({ children }: { children: ReactNode }) {
  return (
    <MotionConfig
      reducedMotion="user"
      transition={{
        duration: 0.56,
        ease: SMOOTH_EASE,
      }}
    >
      {children}
    </MotionConfig>
  );
}

export function RouteTransition({ children, className }: Omit<MotionShellProps, "delay" | "disabled">) {
  const pathname = usePathname();
  const shouldReduceMotion = useReducedMotion();

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={pathname}
        className={className}
        initial={
          shouldReduceMotion
            ? undefined
            : {
                opacity: 0,
                y: 22,
                scale: 0.992,
              }
        }
        animate={{
          opacity: 1,
          y: 0,
          scale: 1,
        }}
        exit={
          shouldReduceMotion
            ? undefined
            : {
                opacity: 0,
                y: -14,
                scale: 0.996,
              }
        }
        transition={
          shouldReduceMotion
            ? { duration: 0 }
            : {
                duration: 0.52,
                ease: SMOOTH_EASE,
              }
        }
      >
        {children}
      </motion.div>
    </AnimatePresence>
  );
}

export function EntranceMotion({ children, className, delay = 0, disabled = false }: MotionShellProps) {
  const shouldReduceMotion = useReducedMotion();

  if (shouldReduceMotion || disabled) {
    return <div className={className}>{children}</div>;
  }

  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, y: 20, scale: 0.99 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{
        duration: 0.58,
        ease: SMOOTH_EASE,
        delay,
      }}
    >
      {children}
    </motion.div>
  );
}

export function SectionReveal({ children, className, delay = 0, disabled = false }: MotionShellProps) {
  const shouldReduceMotion = useReducedMotion();

  if (shouldReduceMotion || disabled) {
    return <div className={className}>{children}</div>;
  }

  return (
    <motion.div
      className={cn("will-change-transform", className)}
      initial={{ opacity: 0, y: 34, scale: 0.985 }}
      whileInView={{ opacity: 1, y: 0, scale: 1 }}
      viewport={{ once: true, amount: 0.18, margin: "0px 0px -8% 0px" }}
      transition={{
        duration: 0.72,
        ease: SMOOTH_EASE,
        delay,
      }}
    >
      {children}
    </motion.div>
  );
}
