"use client"

import { motion } from "framer-motion"
import { cn } from "@/lib/utils"

interface TextShimmerProps {
  children: React.ReactNode
  className?: string
  duration?: number
}

export function TextShimmer({ children, className, duration = 1.5 }: TextShimmerProps) {
  return (
    <motion.div
      className={cn(
        "relative inline-block bg-gradient-to-r",
        "from-[var(--base-color,theme(colors.gray.600))] via-[var(--base-gradient-color,theme(colors.gray.400))] to-[var(--base-color,theme(colors.gray.600))]",
        "bg-[length:200%_100%] bg-clip-text text-transparent",
        className
      )}
      animate={{
        backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"],
      }}
      transition={{
        duration: duration,
        repeat: Infinity,
        ease: "easeInOut",
      }}
    >
      {children}
    </motion.div>
  )
}