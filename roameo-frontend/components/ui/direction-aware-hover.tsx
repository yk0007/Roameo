"use client"

import type React from "react"
import { useRef, useState } from "react"
import Image from "next/image"
import { motion } from "framer-motion"
import { cn } from "@/lib/utils"

export const DirectionAwareHover = ({
  imageUrl,
  children,
  childrenClassName,
  imageClassName,
  className,
}: {
  imageUrl: string
  children: React.ReactNode | string
  childrenClassName?: string
  imageClassName?: string
  className?: string
}) => {
  const ref = useRef<HTMLDivElement>(null)
  const [isHovered, setIsHovered] = useState(false)
  const [imageError, setImageError] = useState(false)

  return (
    <motion.div
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      ref={ref}
      className={cn(
        "h-48 w-full bg-white dark:bg-gray-900 rounded-lg overflow-hidden group/card relative shadow-lg",
        className,
      )}
      transition={{ duration: 0.2, ease: "easeOut" }}
    >
      <div className="flex flex-col h-full w-full">
        <div className="h-full relative bg-gray-100 dark:bg-gray-800 overflow-hidden">
          <Image
            alt="destination image"
            className={cn("h-full w-full object-cover object-center", imageClassName)}
            width="1000"
            height="1000"
            src={
              imageError ? `/placeholder.svg?height=400&width=600&query=scenic travel destination landscape` : imageUrl
            }
            onError={() => setImageError(true)}
            priority={true}
            quality={90}
            placeholder="empty"
          />
          <div
            className={cn(
              "absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/90 via-black/60 to-transparent p-3 transform transition-all duration-300",
              isHovered ? "from-black/95 via-black/70" : "from-black/80 via-black/50",
              childrenClassName,
            )}
          >
            <div className="text-white text-left text-sm font-medium leading-tight drop-shadow-lg">{children}</div>
          </div>
        </div>
      </div>
    </motion.div>
  )
}
