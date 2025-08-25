"use client";
import React, { useRef, useEffect, useState } from "react";
import { motion } from "framer-motion";

interface CoverProps {
  children: React.ReactNode;
  className?: string;
}

export function Cover({ children, className }: CoverProps) {
  const ref = useRef<HTMLSpanElement>(null);
  const [hovered, setHovered] = useState(false);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (ref.current) {
        const rect = ref.current.getBoundingClientRect();
        setMousePosition({
          x: e.clientX - rect.left,
          y: e.clientY - rect.top,
        });
      }
    };

    const element = ref.current;
    if (element) {
      element.addEventListener("mousemove", handleMouseMove);
      return () => element.removeEventListener("mousemove", handleMouseMove);
    }
  }, []);

  return (
    <span
      ref={ref}
      className={`relative inline-block cursor-pointer ${className || ""}`}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <motion.span
        className="relative z-20 bg-clip-text text-transparent bg-gradient-to-b from-blue-500 via-purple-500 to-pink-500"
        animate={{
          backgroundPosition: hovered ? "200% center" : "0% center",
        }}
        transition={{
          duration: 0.5,
          ease: "easeInOut",
        }}
        style={{
          backgroundSize: "200% 100%",
        }}
      >
        {children}
      </motion.span>
      
      {hovered && (
        <motion.span
          className="absolute inset-0 z-10 bg-gradient-to-r from-blue-500/20 via-purple-500/20 to-pink-500/20 rounded-lg blur-sm"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.8 }}
          transition={{ duration: 0.3 }}
          style={{
            background: `radial-gradient(circle at ${mousePosition.x}px ${mousePosition.y}px, rgba(59, 130, 246, 0.3), rgba(147, 51, 234, 0.2), rgba(236, 72, 153, 0.1))`,
          }}
        />
      )}
      
      <motion.span
        className="absolute inset-0 z-0 bg-gradient-to-r from-blue-500/10 via-purple-500/10 to-pink-500/10 rounded-lg"
        animate={{
          scale: hovered ? 1.05 : 1,
          opacity: hovered ? 0.8 : 0.3,
        }}
        transition={{ duration: 0.3 }}
      />
    </span>
  );
}
