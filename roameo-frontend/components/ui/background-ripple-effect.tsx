"use client";
import React, { useRef, useEffect, useState } from "react";
import { motion } from "framer-motion";

interface RippleProps {
  x: number;
  y: number;
  id: number;
}

export function BackgroundRippleEffect() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [ripples, setRipples] = useState<RippleProps[]>([]);
  const [isHovered, setIsHovered] = useState(false);

  const createRipple = (e: React.MouseEvent) => {
    if (!containerRef.current) return;
    
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const newRipple = {
      x,
      y,
      id: Date.now() + Math.random(),
    };
    
    setRipples(prev => [...prev, newRipple]);
    
    // Remove ripple after animation
    setTimeout(() => {
      setRipples(prev => prev.filter(ripple => ripple.id !== newRipple.id));
    }, 1000);
  };

  // Generate grid of boxes
  const boxes = [];
  const gridSize = 20;
  
  for (let i = 0; i < gridSize; i++) {
    for (let j = 0; j < gridSize; j++) {
      boxes.push({ id: `${i}-${j}`, x: i, y: j });
    }
  }

  return (
    <div
      ref={containerRef}
      className="absolute inset-0 overflow-hidden cursor-pointer"
      onClick={createRipple}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Grid of boxes */}
      <div 
        className="absolute inset-0 p-4 opacity-20"
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(20, 1fr)',
          gridTemplateRows: 'repeat(20, 1fr)',
          gap: '4px'
        }}
      >
        {boxes.map((box) => (
          <motion.div
            key={box.id}
            className="w-full h-full bg-gray-400 rounded-sm"
            whileHover={{
              scale: 1.3,
              backgroundColor: "#6b7280",
              transition: { duration: 0.2 }
            }}
            animate={{
              scale: isHovered ? 1.02 : 1,
              transition: { duration: 0.3 }
            }}
          />
        ))}
      </div>

      {/* Ripple effects */}
      {ripples.map((ripple) => (
        <motion.div
          key={ripple.id}
          className="absolute rounded-full border-2 border-gray-500 pointer-events-none"
          style={{
            left: ripple.x,
            top: ripple.y,
            transform: "translate(-50%, -50%)",
          }}
          initial={{
            width: 0,
            height: 0,
            opacity: 1,
          }}
          animate={{
            width: 300,
            height: 300,
            opacity: 0,
          }}
          transition={{
            duration: 1,
            ease: "easeOut",
          }}
        />
      ))}
    </div>
  );
}
