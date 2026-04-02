"use client";

import React, { useRef, useEffect } from "react";

export function DotDistortionShaderBg() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let particles: { x: number; y: number; baseX: number; baseY: number; size: number }[] = [];
    const spacing = 24; // Distance between dots
    const radius = 1.5; // Base dot radius
    const mouseRadius = 250; // How far the distortion reaches

    let mouse = { x: -1000, y: -1000 };

    const init = () => {
      // Ensure we fetch the client parent size, fallback to window
      const parent = canvas.parentElement;
      canvas.width = parent ? parent.clientWidth : window.innerWidth;
      canvas.height = parent ? parent.clientHeight : window.innerHeight;
      
      particles = [];

      // Create a grid of particles
      const cols = Math.floor(canvas.width / spacing) + 1;
      const rows = Math.floor(canvas.height / spacing) + 1;

      for (let i = 0; i < cols; i++) {
        for (let j = 0; j < rows; j++) {
          const x = i * spacing;
          const y = j * spacing;
          particles.push({ x, y, baseX: x, baseY: y, size: radius });
        }
      }
    };

    let animationFrameId: number;

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];

        // Core physics / distortion logic
        const dx = mouse.x - p.baseX;
        const dy = mouse.y - p.baseY;
        const distance = Math.sqrt(dx * dx + dy * dy);

        let targetX = p.baseX;
        let targetY = p.baseY;
        let targetSize = radius;
        let opacity = 0.4;

        if (distance < mouseRadius) {
          // Calculate repulsion force
          const force = (mouseRadius - distance) / mouseRadius;
          // Calculate how far to push the dot (vector normalized)
          const pushX = (dx / (distance || 1)) * force * 50; 
          const pushY = (dy / (distance || 1)) * force * 50;

          targetX -= pushX;
          targetY -= pushY;
          
          // Make dots closer to the mouse slightly larger and more opaque
          targetSize = radius + (force * 1.5);
          opacity = 0.4 + (force * 0.6);
        }

        // Lerp strictly towards target for smooth animation
        p.x += (targetX - p.x) * 0.1;
        p.y += (targetY - p.y) * 0.1;
        p.size += (targetSize - p.size) * 0.1;

        // Draw dot
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(148, 163, 184, ${opacity})`; // slate-400 equivalent RGB with dynamic opacity
        ctx.fill();
      }

      animationFrameId = requestAnimationFrame(animate);
    };

    const handleMouseMove = (e: MouseEvent) => {
      // Get correct mouse coords relative to the canvas
      const rect = canvas.getBoundingClientRect();
      mouse.x = e.clientX - rect.left;
      mouse.y = e.clientY - rect.top;
    };

    const handleMouseLeave = () => {
      // Move mouse far away to let dots settle
      mouse.x = -1000;
      mouse.y = -1000;
    };

    const handleResize = () => {
      init();
    };

    // Initialization
    init();
    animate();

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseleave", handleMouseLeave);
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseleave", handleMouseLeave);
      window.removeEventListener("resize", handleResize);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 pointer-events-none z-0"
      style={{ display: "block", width: "100%", height: "100%" }}
    />
  );
}
