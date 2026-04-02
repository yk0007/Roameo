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
    const spacing = 16; // Tighter grid = more dots
    const radius = 1.2; // Base dot radius
    const mouseRadius = 200; // How far the distortion reaches
    const pushStrength = 40; // How far dots get pushed

    let mouse = { x: -1000, y: -1000 };
    let dpr = 1;

    const init = () => {
      dpr = window.devicePixelRatio || 1;
      const parent = canvas.parentElement;
      const w = parent ? parent.clientWidth : window.innerWidth;
      const h = parent ? parent.clientHeight : window.innerHeight;

      // Set the canvas resolution to match DPR for sharp rendering
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      canvas.style.width = w + "px";
      canvas.style.height = h + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      particles = [];

      // Create a grid of particles across the full area
      const cols = Math.ceil(w / spacing) + 1;
      const rows = Math.ceil(h / spacing) + 1;

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
      const w = canvas.width / dpr;
      const h = canvas.height / dpr;
      ctx.clearRect(0, 0, w, h);

      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];

        const dx = mouse.x - p.baseX;
        const dy = mouse.y - p.baseY;
        const distSq = dx * dx + dy * dy;
        const mouseRadiusSq = mouseRadius * mouseRadius;

        let targetX = p.baseX;
        let targetY = p.baseY;
        let targetSize = radius;
        let opacity = 0.25;

        if (distSq < mouseRadiusSq) {
          const distance = Math.sqrt(distSq);
          const force = (mouseRadius - distance) / mouseRadius;
          const normX = dx / (distance || 1);
          const normY = dy / (distance || 1);

          // Push dots away from mouse
          targetX -= normX * force * pushStrength;
          targetY -= normY * force * pushStrength;

          // Scale up and brighten dots near the cursor
          targetSize = radius + force * 2;
          opacity = 0.25 + force * 0.75;
        }

        // Smooth interpolation towards target
        p.x += (targetX - p.x) * 0.15;
        p.y += (targetY - p.y) * 0.15;
        p.size += (targetSize - p.size) * 0.15;

        // Draw
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(120, 140, 165, ${opacity})`;
        ctx.fill();
      }

      animationFrameId = requestAnimationFrame(animate);
    };

    const handleMouseMove = (e: MouseEvent) => {
      // Use pageX/pageY and subtract the canvas offset to get correct
      // coordinates even when the page is scrolled.
      const rect = canvas.getBoundingClientRect();
      mouse.x = e.clientX - rect.left;
      mouse.y = e.clientY - rect.top;
    };

    const handleMouseLeave = () => {
      mouse.x = -1000;
      mouse.y = -1000;
    };

    const handleResize = () => {
      init();
    };

    init();
    animate();

    // Listen on the document so we capture moves even over elements above the canvas
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseleave", handleMouseLeave);
    window.addEventListener("resize", handleResize);

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseleave", handleMouseLeave);
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
