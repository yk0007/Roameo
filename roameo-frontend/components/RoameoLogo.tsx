interface RoameoLogoProps {
  className?: string
}

export default function RoameoLogo({ className = "w-8 h-8" }: RoameoLogoProps) {
  return (
    <div className={`bg-foreground rounded-full flex items-center justify-center ${className}`}>
      <div className="w-2 h-2 bg-background rounded-full"></div>
    </div>
  )
}
