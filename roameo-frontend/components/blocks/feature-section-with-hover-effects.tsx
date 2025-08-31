import { cn } from "@/lib/utils";
import {
  IconMapPin,
  IconRobot,
  IconCalendarEvent,
  IconRoute,
  IconStar,
  IconUsers,
  IconShield,
  IconClock,
} from "@tabler/icons-react";

export function FeaturesSectionWithHoverEffects() {
  const features = [
    {
      title: "Smart Destination Discovery",
      description:
        "AI-powered recommendations tailored to your preferences, budget, and travel style.",
      icon: <IconMapPin />,
    },
    {
      title: "Intelligent Planning Agent",
      description:
        "Our AI agent creates detailed itineraries with optimized routes and schedules.",
      icon: <IconRobot />,
    },
    {
      title: "Real-time Itinerary Builder",
      description:
        "Dynamic day-by-day plans that adapt to your interests and constraints.",
      icon: <IconCalendarEvent />,
    },
    {
      title: "Optimized Route Planning",
      description: "Efficient travel routes that maximize your time and minimize costs.",
      icon: <IconRoute />,
    },
    {
      title: "Curated POI Recommendations",
      description: "Handpicked points of interest based on reviews, ratings, and relevance.",
      icon: <IconStar />,
    },
    {
      title: "Collaborative Trip Planning",
      description:
        "Share and collaborate on trips with friends and family in real-time.",
      icon: <IconUsers />,
    },
    {
      title: "Secure & Private",
      description:
        "Your travel data is protected with enterprise-grade security measures.",
      icon: <IconShield />,
    },
    {
      title: "24/7 Travel Assistant",
      description: "Get instant help and modifications to your travel plans anytime.",
      icon: <IconClock />,
    },
  ];
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 relative z-10 py-10 max-w-7xl mx-auto">
      {features.map((feature, index) => (
        <Feature key={feature.title} {...feature} index={index} />
      ))}
    </div>
  );
}

const Feature = ({
  title,
  description,
  icon,
  index,
}: {
  title: string;
  description: string;
  icon: React.ReactNode;
  index: number;
}) => {
  return (
    <div
      className={cn(
        "flex flex-col lg:border-r py-10 relative group/feature border-gray-200",
        (index === 0 || index === 4) && "lg:border-l border-gray-200",
        index < 4 && "lg:border-b border-gray-200"
      )}
    >
      {index < 4 && (
        <div className="opacity-0 group-hover/feature:opacity-100 transition duration-200 absolute inset-0 h-full w-full bg-gradient-to-t from-blue-50 to-transparent pointer-events-none" />
      )}
      {index >= 4 && (
        <div className="opacity-0 group-hover/feature:opacity-100 transition duration-200 absolute inset-0 h-full w-full bg-gradient-to-b from-blue-50 to-transparent pointer-events-none" />
      )}
      <div className="mb-4 relative z-10 px-10 text-blue-600">
        {icon}
      </div>
      <div className="text-lg font-bold mb-2 relative z-10 px-10">
        <div className="absolute left-0 inset-y-0 h-6 group-hover/feature:h-8 w-1 rounded-tr-full rounded-br-full bg-gray-300 group-hover/feature:bg-blue-500 transition-all duration-200 origin-center" />
        <span className="group-hover/feature:translate-x-2 transition duration-200 inline-block text-gray-800">
          {title}
        </span>
      </div>
      <p className="text-sm text-gray-600 max-w-xs relative z-10 px-10">
        {description}
      </p>
    </div>
  );
};