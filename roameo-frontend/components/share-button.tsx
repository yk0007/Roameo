"use client"

import { useState } from 'react'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Button } from "@/components/ui/button"
import { Share2, Copy, Mail } from 'lucide-react'
import type { Itinerary } from '../lib/types'

interface ShareButtonProps {
  tripId: string;
  tripTitle?: string;
  itinerary?: Itinerary;
}

const generateShareText = (tripTitle?: string, itinerary?: Itinerary) => {
  if (!itinerary) return "Check out my trip itinerary!";

  let text = `Check out my trip to ${tripTitle || itinerary.destination || 'this amazing place'}!\n\n`;
  itinerary.daysPlan.forEach(day => {
    text += `Day ${day.day}: ${day.title || ''}\n`;
    day.activities.forEach(activity => {
      text += `  - ${activity.name}\n`;
    });
    text += '\n';
  });

  return encodeURIComponent(text);
};

const generateMarkdownText = (shareUrl: string, tripTitle?: string, itinerary?: Itinerary) => {
  if (!itinerary) return `*Check out my trip itinerary!*\n\n${shareUrl}`;

  let text = `*🌟 ${tripTitle || itinerary.destination || 'My Amazing Trip'} 🌟*\n\n`;
  
  itinerary.daysPlan.forEach(day => {
    text += `*Day ${day.day}:* ${day.title || `Day ${day.day}`}\n`;
    day.activities.forEach(activity => {
      text += `  • ${activity.name}\n`;
    });
    text += '\n';
  });
  
  text += `🔗 *View full itinerary:* ${shareUrl}`;
  return text;
};

export function ShareButton({ tripId, tripTitle, itinerary }: ShareButtonProps) {
  const [isCopied, setIsCopied] = useState(false);
  const shareUrl = `${window.location.origin}/chat?sessionId=${encodeURIComponent(tripId)}`;
  const shareText = generateShareText(tripTitle, itinerary);

  const handleCopyMarkdown = () => {
    const markdownText = generateMarkdownText(shareUrl, tripTitle, itinerary);
    navigator.clipboard.writeText(markdownText).then(() => {
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 2000);
    });
  };

  const handleShare = (platform: 'whatsapp' | 'gmail') => {
    let url = '';
    if (platform === 'whatsapp') {
      url = `https://wa.me/?text=${shareText}%20${encodeURIComponent(shareUrl)}`;
    } else if (platform === 'gmail') {
      const subject = encodeURIComponent(tripTitle || "Check out my trip!");
      url = `mailto:?subject=${subject}&body=${shareText}%20${encodeURIComponent(shareUrl)}`;
    }
    window.open(url, '_blank');
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="sm" className="border-0 bg-white shadow-md hover:shadow-lg transition-shadow rounded-xl">
          <Share2 className="w-4 h-4 mr-2" />
          Share
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="border-0 shadow-xl rounded-xl bg-white/95 backdrop-blur-md">
        <DropdownMenuItem onClick={() => handleShare('whatsapp')}>
          <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4 mr-2" viewBox="0 0 24 24" fill="#25D366" stroke="currentColor" strokeWidth="0" strokeLinecap="round" strokeLinejoin="round"><path d="M12.24 6.81c-1.32 0-2.62.33-3.79.96-1.18.64-2.21 1.53-3.04 2.62-1.68 2.22-2.33 4.9-1.79 7.5.33 1.56 1.17 2.98 2.41 4.08s2.79 1.84 4.49 2.19c.52.11 1.04.16 1.57.16 2.59 0 5.02-1 6.84-2.82 1.82-1.82 2.82-4.25 2.82-6.84 0-2.59-1-5.02-2.82-6.84s-4.25-2.82-6.84-2.82zm-8.34 12.59c-1.2-1.06-2-2.4-2.3-3.86-.45-2.2.12-4.49 1.6-6.36.79-1.01 1.77-1.84 2.88-2.43 1.11-.59 2.33-.89 3.59-.89h.01c2.2 0 4.28.86 5.84 2.42s2.42 3.64 2.42 5.84c0 2.2-.86 4.28-2.42 5.84-1.56 1.56-3.64 2.42-5.84 2.42-.46 0-.92-.05-1.37-.15-1.46-.32-2.79-1.03-3.88-2.04l-.2-.18-2.8 1.4.7-2.72z"/></svg>
          WhatsApp
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => handleShare('gmail')}>
          <Mail className="w-4 h-4 mr-2" />
          Gmail
        </DropdownMenuItem>
        <DropdownMenuItem onClick={handleCopyMarkdown}>
          <Copy className="w-4 h-4 mr-2" />
          {isCopied ? 'Copied!' : 'Copy Text'}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
