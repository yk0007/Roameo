"use client";

import { useEffect, useMemo, useState } from "react";
import { Minus, Plus, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle
} from "@/components/ui/dialog";

interface TripTravelersDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  travelers?: string;
  onApply: (totalTravelers: number) => void;
}

type TravelerState = {
  adults: number;
  children: number;
  infants: number;
};

const travelerRows: Array<{
  key: keyof TravelerState;
  label: string;
  hint: string;
  min: number;
}> = [
  { key: "adults", label: "Adults", hint: "Ages 13 or above", min: 1 },
  { key: "children", label: "Children", hint: "Ages 2–12", min: 0 },
  { key: "infants", label: "Infants", hint: "Under 2", min: 0 }
];

function deriveInitialState(travelers?: string): TravelerState {
  const parsed = Number.parseInt(travelers || "", 10);
  const adults = Number.isFinite(parsed) && parsed > 0 ? parsed : 1;
  return {
    adults,
    children: 0,
    infants: 0
  };
}

function labelForTravelers(total: number) {
  return `${total} traveler${total === 1 ? "" : "s"}`;
}

export function TripTravelersDialog({
  open,
  onOpenChange,
  travelers,
  onApply
}: TripTravelersDialogProps) {
  const [counts, setCounts] = useState<TravelerState>(() => deriveInitialState(travelers));

  useEffect(() => {
    if (open) {
      setCounts(deriveInitialState(travelers));
    }
  }, [open, travelers]);

  const totalTravelers = useMemo(
    () => counts.adults + counts.children + counts.infants,
    [counts]
  );

  const updateCount = (key: keyof TravelerState, delta: number) => {
    setCounts((current) => {
      const row = travelerRows.find((item) => item.key === key)!;
      const nextValue = Math.max(row.min, current[key] + delta);
      return {
        ...current,
        [key]: nextValue
      };
    });
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-[520px] gap-0 overflow-hidden rounded-[28px] border border-black/5 bg-white p-0 shadow-[0_30px_120px_rgba(15,23,42,0.24)]">

        <DialogHeader className="items-center border-b border-slate-100 px-10 py-5 text-center">
          <DialogTitle className="text-[34px] font-semibold tracking-[-0.04em] text-slate-900">
            Who
          </DialogTitle>
          <div className="mt-2 text-sm font-medium text-slate-500">
            {labelForTravelers(totalTravelers)}
          </div>
        </DialogHeader>

        <div className="px-6 py-3">
          {travelerRows.map((row) => (
            <div
              key={row.key}
              className="flex items-center justify-between border-b border-slate-100 px-2 py-5 last:border-b-0"
            >
              <div>
                <div className="text-[18px] font-semibold text-slate-900">{row.label}</div>
                <div className="mt-1 text-sm text-slate-500">{row.hint}</div>
              </div>
              <div className="flex items-center gap-3">
                <button
                  type="button"
                  onClick={() => updateCount(row.key, -1)}
                  disabled={counts[row.key] <= row.min}
                  className="flex h-8 w-8 items-center justify-center rounded-full border border-slate-200 text-slate-600 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-35"
                >
                  <Minus className="h-4 w-4" />
                </button>
                <div className="w-5 text-center text-base font-medium text-slate-900">
                  {counts[row.key]}
                </div>
                <button
                  type="button"
                  onClick={() => updateCount(row.key, 1)}
                  className="flex h-8 w-8 items-center justify-center rounded-full border border-slate-200 text-slate-600 transition hover:bg-slate-50"
                >
                  <Plus className="h-4 w-4" />
                </button>
              </div>
            </div>
          ))}
        </div>

        <div className="flex justify-end border-t border-slate-100 px-6 py-5">
          <Button
            type="button"
            className="rounded-full bg-black px-8 text-white hover:bg-black/90"
            onClick={() => {
              onApply(totalTravelers);
              onOpenChange(false);
            }}
          >
            Update
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
