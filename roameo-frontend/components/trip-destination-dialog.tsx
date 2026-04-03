"use client";

import { useEffect, useMemo, useState } from "react";
import { MapPin, Plus, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle
} from "@/components/ui/dialog";

interface TripDestinationDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  destination?: string;
  destinations?: string[];
  onApply: (payload: { destination: string; destinations: string[] }) => void;
}

function deriveLocations(destination?: string, destinations?: string[]) {
  const values = (destinations && destinations.length > 0 ? destinations : destination ? [destination] : [""])
    .map((item) => item.trim())
    .filter(Boolean);

  return values.length > 0 ? values : [""];
}

export function TripDestinationDialog({
  open,
  onOpenChange,
  destination,
  destinations,
  onApply
}: TripDestinationDialogProps) {
  const [locations, setLocations] = useState<string[]>(() =>
    deriveLocations(destination, destinations)
  );

  useEffect(() => {
    if (open) {
      setLocations(deriveLocations(destination, destinations));
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  const cleanedLocations = useMemo(
    () => locations.map((value) => value.trim()).filter(Boolean),
    [locations]
  );
  const hasMultipleStops = locations.length > 1;

  const handleChange = (index: number, value: string) => {
    setLocations((current) =>
      current.map((location, currentIndex) =>
        currentIndex === index ? value : location
      )
    );
  };

  const handleRemove = (index: number) => {
    setLocations((current) => {
      const next = current.filter((_, currentIndex) => currentIndex !== index);
      return next.length > 0 ? next : [""];
    });
  };

  const saveDisabled = cleanedLocations.length === 0;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-[520px] gap-0 overflow-hidden rounded-[28px] border border-black/5 bg-white p-0 shadow-[0_30px_120px_rgba(15,23,42,0.24)]">

        <DialogHeader className="items-center border-b border-slate-100 px-10 py-5 text-center">
          <DialogTitle className="text-[34px] font-semibold tracking-[-0.04em] text-slate-900">
            Where
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-4 px-5 py-5">
          {locations.map((location, index) => (
            <div
              key={`${index}-${location}`}
              className="flex items-center gap-3 rounded-[18px] border border-slate-200 bg-white px-3 py-3 shadow-[0_6px_18px_rgba(15,23,42,0.04)]"
            >
              <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-emerald-100 via-lime-100 to-amber-100">
                <MapPin className="h-5 w-5 text-slate-700" />
              </div>
              <div className="min-w-0 flex-1">
                <Input
                  value={location}
                  onChange={(event) => handleChange(index, event.target.value)}
                  placeholder={index === 0 ? "Primary destination" : "Add another stop"}
                  className="h-auto border-0 bg-transparent px-0 py-0 text-[18px] font-semibold text-slate-900 shadow-none focus-visible:ring-0"
                />
                <div className="mt-1 flex items-center gap-1 text-sm text-slate-500">
                  <MapPin className="h-3.5 w-3.5" />
                  <span>{index === 0 ? "Primary stop" : "Road trip stop"}</span>
                </div>
              </div>
              {locations.length > 1 && (
                <button
                  type="button"
                  onClick={() => handleRemove(index)}
                  className="flex h-8 w-8 items-center justify-center rounded-full text-slate-400 transition hover:bg-slate-100 hover:text-slate-700"
                  aria-label="Remove location"
                >
                  <X className="h-4 w-4" />
                </button>
              )}
            </div>
          ))}

          <div className="flex items-center justify-between gap-4">
            <Button
              type="button"
              variant="outline"
              className="rounded-full border-slate-200 bg-slate-50 px-4 text-slate-700 hover:bg-slate-100"
              onClick={() => {
                setLocations((current) => [...current, ""]);
              }}
            >
              <Plus className="mr-1.5 h-4 w-4" />
              Add location
            </Button>

            <div className="flex items-center gap-3 rounded-full bg-slate-50 px-3 py-2 text-sm text-slate-600">
              <span className="font-medium text-slate-700">Road trip?</span>
              <Switch
                className="data-[state=checked]:bg-black data-[state=unchecked]:bg-slate-300"
                checked={hasMultipleStops}
                onCheckedChange={(checked) => {
                  setLocations((current) => {
                    if (checked) {
                      return current.length > 1 ? current : [...current, ""];
                    }
                    return [current[0] || ""];
                  });
                }}
              />
            </div>
          </div>
        </div>

        <div className="flex justify-end border-t border-slate-100 px-5 py-5">
          <Button
            type="button"
            className="rounded-full bg-black px-8 text-white hover:bg-black/90"
            disabled={saveDisabled}
            onClick={() => {
              const nextDestinations =
                cleanedLocations.length <= 1
                  ? cleanedLocations.slice(0, 1)
                  : hasMultipleStops
                    ? cleanedLocations
                    : cleanedLocations.slice(0, 1);
              onApply({
                destination: nextDestinations[0],
                destinations: nextDestinations
              });
              onOpenChange(false);
            }}
          >
            Save
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
