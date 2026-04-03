"use client";

import { useEffect, useMemo, useState } from "react";
import type { DateFlexibility } from "@roameo/contracts";
import type { DateRange } from "react-day-picker";
import {
  addDays,
  addMonths,
  differenceInCalendarDays,
  format,
  parseISO,
  startOfMonth
} from "date-fns";
import { CalendarDays, ChevronLeft, ChevronRight, Minus, Plus, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Calendar } from "@/components/ui/calendar";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle
} from "@/components/ui/dialog";

interface TripDateDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  startDate?: string;
  endDate?: string;
  dateFlexibility?: DateFlexibility;
  onApply: (range: {
    startDate: string;
    endDate: string;
    totalDays: number;
    dateFlexibility?: DateFlexibility;
  }) => void;
}

function parseRange(startDate?: string, endDate?: string): DateRange | undefined {
  if (!startDate) {
    return undefined;
  }

  return {
    from: parseISO(startDate),
    to: endDate ? parseISO(endDate) : parseISO(startDate)
  };
}

function buildExactLabel(range?: DateRange) {
  if (!range?.from) {
    return "Pick your travel dates";
  }

  const to = range.to || range.from;
  const totalDays = differenceInCalendarDays(to, range.from) + 1;
  const sameMonth = format(range.from, "MMM yyyy") === format(to, "MMM yyyy");
  const dateLabel = sameMonth
    ? `${format(range.from, "MMM d")} – ${format(to, "d")}`
    : `${format(range.from, "MMM d")} – ${format(to, "MMM d")}`;

  return `${dateLabel} · ${totalDays} day${totalDays === 1 ? "" : "s"}`;
}

function buildFlexibleLabel(month: Date, totalDays: number) {
  return `${format(month, "MMM yyyy")} · ${totalDays} day${totalDays === 1 ? "" : "s"}`;
}

export function TripDateDialog({
  open,
  onOpenChange,
  startDate,
  endDate,
  dateFlexibility,
  onApply
}: TripDateDialogProps) {
  const initialRange = useMemo(() => parseRange(startDate, endDate), [startDate, endDate]);
  const initialMode = dateFlexibility && dateFlexibility !== "exact" ? "flexible" : "dates";
  const initialFlexibleDays = useMemo(() => {
    if (!initialRange?.from) {
      return 2;
    }
    const to = initialRange.to || initialRange.from;
    return Math.max(differenceInCalendarDays(to, initialRange.from) + 1, 1);
  }, [initialRange]);
  const initialMonth = useMemo(
    () => startOfMonth(initialRange?.from || new Date()),
    [initialRange]
  );

  const [mode, setMode] = useState<"dates" | "flexible">(initialMode);
  const [range, setRange] = useState<DateRange | undefined>(initialRange);
  const [flexibleDays, setFlexibleDays] = useState(initialFlexibleDays);
  const [monthCursor, setMonthCursor] = useState(initialMonth);
  const [selectedFlexibleMonth, setSelectedFlexibleMonth] = useState(initialMonth);

  useEffect(() => {
    if (!open) {
      return;
    }
    setMode(initialMode);
    setRange(initialRange);
    setFlexibleDays(initialFlexibleDays);
    setMonthCursor(initialMonth);
    setSelectedFlexibleMonth(initialMonth);
  }, [open, initialFlexibleDays, initialMode, initialMonth, initialRange]);

  const monthOptions = useMemo(
    () => Array.from({ length: 6 }, (_, index) => addMonths(monthCursor, index)),
    [monthCursor]
  );

  const selectedLabel =
    mode === "dates"
      ? buildExactLabel(range)
      : buildFlexibleLabel(selectedFlexibleMonth, flexibleDays);

  const handleApply = () => {
    if (mode === "dates") {
      if (!range?.from) {
        return;
      }
      const to = range.to || addDays(range.from, 1);
      onApply({
        startDate: format(range.from, "yyyy-MM-dd"),
        endDate: format(to, "yyyy-MM-dd"),
        totalDays: differenceInCalendarDays(to, range.from) + 1,
        dateFlexibility: "exact"
      });
      onOpenChange(false);
      return;
    }

    const approximateStart = startOfMonth(selectedFlexibleMonth);
    const approximateEnd = addDays(approximateStart, Math.max(flexibleDays - 1, 0));
    onApply({
      startDate: format(approximateStart, "yyyy-MM-dd"),
      endDate: format(approximateEnd, "yyyy-MM-dd"),
      totalDays: flexibleDays,
      dateFlexibility: "approximate"
    });
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-[860px] gap-0 overflow-hidden rounded-[28px] border border-black/5 bg-white p-0 shadow-[0_30px_120px_rgba(15,23,42,0.24)]">

        <DialogHeader className="items-center border-b border-slate-100 px-10 py-5 text-center">
          <DialogTitle className="text-[34px] font-semibold tracking-[-0.04em] text-slate-900">
            When
          </DialogTitle>
          <div className="mt-2 text-sm font-medium text-slate-500">{selectedLabel}</div>
        </DialogHeader>

        <div className="px-8 py-6">
          <div className="mx-auto flex w-fit items-center rounded-full bg-slate-100 p-1">
            <button
              type="button"
              onClick={() => setMode("dates")}
              className={`rounded-full px-5 py-2 text-sm font-medium transition ${
                mode === "dates" ? "bg-white text-slate-900 shadow-sm" : "text-slate-500"
              }`}
            >
              Dates
            </button>
            <button
              type="button"
              onClick={() => setMode("flexible")}
              className={`rounded-full px-5 py-2 text-sm font-medium transition ${
                mode === "flexible" ? "bg-white text-slate-900 shadow-sm" : "text-slate-500"
              }`}
            >
              Flexible
            </button>
          </div>

          {mode === "dates" ? (
            <div className="mt-6 flex justify-center pt-2">
              <Calendar
                mode="range"
                numberOfMonths={2}
                selected={range}
                onSelect={setRange}
                defaultMonth={range?.from}
                className="rounded-3xl bg-white p-0"
                classNames={{
                  months: "flex flex-col gap-10 md:flex-row",
                  month: "relative flex flex-col gap-4",
                  nav: "pointer-events-none absolute inset-x-0 top-1 flex items-center justify-between px-3 md:px-5",
                  month_caption:
                    "relative mb-4 flex h-10 items-center justify-center px-16 text-base font-semibold text-slate-900",
                  weekday:
                    "flex-1 text-center text-xs font-medium uppercase tracking-[0.18em] text-slate-400",
                  day: "aspect-square p-0 text-center",
                  button_previous:
                    "pointer-events-auto flex h-9 w-9 items-center justify-center rounded-full border border-slate-200 bg-white text-slate-700 shadow-sm hover:bg-slate-50",
                  button_next:
                    "pointer-events-auto flex h-9 w-9 items-center justify-center rounded-full border border-slate-200 bg-white text-slate-700 shadow-sm hover:bg-slate-50"
                }}
                components={{
                  DayButton: ({ className, day, modifiers, ...props }) => {
                    const selected =
                      modifiers.selected ||
                      modifiers.range_start ||
                      modifiers.range_end;
                    return (
                      <button
                        type="button"
                        data-selected={selected}
                        className={`h-11 w-11 rounded-full text-sm font-medium transition ${
                          selected
                            ? "bg-black text-white shadow-[0_10px_24px_rgba(15,23,42,0.24)]"
                            : modifiers.range_middle
                              ? "rounded-none bg-slate-100 text-slate-900"
                              : "text-slate-700 hover:bg-slate-100"
                        } ${className || ""}`}
                        {...props}
                      >
                        {day.date.getDate()}
                      </button>
                    );
                  }
                }}
              />
            </div>
          ) : (
            <div className="mx-auto mt-8 max-w-[700px]">
              <div className="text-center">
                <div className="text-[30px] font-semibold tracking-[-0.04em] text-slate-900">
                  How many days?
                </div>
                <div className="mt-4 flex items-center justify-center gap-3">
                  <button
                    type="button"
                    onClick={() => setFlexibleDays((current) => Math.max(current - 1, 1))}
                    className="flex h-11 w-11 items-center justify-center rounded-full border border-slate-200 bg-white text-slate-700 shadow-sm transition hover:bg-slate-50"
                    aria-label="Decrease trip duration"
                  >
                    <Minus className="h-4 w-4" />
                  </button>
                  <div className="min-w-[78px] rounded-full border border-slate-200 bg-white px-6 py-3 text-center text-[24px] font-semibold text-slate-900 shadow-sm">
                    {flexibleDays}
                  </div>
                  <button
                    type="button"
                    onClick={() => setFlexibleDays((current) => Math.min(current + 1, 21))}
                    className="flex h-11 w-11 items-center justify-center rounded-full border border-slate-200 bg-white text-slate-700 shadow-sm transition hover:bg-slate-50"
                    aria-label="Increase trip duration"
                  >
                    <Plus className="h-4 w-4" />
                  </button>
                </div>
              </div>

              <div className="mt-12 text-center">
                <div className="text-[28px] font-semibold tracking-[-0.04em] text-slate-900">
                  Travel anytime
                </div>
                <div className="mt-5 flex items-center gap-3">
                  <button
                    type="button"
                    onClick={() => setMonthCursor((current) => addMonths(current, -6))}
                    className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full border border-slate-200 bg-white text-slate-700 shadow-sm transition hover:bg-slate-50"
                    aria-label="Show earlier months"
                  >
                    <ChevronLeft className="h-4 w-4" />
                  </button>
                  <div className="grid flex-1 grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-6">
                    {monthOptions.map((month) => {
                      const selected = format(month, "yyyy-MM") === format(selectedFlexibleMonth, "yyyy-MM");
                      return (
                        <button
                          key={format(month, "yyyy-MM")}
                          type="button"
                          onClick={() => setSelectedFlexibleMonth(month)}
                          className={`flex min-h-[104px] flex-col items-center justify-center rounded-[22px] border px-4 py-4 text-center transition ${
                            selected
                              ? "border-black bg-black text-white shadow-[0_18px_48px_rgba(15,23,42,0.2)]"
                              : "border-slate-200 bg-white text-slate-800 hover:border-slate-300 hover:bg-slate-50"
                          }`}
                        >
                          <CalendarDays
                            className={`h-5 w-5 ${selected ? "text-white" : "text-slate-400"}`}
                          />
                          <span className="mt-3 text-[17px] font-semibold">
                            {format(month, "MMMM")}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                  <button
                    type="button"
                    onClick={() => setMonthCursor((current) => addMonths(current, 6))}
                    className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full border border-slate-200 bg-white text-slate-700 shadow-sm transition hover:bg-slate-50"
                    aria-label="Show later months"
                  >
                    <ChevronRight className="h-4 w-4" />
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="flex items-center justify-between border-t border-slate-100 px-8 py-5">
          <Button
            type="button"
            variant="ghost"
            className="rounded-full px-4 text-slate-500 hover:bg-slate-100 hover:text-slate-900"
            onClick={() => {
              setRange(undefined);
              setFlexibleDays(2);
              setSelectedFlexibleMonth(startOfMonth(new Date()));
            }}
          >
            Clear
          </Button>
          <Button
            type="button"
            className="rounded-full bg-black px-8 text-white hover:bg-black/90"
            onClick={handleApply}
            disabled={mode === "dates" ? !range?.from : false}
          >
            Update
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
