"use client";

import { useEffect, useState } from "react";
import { CheckCircle2, Circle, IndianRupee, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle
} from "@/components/ui/dialog";
import { BUDGET_OPTIONS, getBudgetOptionByLabel } from "@/lib/budget-options";

interface TripBudgetDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  budget?: string;
  onApply: (budget: string) => void;
}

export function TripBudgetDialog({
  open,
  onOpenChange,
  budget,
  onApply
}: TripBudgetDialogProps) {
  const [selectedBudget, setSelectedBudget] = useState<string>(
    budget && getBudgetOptionByLabel(budget) ? budget : BUDGET_OPTIONS[1].label
  );

  useEffect(() => {
    if (!open) {
      return;
    }
    setSelectedBudget(
      budget && getBudgetOptionByLabel(budget) ? budget : BUDGET_OPTIONS[1].label
    );
  }, [open, budget]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-[460px] gap-0 overflow-hidden rounded-[28px] border border-black/5 bg-white p-0 shadow-[0_30px_120px_rgba(15,23,42,0.24)]">

        <DialogHeader className="items-center border-b border-slate-100 px-10 py-5 text-center">
          <DialogTitle className="text-[34px] font-semibold tracking-[-0.04em] text-slate-900">
            Budget
          </DialogTitle>
          <div className="mt-2 text-sm font-medium text-slate-500">
            Pick a budget style
          </div>
        </DialogHeader>

        <div className="space-y-2 px-5 py-5">
          {BUDGET_OPTIONS.map((option) => {
            const selected = option.label === selectedBudget;
            return (
              <button
                key={option.id}
                type="button"
                onClick={() => setSelectedBudget(option.label)}
                className={`flex w-full items-start gap-3 rounded-[20px] border px-4 py-4 text-left transition ${
                  selected
                    ? "border-black bg-slate-50 shadow-[0_10px_30px_rgba(15,23,42,0.08)]"
                    : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50"
                }`}
              >
                <span className="mt-0.5 text-slate-800">
                  {selected ? (
                    <CheckCircle2 className="h-5 w-5" />
                  ) : (
                    <Circle className="h-5 w-5" />
                  )}
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <IndianRupee className="h-4 w-4 text-slate-400" />
                    <span className="text-[16px] font-semibold text-slate-900">
                      {option.label}
                    </span>
                  </div>
                  <div className="mt-1 text-sm leading-6 text-slate-500">
                    {option.detail}
                  </div>
                </div>
              </button>
            );
          })}
        </div>

        <div className="flex justify-end border-t border-slate-100 px-5 py-5">
          <Button
            type="button"
            className="rounded-full bg-black px-8 text-white hover:bg-black/90"
            onClick={() => {
              onApply(selectedBudget);
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

