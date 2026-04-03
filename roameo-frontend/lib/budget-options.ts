export interface BudgetOption {
  id: string;
  label: string;
  detail: string;
  total: number;
  currency: string;
}

export const BUDGET_OPTIONS: BudgetOption[] = [
  {
    id: "budget_friendly",
    label: "Budget-friendly",
    detail: "Lower-cost picks and practical stays",
    total: 15000,
    currency: "INR"
  },
  {
    id: "mid_range",
    label: "Mid-range",
    detail: "Balanced comfort and value",
    total: 35000,
    currency: "INR"
  },
  {
    id: "comfortable",
    label: "Comfortable",
    detail: "Nicer stays and smoother logistics",
    total: 70000,
    currency: "INR"
  },
  {
    id: "premium",
    label: "Premium",
    detail: "Upscale stays and elevated experiences",
    total: 140000,
    currency: "INR"
  }
];

export function getBudgetOptionByLabel(label?: string) {
  if (!label) {
    return undefined;
  }
  return BUDGET_OPTIONS.find((option) => option.label === label);
}

export function getBudgetOptionByTotal(total?: number) {
  if (typeof total !== "number" || Number.isNaN(total) || total <= 0) {
    return undefined;
  }

  let closest = BUDGET_OPTIONS[0];
  let minDistance = Math.abs(total - closest.total);
  for (const option of BUDGET_OPTIONS.slice(1)) {
    const distance = Math.abs(total - option.total);
    if (distance < minDistance) {
      closest = option;
      minDistance = distance;
    }
  }
  return closest;
}

