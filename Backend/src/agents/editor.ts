import type { ChatMessage, Itinerary } from "../types/schemas.js";
import { searchPOIByName, estimateTravelMinutes } from "./poiLookup.js";

export type EditResult = {
  itinerary: Itinerary;
  chatResponse: string;
};

// --- Simple edit tools on itinerary ---
function addDay(itin: Itinerary, afterDay?: number): Itinerary {
  const next: Itinerary = JSON.parse(JSON.stringify(itin));
  const insertIndex = (afterDay && afterDay > 0)
    ? Math.min(afterDay, next.daysPlan.length)
    : next.daysPlan.length;
  const newDayNumber = (next.daysPlan[insertIndex - 1]?.day || insertIndex) + 1;
  const newDay: any = {
    day: newDayNumber,
    date: new Date().toISOString().slice(0, 10),
    title: "Added day",
    activities: [],
  };
  next.daysPlan.splice(insertIndex, 0, newDay);
  // Re-number days sequentially
  next.daysPlan = next.daysPlan.map((d, i) => ({ ...d, day: i + 1 }));
  next.days = next.daysPlan.length;
  return next;
}

function removeDay(itin: Itinerary, dayNumber: number): Itinerary {
  const next: Itinerary = JSON.parse(JSON.stringify(itin));
  next.daysPlan = next.daysPlan.filter((d) => d.day !== dayNumber);
  next.daysPlan = next.daysPlan.map((d, i) => ({ ...d, day: i + 1 }));
  next.days = next.daysPlan.length;
  return next;
}

async function addActivity(itin: Itinerary, dayNumber: number, name: string): Promise<Itinerary> {
  const next: Itinerary = JSON.parse(JSON.stringify(itin));
  const day = next.daysPlan.find((d) => d.day === dayNumber);
  if (!day) return next;
  day.activities = day.activities || [];
  // Attempt POI lookup to enrich activity with coordinates
  let act: any = { name, start: "09:00", end: "10:00" };
  try {
    const poi = await searchPOIByName(name);
    if (poi) {
      act = {
        ...act,
        poiId: poi.id,
        lat: poi.lat,
        lng: poi.lng,
        location: poi.address,
      };
    }
  } catch {}
  day.activities.push(act);
  return next;
}

function removeActivity(itin: Itinerary, dayNumber: number, name: string): Itinerary {
  const next: Itinerary = JSON.parse(JSON.stringify(itin));
  const day = next.daysPlan.find((d) => d.day === dayNumber);
  if (!day) return next;
  day.activities = (day.activities || []).filter((a) => (a.name || "").toLowerCase() !== name.toLowerCase());
  return next;
}

function moveActivity(itin: Itinerary, fromDay: number, toDay: number, name: string): Itinerary {
  const next: Itinerary = JSON.parse(JSON.stringify(itin));
  const src = next.daysPlan.find((d) => d.day === fromDay);
  const dst = next.daysPlan.find((d) => d.day === toDay);
  if (!src || !dst) return next;
  const idx = (src.activities || []).findIndex((a) => (a.name || "").toLowerCase() === name.toLowerCase());
  if (idx < 0) return next;
  const [act] = (src.activities as any[]).splice(idx, 1);
  dst.activities = dst.activities || [];
  dst.activities.push(act);
  return next;
}

// ---- Travel-time-aware re-timing ----
function parseTime(t: string): number {
  const [h, m] = t.split(":").map((x) => parseInt(x, 10));
  return h * 60 + m;
}
function fmtTime(mins: number): string {
  const h = Math.floor(mins / 60);
  const m = mins % 60;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}

function reTimeDay(itin: Itinerary, dayNumber: number, insertTravel = true): Itinerary {
  const next: Itinerary = JSON.parse(JSON.stringify(itin));
  const day = next.daysPlan.find((d) => d.day === dayNumber);
  if (!day) return next;
  const baseStart = parseTime("09:00");
  let cursor = baseStart;
  const newActs: any[] = [];
  const acts = (day.activities || []) as any[];
  for (let i = 0; i < acts.length; i++) {
    const prev = newActs.length > 0 ? newActs[newActs.length - 1] : null;
    // Insert travel segment if both points have coordinates
    if (
      insertTravel &&
      prev && prev.lat != null && prev.lng != null &&
      acts[i]?.lat != null && acts[i]?.lng != null
    ) {
      const travelMin = estimateTravelMinutes(
        { lat: prev.lat, lng: prev.lng },
        { lat: acts[i].lat, lng: acts[i].lng },
      );
      const tStart = cursor;
      const tEnd = cursor + travelMin;
      newActs.push({
        name: `Travel to ${acts[i].name || "next stop"}`,
        start: fmtTime(tStart),
        end: fmtTime(tEnd),
        description: "Estimated travel time",
      });
      cursor = tEnd;
    }
    const duration = 60; // default 1h per stop for now
    const aStart = cursor;
    const aEnd = cursor + duration;
    newActs.push({ ...acts[i], start: fmtTime(aStart), end: fmtTime(aEnd) });
    cursor = aEnd + 15; // 15m buffer
  }
  day.activities = newActs;
  return next;
}

// --- Very simple rule-based parser ---
function parseEdit(message: string):
  | { kind: "add_day"; afterDay?: number }
  | { kind: "remove_day"; day: number }
  | { kind: "add_activity"; day: number; name: string }
  | { kind: "remove_activity"; day: number; name: string }
  | { kind: "move_activity"; fromDay: number; toDay: number; name: string }
  | { kind: "insert_day"; position: "before" | "after"; refDay: number }
  | { kind: "replace_day"; day: number; theme: string }
  | null {
  const m = message.toLowerCase();
  const addDayMatch = m.match(/add (a )?day( after (day )?(\d+))?/);
  if (addDayMatch) {
    const after = addDayMatch[4] ? parseInt(addDayMatch[4], 10) : undefined;
    return { kind: "add_day", afterDay: after };
  }
  const removeDayMatch = m.match(/remove (day )?(\d+)/);
  if (removeDayMatch) {
    return { kind: "remove_day", day: parseInt(removeDayMatch[2], 10) };
  }
  const addActMatch = m.match(/add (.+) to day (\d+)/);
  if (addActMatch) {
    return { kind: "add_activity", day: parseInt(addActMatch[2], 10), name: addActMatch[1].trim() };
  }
  const removeActMatch = m.match(/remove (.+) from day (\d+)/);
  if (removeActMatch) {
    return { kind: "remove_activity", day: parseInt(removeActMatch[2], 10), name: removeActMatch[1].trim() };
  }
  const moveActMatch = m.match(/move (.+) from day (\d+) to day (\d+)/);
  if (moveActMatch) {
    return {
      kind: "move_activity",
      name: moveActMatch[1].trim(),
      fromDay: parseInt(moveActMatch[2], 10),
      toDay: parseInt(moveActMatch[3], 10),
    };
  }
  const insertBefore = m.match(/insert (a )?day before (day )?(\d+)/);
  if (insertBefore) {
    return { kind: "insert_day", position: "before", refDay: parseInt(insertBefore[3], 10) };
  }
  const insertAfter = m.match(/insert (a )?day after (day )?(\d+)/);
  if (insertAfter) {
    return { kind: "insert_day", position: "after", refDay: parseInt(insertAfter[3], 10) };
  }
  const replaceDay = m.match(/replace (day )?(\d+) with (.+)/);
  if (replaceDay) {
    return { kind: "replace_day", day: parseInt(replaceDay[2], 10), theme: replaceDay[3].trim() };
  }
  return null;
}

export async function editorAgent(itinerary: Itinerary | undefined, message: string, _history: ChatMessage[]): Promise<EditResult | null> {
  if (!itinerary || !itinerary.daysPlan || itinerary.daysPlan.length === 0) {
    return {
      itinerary: itinerary as any,
      chatResponse: "There is no current itinerary to edit. Try planning a trip first.",
    };
  }
  const intent = parseEdit(message);
  if (!intent) {
    return {
      itinerary,
      chatResponse: "I can edit your itinerary. For example: 'add a day', 'remove day 2', 'add breakfast near Ooty to day 1', 'move dinner from day 1 to day 2'.",
    };
  }
  let updated = itinerary;
  switch (intent.kind) {
    case "add_day":
      updated = addDay(itinerary, intent.afterDay);
      return { itinerary: updated, chatResponse: "Added a day to your itinerary." };
    case "remove_day":
      updated = removeDay(itinerary, intent.day);
      return { itinerary: updated, chatResponse: `Removed day ${intent.day}.` };
    case "add_activity":
      updated = await addActivity(itinerary, intent.day, intent.name);
      updated = reTimeDay(updated, intent.day);
      return { itinerary: updated, chatResponse: `Added '${intent.name}' to day ${intent.day}.` };
    case "remove_activity":
      updated = removeActivity(itinerary, intent.day, intent.name);
      updated = reTimeDay(updated, intent.day);
      return { itinerary: updated, chatResponse: `Removed '${intent.name}' from day ${intent.day}.` };
    case "move_activity":
      updated = moveActivity(itinerary, intent.fromDay, intent.toDay, intent.name);
      updated = reTimeDay(updated, intent.fromDay);
      updated = reTimeDay(updated, intent.toDay);
      return { itinerary: updated, chatResponse: `Moved '${intent.name}' to day ${intent.toDay}.` };
    case "insert_day": {
      const next: Itinerary = JSON.parse(JSON.stringify(itinerary));
      const refIndex = Math.max(1, Math.min(next.daysPlan.length, intent.refDay));
      const insertIndex = intent.position === "before" ? refIndex - 1 : refIndex;
      const newDay: any = { day: 0, date: new Date().toISOString().slice(0, 10), title: "New day", activities: [] };
      next.daysPlan.splice(insertIndex, 0, newDay);
      next.daysPlan = next.daysPlan.map((d, i) => ({ ...d, day: i + 1 }));
      next.days = next.daysPlan.length;
      return { itinerary: next, chatResponse: `Inserted a day ${intent.position} day ${intent.refDay}.` };
    }
    case "replace_day": {
      const dayNum = intent.day;
      const theme = intent.theme;
      const next: Itinerary = JSON.parse(JSON.stringify(itinerary));
      const day = next.daysPlan.find((d) => d.day === dayNum);
      if (!day) return { itinerary, chatResponse: `Day ${dayNum} was not found.` };
      // Build a themed scaffold
      const slots = [
        `Breakfast - ${theme}`,
        `Morning experience - ${theme}`,
        `Lunch - ${theme}`,
        `Afternoon highlight - ${theme}`,
        `Dinner - ${theme}`,
      ];
      day.activities = [];
      for (const s of slots) {
        const added = await addActivity(next, dayNum, s);
        // merge back just appended activity to target day (avoid double re-number)
        const target = added.daysPlan.find((d) => d.day === dayNum)!;
        const current = next.daysPlan.find((d) => d.day === dayNum)!;
        current.activities = target.activities;
      }
      const reTimed = reTimeDay(next, dayNum);
      return { itinerary: reTimed, chatResponse: `Replaced day ${dayNum} with a '${theme}' plan.` };
    }
    default:
      return { itinerary, chatResponse: "I couldn't understand that edit. Try 'add a day' or 'remove day 2'." };
  }
}
