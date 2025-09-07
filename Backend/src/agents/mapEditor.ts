import type { ChatMessage, Itinerary, POI } from "../types/schemas.js";
import type { WsEvent } from "../types/schemas.js";

export type MapEditResult = {
  map: { pois: POI[]; routes: Array<{ from: [number, number]; to: [number, number]; polyline?: string }> };
  chatResponse: string;
};

// Extract POIs from itinerary activities where lat/lng exist
function collectItineraryPois(itinerary?: Itinerary): POI[] {
  if (!itinerary?.daysPlan?.length) return [] as POI[];
  const items: POI[] = [] as POI[];
  itinerary.daysPlan.forEach((d) => {
    d.activities?.forEach((a: any) => {
      if (a?.lat && a?.lng) {
        items.push({
          id: a.poiId || `${a.lat},${a.lng}`,
          name: a.name || "POI",
          type: "attraction",
          lat: a.lat,
          lng: a.lng,
          address: a.location,
          source: "custom",
        } as POI);
      }
    });
  });
  return items;
}

// Very simple map command parser
function parseMapEdit(message: string):
  | { kind: "show_routes" }
  | { kind: "hide_routes" }
  | { kind: "clear_map" }
  | { kind: "add_marker"; lat: number; lng: number; name?: string }
  | { kind: "fit_itinerary" }
  | null {
  const m = message.toLowerCase();
  if (/show routes?/.test(m)) return { kind: "show_routes" };
  if (/hide routes?/.test(m)) return { kind: "hide_routes" };
  if (/(clear|reset) map/.test(m)) return { kind: "clear_map" };
  const add = m.match(/add marker ([-+]?\d+\.?\d*),\s*([-+]?\d+\.?\d*)/);
  if (add) {
    return { kind: "add_marker", lat: parseFloat(add[1]), lng: parseFloat(add[2]) };
  }
  if (/fit (map|to itinerary|to route|to points)/.test(m)) return { kind: "fit_itinerary" };
  return null;
}

export async function mapEditorAgent(
  itinerary: Itinerary | undefined,
  message: string,
  _history: ChatMessage[],
): Promise<MapEditResult | null> {
  const intent = parseMapEdit(message);
  if (!intent) {
    return {
      map: { pois: collectItineraryPois(itinerary), routes: [] },
      chatResponse:
        "I can adjust the map. Try: 'show routes', 'hide routes', 'clear map', 'add marker 11.41, 76.70', or 'fit to itinerary'.",
    };
  }

  // Base POIs from itinerary
  let pois = collectItineraryPois(itinerary);
  let routes: Array<{ from: [number, number]; to: [number, number]; polyline?: string }> = [];

  switch (intent.kind) {
    case "clear_map":
      pois = [];
      routes = [];
      return { map: { pois, routes }, chatResponse: "Cleared the map." };
    case "add_marker":
      pois = [
        ...pois,
        {
          id: `${intent.lat},${intent.lng}`,
          name: intent.name || "Custom marker",
          type: "attraction",
          lat: intent.lat,
          lng: intent.lng,
          source: "custom",
        } as POI,
      ];
      return { map: { pois, routes }, chatResponse: "Added a marker to the map." };
    case "show_routes":
      // naive route between consecutive itinerary POIs
      for (let i = 1; i < pois.length; i++) {
        routes.push({ from: [pois[i - 1].lat, pois[i - 1].lng], to: [pois[i].lat, pois[i].lng] });
      }
      return { map: { pois, routes }, chatResponse: "Showing routes between itinerary points." };
    case "hide_routes":
      return { map: { pois, routes: [] }, chatResponse: "Hid all routes." };
    case "fit_itinerary":
      // client will auto-fit to provided points
      return { map: { pois, routes }, chatResponse: "Fitting the map to the itinerary points." };
    default:
      return { map: { pois, routes }, chatResponse: "Done." };
  }
}
