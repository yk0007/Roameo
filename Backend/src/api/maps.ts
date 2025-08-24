import { Router } from "express";
import fetch from "node-fetch";

const GOOGLE_MAPS_API_BASE = "https://maps.googleapis.com/maps/api";

const GOOGLE_MAPS_API_KEY = process.env.GOOGLE_MAPS_API_KEY;

export function buildMapsRouter(): Router {
  const router = Router();

  // Endpoint to get the API key for loading the Maps JavaScript API
  router.get("/api-key", (req, res) => {
    if (!GOOGLE_MAPS_API_KEY) {
      return res.status(500).json({ error: "Google Maps API key not configured" });
    }

    // Return a masked version for security - only show first 6 and last 4 characters
    const maskedKey = GOOGLE_MAPS_API_KEY.substring(0, 6) + '...' + GOOGLE_MAPS_API_KEY.substring(GOOGLE_MAPS_API_KEY.length - 4);
    console.log(`Maps API key requested - using key: ${maskedKey}`);
    
    res.json({ apiKey: GOOGLE_MAPS_API_KEY });
  });

  // Proxy for Place Autocomplete API
  router.get("/autocomplete", async (req, res) => {
    const input = req.query.input as string;
    if (!input) {
      return res.status(400).json({ error: "Missing input query parameter" });
    }

    const apiKey = process.env.GOOGLE_MAPS_API_KEY;
    const url = `${GOOGLE_MAPS_API_BASE}/place/autocomplete/json?input=${encodeURIComponent(input)}&key=${apiKey}`;

    try {
      const response = await fetch(url);
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error("Error proxying Google Maps Autocomplete API:", error);
      res.status(500).json({ error: "Internal server error" });
    }
  });

  // Proxy for Place Details API
  router.get("/details", async (req, res) => {
    const place_id = req.query.place_id as string;
    if (!place_id) {
      return res.status(400).json({ error: "Missing place_id query parameter" });
    }

    const apiKey = process.env.GOOGLE_MAPS_API_KEY;
    const url = `${GOOGLE_MAPS_API_BASE}/place/details/json?place_id=${encodeURIComponent(place_id)}&key=${apiKey}`;

    try {
      const response = await fetch(url);
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error("Error proxying Google Maps Details API:", error);
      res.status(500).json({ error: "Internal server error" });
    }
  });

  return router;
}
