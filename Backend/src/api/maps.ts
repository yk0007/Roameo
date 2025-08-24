import { Router } from "express";
import fetch from "node-fetch";

const GOOGLE_MAPS_API_BASE = "https://maps.googleapis.com/maps/api";

export function buildMapsRouter(): Router {
  const router = Router();

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
