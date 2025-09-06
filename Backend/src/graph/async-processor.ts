import type { WsEvent } from "../types/schemas.js";
import type { WsHub } from "../ws/emit.js";
import { plannerAgent } from "../agents/planner.js";
import { poiAgent } from "../agents/poi.js";
import { mapAgent } from "../agents/map.js";
import { generateSessionTitle } from "../agents/title.js";
import { randomUUID } from "crypto";
import type { TripContext } from "../types/schemas.js";
import type { Message } from "../db/types.js";

export class AsyncBackgroundProcessor {
  private hub: WsHub;
  private processingQueue = new Map<string, boolean>();

  constructor(hub: WsHub) {
    this.hub = hub;
  }

  // Process itinerary generation in background and emit events
  async processItineraryGeneration(
    sessionId: string,
    trip: TripContext,
    message: string,
    history: Message[],
  ): Promise<void> {
    // Prevent duplicate processing for same session
    if (this.processingQueue.has(sessionId)) {
      console.log(
        `[async-processor] Already processing itinerary for session ${sessionId}`,
      );
      return;
    }

    this.processingQueue.set(sessionId, true);

    try {
      console.log(
        `[async-processor] Starting background itinerary generation for session ${sessionId}`,
      );

      // Emit status update
      this.hub.emit(sessionId, {
        type: "planning.status",
        data: { status: "Creating detailed itinerary..." },
      });

      console.log(
        `[async-processor] Calling plannerAgent for destination: ${trip.destination}, days: ${trip.days}`,
      );
      const res = await plannerAgent(trip, message, history);
      if (!res) {
        console.warn(
          `[async-processor] Planning failed for session ${sessionId} - no result returned`,
        );
        this.hub.emit(sessionId, {
          type: "chat.append",
          data: {
            id: randomUUID(),
            role: "assistant",
            content:
              "I encountered a small issue while creating your detailed itinerary. The good news is I can still help you plan your trip! Feel free to ask me specific questions about your destination.",
            createdAt: new Date().toISOString(),
          },
        });
        this.hub.emit(sessionId, {
          type: "planning.status",
          data: { status: "Ready to help with your planning!" },
        });
        return;
      }

      console.log(
        `[async-processor] Planning successful for session ${sessionId}, generated ${res.itinerary?.daysPlan?.length || 0} days`,
      );

      const updatedTrip = {
        ...trip,
        destination: res.destination,
        destinations: res.destinations,
        days: res.days,
        destinationImageUrl: res.destinationImageUrl,
      };

      // Generate title with fallback
      let title: string;
      try {
        title = await generateSessionTitle({
          message,
          origin: updatedTrip.origin,
          destination: res.destination,
          days: res.days,
          existingTitle: trip.title,
        });
      } catch (error: any) {
        console.log(
          `[async-processor] Title generation failed: ${error?.message || error}`,
        );
        const sessionSuffix = Math.random()
          .toString(36)
          .substring(2, 5)
          .toUpperCase();
        title = res.destination
          ? `✨ ${res.destination} Adventure #${sessionSuffix}`
          : `✨ Dream Trip #${sessionSuffix}`;
      }

      // Emit navbar update
      this.hub.emit(sessionId, {
        type: "navbar.update",
        data: {
          title,
          destination: res.destination,
          destinations: res.destinations,
          days: res.days,
          destinationImageUrl: res.destinationImageUrl,
        },
      });

      // Emit detailed AI response if different from quick response
      if (res.chatResponse && res.chatResponse.length > 100) {
        this.hub.emit(sessionId, {
          type: "chat.append",
          data: {
            id: randomUUID(),
            role: "assistant",
            content: `Here's your detailed itinerary:\n\n${res.chatResponse}`,
            createdAt: new Date().toISOString(),
          },
        });
      }

      // Emit itinerary if available
      if (res.itinerary.daysPlan && res.itinerary.daysPlan.length > 0) {
        console.log(
          `[async-processor] Emitting itinerary with ${res.itinerary.daysPlan.length} days for session ${sessionId}`,
        );
        this.hub.emit(sessionId, this.emitItineraryUpdate(res.itinerary));

        // Process POI search and mapping
        await this.processPoiAndMapping(
          sessionId,
          res.destinations || [res.destination],
        );
      } else {
        console.warn(
          `[async-processor] No itinerary days generated for session ${sessionId}`,
        );
        this.hub.emit(sessionId, {
          type: "chat.append",
          data: {
            id: randomUUID(),
            role: "assistant",
            content:
              "I've processed your request! While I couldn't generate a detailed daily itinerary this time, I'm here to help you explore your destination. What specific aspects of your trip would you like to discuss?",
            createdAt: new Date().toISOString(),
          },
        });
      }

      this.hub.emit(sessionId, {
        type: "planning.status",
        data: { status: "Planning completed successfully!" },
      });

      console.log(
        `[async-processor] Completed background processing for session ${sessionId}`,
      );
    } catch (error) {
      console.error(
        `[async-processor] Background processing failed for session ${sessionId}:`,
        error,
      );

      // Determine error type for better user feedback
      let errorMessage =
        "I had a small hiccup creating your detailed itinerary, but I'm working on it! Feel free to ask me any questions about your trip in the meantime.";

      if (error instanceof Error) {
        if (error.message.includes("JSON")) {
          console.error(
            `[async-processor] JSON parsing error for session ${sessionId}:`,
            error.message,
          );
          errorMessage =
            "I'm having a technical issue with the itinerary format, but I can still help you plan your trip! What would you like to know about your destination?";
        } else if (error.message.includes("timeout")) {
          console.error(
            `[async-processor] Timeout error for session ${sessionId}`,
          );
          errorMessage =
            "Creating your itinerary is taking longer than expected. I'm still working on it in the background! In the meantime, what questions do you have about your trip?";
        } else if (error.message.includes("API")) {
          console.error(
            `[async-processor] API error for session ${sessionId}:`,
            error.message,
          );
          errorMessage =
            "I'm experiencing some connectivity issues, but I'm still here to help plan your trip! What aspects of your journey would you like to discuss?";
        }
      }

      // Emit error status
      this.hub.emit(sessionId, {
        type: "planning.status",
        data: { status: "Ready to help with your planning!" },
      });

      // Emit fallback response with specific error handling
      this.hub.emit(sessionId, {
        type: "chat.append",
        data: {
          id: randomUUID(),
          role: "assistant",
          content: errorMessage,
          createdAt: new Date().toISOString(),
        },
      });
    } finally {
      this.processingQueue.delete(sessionId);
    }
  }

  // Process POI search and mapping
  private async processPoiAndMapping(
    sessionId: string,
    destinations: string[],
  ): Promise<void> {
    try {
      const searchDestination = destinations[0]; // Use first destination
      console.log(
        `[async-processor] Starting POI search for ${searchDestination} in session ${sessionId}`,
      );

      this.hub.emit(sessionId, {
        type: "search.status",
        data: { status: `Finding amazing places in ${searchDestination}...` },
      });

      const poiEvt = await poiAgent({ destination: searchDestination });
      if (poiEvt) {
        console.log(
          `[async-processor] POI search successful for ${searchDestination}, emitting results`,
        );
        this.hub.emit(sessionId, poiEvt);

        if (poiEvt.type === "search.results") {
          const pois = [
            ...poiEvt.data.stays,
            ...poiEvt.data.restaurants,
            ...poiEvt.data.attractions,
          ];

          console.log(
            `[async-processor] Found ${pois.length} POIs, generating map for session ${sessionId}`,
          );

          this.hub.emit(sessionId, {
            type: "map.status",
            data: { status: "Updating map with your destinations..." },
          });

          const mapEvt = await mapAgent(pois);
          this.hub.emit(sessionId, mapEvt);
          console.log(
            `[async-processor] Map update completed for session ${sessionId}`,
          );
        }
      } else {
        console.warn(
          `[async-processor] No POI results for ${searchDestination} in session ${sessionId}`,
        );
      }
    } catch (error) {
      console.error(
        `[async-processor] POI and mapping failed for session ${sessionId}:`,
        error,
      );

      // Don't let POI/mapping failures break the entire flow
      this.hub.emit(sessionId, {
        type: "chat.append",
        data: {
          id: randomUUID(),
          role: "assistant",
          content:
            "I had a small issue finding places and updating the map, but your itinerary is ready! You can still explore and plan your trip.",
          createdAt: new Date().toISOString(),
        },
      });
    }
  }

  // Helper method to create itinerary update event
  private emitItineraryUpdate(itinerary: any): WsEvent {
    return {
      type: "itinerary.update",
      data: itinerary,
    };
  }

  // Process destination search in background
  async processDestinationSearch(
    sessionId: string,
    destination: string,
    message: string,
    history: Message[],
  ): Promise<void> {
    if (this.processingQueue.has(sessionId)) {
      return;
    }

    this.processingQueue.set(sessionId, true);

    try {
      console.log(
        `[async-processor] Starting destination search for ${destination} in session ${sessionId}`,
      );

      this.hub.emit(sessionId, {
        type: "search.status",
        data: { status: `Exploring ${destination} for you...` },
      });

      const poiEvt = await poiAgent({ destination });
      if (poiEvt) {
        this.hub.emit(sessionId, poiEvt);

        if (poiEvt.type === "search.results") {
          const pois = [
            ...poiEvt.data.stays,
            ...poiEvt.data.restaurants,
            ...poiEvt.data.attractions,
          ];

          const mapEvt = await mapAgent(pois);
          this.hub.emit(sessionId, mapEvt);

          // Emit follow-up response with POI insights
          const totalPois = pois.length;
          if (totalPois > 0) {
            this.hub.emit(sessionId, {
              type: "chat.append",
              data: {
                id: randomUUID(),
                role: "assistant",
                content: `I found ${totalPois} amazing places in ${destination}! Check out the search results and map. Would you like me to create a detailed itinerary for your visit?`,
                createdAt: new Date().toISOString(),
              },
            });
          }
        }
      }
    } catch (error) {
      console.error(
        `[async-processor] Destination search failed for session ${sessionId}:`,
        error,
      );
    } finally {
      this.processingQueue.delete(sessionId);
    }
  }

  // Check if processing is active for a session
  isProcessing(sessionId: string): boolean {
    return this.processingQueue.has(sessionId);
  }

  // Get processing status for debugging
  getProcessingStatus(): { sessionId: string; active: boolean }[] {
    return Array.from(this.processingQueue.entries()).map(
      ([sessionId, active]) => ({
        sessionId,
        active,
      }),
    );
  }

  // Clear processing queue (for cleanup)
  clearQueue(): void {
    this.processingQueue.clear();
  }
}
