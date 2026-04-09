# Roameo Final Year Project Presentation

## Slide 1: Title
**Title:** Roameo  
**Subtitle:** An AI-Powered Travel Planning Workspace with Canonical Session State

**What to say:**
Good morning. My final year project is Roameo, an AI-powered travel planning workspace. The main goal of this project is to move beyond a simple travel chatbot and build a system that can understand user travel requirements, generate structured plans, and keep chat, itinerary, map, and saved places synchronized in one live workspace.

## Slide 2: Problem Statement
**Title:** Problem Statement

**What to say:**
Trip planning is usually fragmented. Users search destinations in one app, save hotels in another, check routes in maps separately, and manually build itineraries in notes. Even many AI-based systems only provide suggestions in chat form, but those suggestions are not converted into a persistent and editable plan. This creates inconsistency and extra user effort. Roameo solves that by making travel planning conversational, structured, and stateful.

## Slide 3: Proposed Solution
**Title:** Proposed Solution

**What to say:**
Roameo is a text-first intelligent travel planning workspace. The user interacts through natural language on the left side, and the right side shows either the map or the itinerary. The system understands the travel request, grounds it using real-world travel tools, creates a structured itinerary, and persists everything in a canonical session snapshot. This gives the user one connected environment instead of many disconnected tools.

## Slide 4: Objectives
**Title:** Project Objectives

**What to say:**
The main objectives were:
- build a conversational travel planner that supports planning and refinement across multiple turns
- create a single source of truth for chat, itinerary, map, and saved POIs
- use AI only where semantic understanding and synthesis are needed
- keep the rest of the pipeline deterministic and reliable
- support real-world travel grounding through maps, weather, holidays, and web research
- provide live updates during planning using streaming

## Slide 5: Core Innovation
**Title:** Core Innovation: Canonical Session Snapshot

**What to say:**
The main technical innovation of Roameo is the canonical session snapshot architecture. Instead of treating chat, itinerary, and map as separate states, the system stores one structured `SessionSnapshot`. That snapshot contains messages, planning state, plan, POI catalog, saved place IDs, follow-up context, and trace events. Because of this, every part of the product reads from the same truth. This prevents the common problem where the chatbot says one thing, but the itinerary or map shows something else.

## Slide 6: System Architecture
**Title:** High-Level Architecture

**What to say:**
The system is divided into four main layers:
- frontend workspace built with Next.js and React
- backend API and turn runtime built with Express and TypeScript
- shared contracts package built with Zod schemas and TypeScript types
- persistence through Supabase, with in-memory fallback for development

The shared contracts layer is important because both frontend and backend use the same schema definitions. That reduces duplication and keeps the data model consistent.

## Slide 7: Tech Stack
**Title:** Technology Stack

**What to say:**
For the frontend I used Next.js 16, React 19, Zustand, and React Query.  
For the backend I used Express, TypeScript, Zod, and Supabase.  
For AI providers the system supports both Gemini and OpenAI.  
For travel grounding it integrates Google Places, Google Directions, Google Geocoding, Open-Meteo, Nager.Date, and Tavily.

This stack was chosen to balance developer productivity, strong typing, real-time UX, and integration with external APIs.

## Slide 8: User Interface
**Title:** Product Interface

**What to say:**
The user interface is a split workspace:
- chat panel on the left
- map or itinerary panel on the right
- top navigation for destination, dates, travelers, and budget

This layout supports both conversation and structured visualization at the same time. The user can ask for a plan, inspect it visually, refine it, and save useful places without leaving the workspace.

## Slide 9: How a User Request Flows
**Title:** End-to-End Request Flow

**What to say:**
When the user sends a message, the system follows a deterministic execution flow:
1. save the user message
2. mark planning state as running
3. check if it is a trivial conversational turn
4. resolve the intent and travel context
5. run discovery or research if needed
6. synthesize or refine the itinerary
7. run feasibility and transit checks
8. generate the final narrative response
9. update memory and follow-up context
10. emit the final session snapshot to the frontend

This makes the system predictable and easier to maintain than a freeform agent loop.

## Slide 10: Agentic Runtime Design
**Title:** Agentic Runtime Design

**What to say:**
Roameo uses a router-first agentic runtime. But here, agentic does not mean uncontrolled autonomy. It means that the system uses specialized stages for understanding, research, plan synthesis, feasibility criticism, transit advice, and narrative generation. The LLM is used only where it adds value, while state writes and tool execution remain deterministic. This design improves reliability and reduces hallucination risk.

## Slide 11: Specialized Sub-Agents
**Title:** Specialized Sub-Agents

**What to say:**
Some important sub-agent responsibilities are:
- semantic router for understanding travel intent
- destination research and discovery handling
- structured itinerary synthesis
- feasibility critic to refine practical quality
- transit advisor to improve movement between activities
- response block assembly for frontend rendering

This is one of the strongest parts of the project because it breaks the problem into controlled reasoning stages instead of using one huge prompt for everything.

## Slide 12: Deterministic Tool Layer
**Title:** Internal Tool and Mutation Layer

**What to say:**
Roameo has a first-class internal tool surface for safe state changes. Examples include:
- reading the session snapshot
- updating trip header details
- editing itinerary
- updating session memory
- resetting active trip context
- saving follow-up context

It also supports direct itinerary mutations such as adding a POI, removing a POI, moving an activity, regenerating a day, rebalancing the trip, and updating the overview. This means the system supports controlled edits rather than blindly regenerating everything from scratch.

## Slide 13: Data Grounding
**Title:** Real-World Data Grounding

**What to say:**
Roameo does not rely only on generated text. It grounds travel planning using external services:
- Google Places for hotels, restaurants, and attractions
- Google Geocoding for destinations and origin resolution
- Google Directions for route support
- Open-Meteo for weather enrichment
- Nager.Date for holidays
- Tavily for deep web research and destination facts

This makes the output more practical and reduces fabricated recommendations.

## Slide 14: State Management and Consistency
**Title:** Why the State Model Matters

**What to say:**
The state model enforces several important rules:
- explicit new trip requests replace stale trip context
- multi-city trips preserve the full destination set
- itinerary updates happen only when the canonical plan changes
- discovery results expand the POI catalog without corrupting the itinerary
- map routes come only from itinerary-linked POIs

These rules matter because travel conversations are iterative. Without proper context management, the system would quickly become inconsistent.

## Slide 15: Frontend Data Flow
**Title:** Frontend as a Thin Consumer

**What to say:**
The frontend is intentionally thin. It loads the current session, opens a live SSE stream, hydrates the Zustand store, and derives map and itinerary views from the session snapshot. The business logic stays on the backend. This is a good architectural choice because it keeps the UI responsive while avoiding duplicate logic across frontend and backend.

## Slide 16: Real-Time Streaming
**Title:** Live Planning Experience

**What to say:**
Roameo uses Server-Sent Events for real-time updates. Events such as turn start, message delta, trace updates, plan updates, and final completion are streamed to the frontend. This gives a live planning experience where users can see progress instead of waiting for one final static answer.

## Slide 17: Persistence and Reliability
**Title:** Persistence Strategy

**What to say:**
The persistence layer stores sessions, messages, plan snapshots, POI catalogs, saved POIs, traces, and provider settings. Supabase is used for the primary persistent backend, and the system also supports an in-memory fallback for development. This makes the project practical for both production-style architecture and local testing.

## Slide 18: Key Strengths
**Title:** Project Strengths

**What to say:**
The major strengths of Roameo are:
- single-source session architecture
- strong separation between AI reasoning and deterministic logic
- multi-turn contextual planning
- real-world data grounding
- live synchronized UI
- modular provider support with Gemini and OpenAI

These are the reasons I consider Roameo more than a chatbot. It is a travel planning system.

## Slide 19: Limitations
**Title:** Limitations

**What to say:**
There are still some limitations:
- quality depends on availability and performance of external APIs
- travel data freshness can vary by provider
- deeper personalization requires more long-term user history
- real-world route and pricing optimization can be expanded further
- full-scale user studies are still a future step

These are normal limitations for a full-stack AI system, and the architecture leaves room to address them in future work.

## Slide 20: Future Scope
**Title:** Future Enhancements

**What to say:**
Future scope includes:
- flight and train integration
- cost-aware optimization
- collaborative group trip planning
- stronger personalization from user preferences and history
- multilingual support
- booking workflows and notification assistance

The system is already designed in a modular way, so these features can be added without breaking the core architecture.

## Slide 21: Conclusion
**Title:** Conclusion

**What to say:**
To conclude, Roameo is an AI-powered travel planning workspace that combines natural language interaction, deterministic orchestration, real-world travel tools, and a canonical session state model. The main contribution of this project is not only generating itineraries, but building a reliable architecture where chat, map, itinerary, and saved travel data remain synchronized across the whole user journey.

## Slide 22: Demo / Q&A
**Title:** Demo and Questions

**What to say:**
Thank you. I can now demonstrate how Roameo plans a trip, streams updates live, supports follow-up refinement, and keeps the itinerary and map consistent inside one session.
