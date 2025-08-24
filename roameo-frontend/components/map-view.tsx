"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import { createRoot } from "react-dom/client"
import { Plus, Minus, Navigation, SlidersHorizontal } from "lucide-react"
import { Button } from "@/components/ui/button"
import { CompactPoiCard } from "./poi-card"
import type { Itinerary, POI } from "@/lib/types"

type MapData = { pois: POI[]; routes: Array<{ from: [number, number]; to: [number, number]; polyline?: string }> }

// Custom Google Maps style provided by the user
const CUSTOM_MAP_STYLE: any[] = [
  { featureType: "administrative", elementType: "all", stylers: [{ visibility: "on" }] },
  { featureType: "administrative", elementType: "labels.text.fill", stylers: [{ color: "#444444" }] },
  { featureType: "administrative.neighborhood", elementType: "all", stylers: [{ visibility: "off" }] },
  { featureType: "administrative.land_parcel", elementType: "all", stylers: [{ visibility: "off" }] },
  { featureType: "landscape", elementType: "all", stylers: [{ color: "#f2f2f2" }] },
  { featureType: "landscape.man_made", elementType: "all", stylers: [{ visibility: "on" }] },
  { featureType: "poi", elementType: "all", stylers: [{ visibility: "off" }] },
  { featureType: "poi.attraction", elementType: "all", stylers: [{ visibility: "off" }] },
  { featureType: "poi.business", elementType: "all", stylers: [{ visibility: "off" }] },
  { featureType: "poi.government", elementType: "all", stylers: [{ visibility: "off" }] },
  { featureType: "poi.medical", elementType: "all", stylers: [{ visibility: "off" }] },
  { featureType: "poi.park", elementType: "all", stylers: [{ visibility: "off" }] },
  { featureType: "poi.place_of_worship", elementType: "all", stylers: [{ visibility: "off" }] },
  { featureType: "poi.school", elementType: "all", stylers: [{ visibility: "off" }] },
  { featureType: "poi.sports_complex", elementType: "all", stylers: [{ visibility: "off" }] },
  { featureType: "road", elementType: "all", stylers: [{ saturation: -100 }, { lightness: 45 }, { visibility: "on" }] },
  { featureType: "road", elementType: "geometry.fill", stylers: [{ saturation: 0 }, { visibility: "on" }, { color: "#fefefe" }] },
  { featureType: "road", elementType: "labels.text", stylers: [{ color: "#303030" }] },
  { featureType: "road", elementType: "labels.text.fill", stylers: [{ color: "#aca9a9" }, { visibility: "off" }] },
  { featureType: "road", elementType: "labels.text.stroke", stylers: [{ weight: 0.64 }, { color: "#393939" }, { visibility: "on" }] },
  { featureType: "road.highway", elementType: "all", stylers: [{ visibility: "on" }] },
  { featureType: "road.highway", elementType: "geometry.fill", stylers: [{ color: "#f9bc1e" }] },
  { featureType: "road.highway", elementType: "geometry.stroke", stylers: [{ visibility: "off" }] },
  { featureType: "road.highway", elementType: "labels.text.fill", stylers: [{ visibility: "off" }] },
  { featureType: "road.highway", elementType: "labels.text.stroke", stylers: [{ weight: 2.99 }, { visibility: "on" }] },
  { featureType: "road.highway.controlled_access", elementType: "all", stylers: [{ visibility: "on" }] },
  { featureType: "road.arterial", elementType: "all", stylers: [{ visibility: "off" }] },
  { featureType: "road.arterial", elementType: "labels.icon", stylers: [{ visibility: "off" }] },
  { featureType: "road.local", elementType: "all", stylers: [{ visibility: "simplified" }] },
  { featureType: "transit", elementType: "all", stylers: [{ visibility: "on" }] },
  { featureType: "water", elementType: "all", stylers: [{ color: "#46bcec" }, { visibility: "on" }] },
]

declare global {
  interface Window {
    google?: any
    __gmapsLoading?: boolean
    __gmapsInitCallbacks?: Array<() => void>
  }
}


export function MapView({
  mapData,
  savedIds,
  itinerary,
  onToggleSave,
  onAddPoi,
  onReplan,
  isVisible = true,
}: {
  mapData?: MapData
  savedIds?: Set<string>
  itinerary?: Itinerary
  onToggleSave?: (poi: POI, nextSaved: boolean) => void
  onAddPoi?: (poi: POI) => void
  onReplan?: (poi: POI) => void
  isVisible?: boolean
}) {
  const mapRef = useRef<HTMLDivElement | null>(null)
  const mapInstance = useRef<any | null>(null)
  const markersRef = useRef<any[]>([])
  const polylinesRef = useRef<any[]>([])
  const userMarkerRef = useRef<any | null>(null)
  const infoWindowRef = useRef<any | null>(null)
  const activeInfoWindowRef = useRef<any | null>(null)
  const overlayRef = useRef<any | null>(null)
  const markersByIdRef = useRef<Record<string, any>>({})
  const hoverTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const persistentInfoWindowRef = useRef<any | null>(null)
  const [customStyle, setCustomStyle] = useState(true)
  const [zoom, setZoom] = useState(5)
  const [autoZoomOnHover, setAutoZoomOnHover] = useState(true)
  const [filterAll, setFilterAll] = useState(true)
  const [filterTypes, setFilterTypes] = useState<{ stay: boolean; restaurant: boolean; attraction: boolean }>(
    { stay: true, restaurant: true, attraction: true }
  )
  const [savedOnly, setSavedOnly] = useState(false)
  const itineraryPoiIds = useMemo(() => {
    if (!itinerary) return new Set<string>()
    const ids = new Set<string>()
    itinerary.daysPlan.forEach(day => {
      day.activities.forEach(act => act.poiId && ids.add(act.poiId))
      if (day.accommodation?.poiId) ids.add(day.accommodation.poiId)
    })
    return ids
  }, [itinerary])

  const [addedOnly, setAddedOnly] = useState(!!(itineraryPoiIds && itineraryPoiIds.size > 0))

  useEffect(() => {
    setAddedOnly(!!(itineraryPoiIds && itineraryPoiIds.size > 0))
  }, [itineraryPoiIds])
  const [showPrices, setShowPrices] = useState(true)
  const [showFilterMenu, setShowFilterMenu] = useState(false)
  const filterMenuRef = useRef<HTMLDivElement | null>(null)
  const [gmapsReady, setGmapsReady] = useState<boolean>(!!(typeof window !== "undefined" && (window as any).google?.maps))
  const resizeObsRef = useRef<ResizeObserver | null>(null)

  const pois = mapData?.pois || []
  const filteredPois = useMemo(() => {
    const list = pois.filter((p) => {
      if (!filterAll) {
        if (p.type === "stay" && !filterTypes.stay) return false
        if (p.type === "restaurant" && !filterTypes.restaurant) return false
        if (p.type === "attraction" && !filterTypes.attraction) return false
      }
      if (savedOnly) {
        if (!savedIds) return false
        if (!savedIds.has(p.id)) return false
      }
      if (addedOnly) {
        if (!itineraryPoiIds) return false
        if (!itineraryPoiIds.has(p.id)) return false
      }
      return true
    })
    return list
  }, [pois, filterAll, filterTypes, savedOnly, savedIds, addedOnly, itineraryPoiIds])

  // Close filter menu on outside click / Esc
  useEffect(() => {
    function onDocMouseDown(e: MouseEvent) {
      if (!showFilterMenu) return
      if (!filterMenuRef.current) return
      if (e.target instanceof Node && !filterMenuRef.current.contains(e.target)) {
        setShowFilterMenu(false)
      }
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setShowFilterMenu(false)
    }
    document.addEventListener("mousedown", onDocMouseDown)
    document.addEventListener("keydown", onKey)
    return () => {
      document.removeEventListener("mousedown", onDocMouseDown)
      document.removeEventListener("keydown", onKey)
    }
  }, [showFilterMenu])

  // Load Google Maps script (and notify when ready)
  useEffect(() => {
    const apiKey = process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY
    if (!apiKey) return
    // Only mark ready when the Map constructor exists
    if (window.google?.maps?.Map) {
      setGmapsReady(true)
      return
    }
    if (!window.__gmapsInitCallbacks) window.__gmapsInitCallbacks = []
    // ensure our callback runs when the script signals ready
    window.__gmapsInitCallbacks.push(() => setGmapsReady(true))
    if (!window.__gmapsLoading) {
      window.__gmapsLoading = true
      const script = document.createElement("script")
      // Use callback to ensure the core 'maps' library (with google.maps.Map) is fully available
      script.src = `https://maps.googleapis.com/maps/api/js?key=${apiKey}&libraries=geometry,marker&v=weekly&callback=initMap`
      script.async = true
      const init = () => {
        const ensureReady = () => {
          if (window.google?.maps?.Map) {
            // Mark as ready and flush callbacks
            setGmapsReady(true)
            window.__gmapsInitCallbacks?.forEach((cb) => cb())
            window.__gmapsInitCallbacks = []
          } else {
            // Retry shortly until constructor is available
            setTimeout(ensureReady, 50)
          }
        }
        ensureReady()
      }
      ;(window as any).initMap = init
      script.addEventListener("load", init)
      document.body.appendChild(script)
    }
  }, [])

  // Listen for hover events from chat to focus a POI on the map
  useEffect(() => {
    function onPoiHover(e: any) {
      if (!mapInstance.current || !window.google?.maps) return
      const poiId = e?.detail?.poiId as string | undefined
      if (!poiId) return
      const marker = markersByIdRef.current[poiId]
      if (!marker) return
      const pos = marker.getPosition()
      mapInstance.current!.panTo(pos)
      if (autoZoomOnHover) {
        const z = mapInstance.current.getZoom?.() ?? 2
        if (z < 13) mapInstance.current.setZoom(13)
      }
      if (!infoWindowRef.current) infoWindowRef.current = new window.google.maps.InfoWindow()
      infoWindowRef.current.setContent(`<div style="font-weight:600">${marker.getTitle?.() || "Place"}</div>`)
      infoWindowRef.current.open({ map: mapInstance.current!, anchor: marker })
    }
    window.addEventListener("poi-hover", onPoiHover as any)
    return () => window.removeEventListener("poi-hover", onPoiHover as any)
  }, [autoZoomOnHover])

  // Initialize the map once Google Maps is ready
  useEffect(() => {
    if (!gmapsReady || !mapRef.current || mapInstance.current) return
    if (!window.google?.maps?.Map) return

    const mapId = process.env.NEXT_PUBLIC_GOOGLE_MAPS_MAP_ID
    mapInstance.current = new window.google.maps.Map(mapRef.current, {
      center: { lat: 20, lng: 0 },
      zoom: 2,
      styles: CUSTOM_MAP_STYLE,
      mapTypeControl: false,
      streetViewControl: false,
      fullscreenControl: false,
      ...(mapId ? { mapId } : {}),
    })

    // Observe container resizes and nudge map
    // Create a single OverlayView instance to get projection.
    overlayRef.current = new window.google.maps.OverlayView()
    overlayRef.current.onAdd = function () {}
    overlayRef.current.draw = function () {}
    overlayRef.current.onRemove = function () {}
    overlayRef.current.setMap(mapInstance.current)

    // Add map click listener to close persistent info windows
    mapInstance.current.addListener('click', () => {
      if (persistentInfoWindowRef.current) {
        persistentInfoWindowRef.current.close()
        persistentInfoWindowRef.current = null
      }
    })

    if ((window as any).ResizeObserver) {
      const resizeObserver = new ResizeObserver(() => {
        window.google.maps.event.trigger(mapInstance.current, 'resize')
      })
      resizeObserver.observe(mapRef.current)
      return () => {
        resizeObserver.disconnect()
        overlayRef.current?.setMap(null) // Clean up overlay
      }
    }
  }, [gmapsReady])

  // Handle map resize when visibility changes
  useEffect(() => {
    if (isVisible && mapInstance.current && window.google?.maps) {
      // Trigger resize after a small delay to ensure the container is visible
      setTimeout(() => {
        window.google.maps.event.trigger(mapInstance.current, 'resize')
      }, 50)
    }
  }, [isVisible])

  // Render POI markers
  useEffect(() => {
    if (!mapInstance.current || !window.google?.maps) return

    // Add a small delay to ensure map is fully ready after tab switch
    const timeoutId = setTimeout(() => {
      // Clear existing markers and info windows
      markersRef.current.forEach((m) => m.setMap(null))
      markersRef.current = []
      markersByIdRef.current = {}
      
      // Close any open info windows when re-rendering markers
      if (activeInfoWindowRef.current) {
        activeInfoWindowRef.current.close()
        activeInfoWindowRef.current = null
      }
      if (persistentInfoWindowRef.current) {
        persistentInfoWindowRef.current.close()
        persistentInfoWindowRef.current = null
      }

      filteredPois.forEach((poi) => {
      if (!poi.lat || !poi.lng) return

      const isSaved = savedIds?.has(poi.id)
      const isAdded = itineraryPoiIds?.has(poi.id)

      const itineraryPois = itinerary?.daysPlan.flatMap(d => d.activities).map(a => a.poiId) || []
      const poiIndex = isAdded ? itineraryPois.indexOf(poi.id) : -1

      const marker = new window.google.maps.Marker({
        position: { lat: poi.lat, lng: poi.lng },
        map: mapInstance.current!,
        title: poi.name,
        icon: isAdded
          ? {
              url: `data:image/svg+xml;utf-8,${encodeURIComponent(
                `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><circle cx="12" cy="12" r="11" fill="black" stroke="white" stroke-width="2"/><text x="12" y="16" font-size="12" fill="white" text-anchor="middle" font-family="Arial, sans-serif" font-weight="bold">${poiIndex + 1}</text></svg>`
              )}`,
              scaledSize: new window.google.maps.Size(24, 24),
              anchor: new window.google.maps.Point(12, 12),
            }
          : {
              path: window.google.maps.SymbolPath.CIRCLE,
              scale: 6,
              fillColor: isSaved ? '#f9a8d4' : '#60a5fa',
              fillOpacity: 1,
              strokeColor: isSaved ? '#f472b6' : '#2563eb',
              strokeWeight: 2,
            },
      })

      const infoWindow = new window.google.maps.InfoWindow({
        content: `<div id="info-window-content-${poi.id}"></div>`,
        pixelOffset: new window.google.maps.Size(0, -15),
        disableAutoPan: true,
      })

      const showInfoWindow = (e: google.maps.MapMouseEvent, isPersistent = false) => {
        if (!mapInstance.current || !overlayRef.current) return

        const projection = overlayRef.current.getProjection()
        if (!projection || !e.latLng) return

        const pixel = projection.fromLatLngToContainerPixel(e.latLng)
        const mapDiv = mapInstance.current.getDiv()
        const mapWidth = mapDiv.clientWidth
        const mapHeight = mapDiv.clientHeight

        const cardWidth = 240
        const cardHeight = 180 // Compact card height
        const markerSize = 12
        const padding = 10

        let newPixelOffset

        // Check if card fits above the marker
        const fitsAbove = pixel.y - cardHeight - markerSize - padding > 0
        // Check if card fits below the marker  
        const fitsBelow = pixel.y + cardHeight + markerSize + padding < mapHeight
        // Check if card fits to the right
        const fitsRight = pixel.x + cardWidth + markerSize + padding < mapWidth
        // Check if card fits to the left
        const fitsLeft = pixel.x - cardWidth - markerSize - padding > 0

        if (fitsAbove) {
          // Show above (preferred)
          newPixelOffset = new window.google.maps.Size(0, -markerSize - 5)
        } else if (fitsBelow) {
          // Show below
          newPixelOffset = new window.google.maps.Size(0, markerSize + 5)
        } else if (fitsRight) {
          // Show to the right
          newPixelOffset = new window.google.maps.Size(markerSize + 5, -cardHeight / 2)
        } else if (fitsLeft) {
          // Show to the left
          newPixelOffset = new window.google.maps.Size(-markerSize - 5, -cardHeight / 2)
        } else {
          // Default fallback - show above even if it clips
          newPixelOffset = new window.google.maps.Size(0, -markerSize - 5)
        }
        
        infoWindow.setOptions({ pixelOffset: newPixelOffset });

        // Close any existing info windows
        if (activeInfoWindowRef.current && activeInfoWindowRef.current !== persistentInfoWindowRef.current) {
          activeInfoWindowRef.current.close()
        }
        if (persistentInfoWindowRef.current && persistentInfoWindowRef.current !== infoWindow) {
          persistentInfoWindowRef.current.close()
          persistentInfoWindowRef.current = null
        }

        infoWindow.addListener('domready', () => {
          const content = document.getElementById(`info-window-content-${poi.id}`)
          if (content) {
            // Apply styles to remove InfoWindow chrome after DOM is ready
            const iwContainer = content.closest('.gm-style-iw-c') as HTMLElement
            const iwTitle = content.closest('.gm-style-iw')?.querySelector('.gm-style-iw-t') as HTMLElement
            const iwCloseBtn = content.closest('.gm-style-iw')?.querySelector('.gm-ui-hover-effect') as HTMLElement
            
            if (iwContainer) {
              iwContainer.style.padding = '0'
              iwContainer.style.border = 'none'
              iwContainer.style.borderRadius = '0'
              iwContainer.style.boxShadow = 'none'
              iwContainer.style.background = 'transparent'
              iwContainer.style.maxWidth = '240px'
              iwContainer.style.outline = 'none'
            }
            
            if (iwTitle) iwTitle.style.display = 'none'
            if (iwCloseBtn) iwCloseBtn.style.display = 'none'
            
            const iwContent = content.closest('.gm-style-iw-d') as HTMLElement
            if (iwContent) {
              iwContent.style.padding = '0'
              iwContent.style.margin = '0'
              iwContent.style.border = 'none'
              iwContent.style.background = 'transparent'
              iwContent.style.overflow = 'visible'
              iwContent.style.outline = 'none'
            }

            // Remove any parent container borders
            const iwWrapper = content.closest('.gm-style-iw') as HTMLElement
            if (iwWrapper) {
              iwWrapper.style.border = 'none'
              iwWrapper.style.outline = 'none'
              iwWrapper.style.boxShadow = 'none'
            }
            
            const root = createRoot(content)
            root.render(
              <CompactPoiCard
                poi={poi}
                isSaved={isSaved || false}
                isItineraryItem={isAdded || false}
                onToggleSave={onToggleSave ?? (() => {})}
                onAddPoi={onAddPoi ?? (() => {})}
                onReplan={onReplan ?? (() => {})}
              />
            )
            
            // Add mouse listener only for non-persistent info windows
            if (!isPersistent) {
              const iwWrapperEl = content.closest('.gm-style-iw-wrapper');
              if (iwWrapperEl) {
                let hoverTimeout: NodeJS.Timeout | null = null;
                
                iwWrapperEl.addEventListener('mouseenter', () => {
                  if (hoverTimeout) {
                    clearTimeout(hoverTimeout);
                    hoverTimeout = null;
                  }
                });
                
                iwWrapperEl.addEventListener('mouseleave', () => {
                  hoverTimeout = setTimeout(() => {
                    if (activeInfoWindowRef.current && activeInfoWindowRef.current !== persistentInfoWindowRef.current) {
                      activeInfoWindowRef.current.close();
                    }
                  }, 200);
                });
              }
            }
          }
        })

        infoWindow.open(mapInstance.current, marker)
        activeInfoWindowRef.current = infoWindow
        
        if (isPersistent) {
          persistentInfoWindowRef.current = infoWindow
        }
      }

      marker.addListener("mouseover", (e: google.maps.MapMouseEvent) => {
        // Don't show hover card if there's already a persistent one
        if (persistentInfoWindowRef.current) return
        showInfoWindow(e, false)
      })

      marker.addListener("click", (e: google.maps.MapMouseEvent) => {
        showInfoWindow(e, true)
      })

      marker.addListener("mouseout", () => {
        // Only close hover cards, not persistent ones
        setTimeout(() => {
          if (activeInfoWindowRef.current && activeInfoWindowRef.current !== persistentInfoWindowRef.current) {
            const iwWrapper = document.querySelector('.gm-style-iw-wrapper');
            
            // Only close if not hovering over the card
            if (!iwWrapper || !iwWrapper.matches(':hover')) {
              activeInfoWindowRef.current.close();
            }
          }
        }, 200);
      })

      markersRef.current.push(marker)
      markersByIdRef.current[poi.id] = marker
    })
    }, 100) // Small delay to ensure map is ready

    return () => clearTimeout(timeoutId)
  }, [filteredPois, gmapsReady, savedIds, itineraryPoiIds, onToggleSave, onAddPoi, onReplan])

  // Auto-zoom to fit markers and routes
  useEffect(() => {
    if (!mapInstance.current || !window.google?.maps) return

    const bounds = new window.google.maps.LatLngBounds()
    let hasContent = false

    // Include markers in bounds
    markersRef.current.forEach((marker) => {
      bounds.extend(marker.getPosition()!)
      hasContent = true
    })

    // Include routes in bounds
    polylinesRef.current.forEach((renderer) => {
      const directions = renderer.getDirections()
      if (directions && directions.routes && directions.routes.length > 0) {
        const route = directions.routes[0]
        if (route.overview_path) {
          route.overview_path.forEach((latLng: any) => {
            bounds.extend(latLng)
            hasContent = true
          })
        }
      }
    })

    if (hasContent) {
      mapInstance.current.fitBounds(bounds, 50) // 50px padding
    }
  }, [filteredPois, mapData?.routes])

  // Sync polylines with routes
  useEffect(() => {
    if (!mapInstance.current || !window.google?.maps || !itinerary) return

    // Clear old polylines
    polylinesRef.current.forEach((pl) => pl.setMap(null))
    polylinesRef.current = []

    const directionsService = new window.google.maps.DirectionsService()
    const directionsRenderer = new window.google.maps.DirectionsRenderer({
      map: mapInstance.current,
      suppressMarkers: true, // We use our own markers
      polylineOptions: {
        strokeColor: "#0a66ff",
        strokeOpacity: 0.9,
        strokeWeight: 4,
      },
    })

    polylinesRef.current.push(directionsRenderer) // Store renderer to clear it later

    const allPois = mapData?.pois || []
    const waypoints = itinerary.daysPlan
      .flatMap((day) => day.activities)
      .map((activity) => {
        const poi = allPois.find((p) => p.id === activity.poiId)
        if (!poi || !poi.lat || !poi.lng) return null
        return { location: { lat: poi.lat, lng: poi.lng }, stopover: true }
      })
      .filter(Boolean) as google.maps.DirectionsWaypoint[]

    if (waypoints.length < 2) return

    const origin = waypoints.shift()!.location
    const destination = waypoints.pop()!.location

    directionsService.route(
      {
        origin,
        destination,
        waypoints,
        travelMode: window.google.maps.TravelMode.DRIVING,
      },
      (response: google.maps.DirectionsResult, status: google.maps.DirectionsStatus) => {
        if (status === "OK") {
          directionsRenderer.setDirections(response)
        } else {
          console.warn("Directions request failed due to " + status)
        }
      }
    )
  }, [itinerary, mapData?.pois, gmapsReady])


  return (
    <div className="relative h-full overflow-hidden">
      <div ref={mapRef} className="w-full h-full" />

      {/* Map Controls - bottom right */}
      <div className="absolute bottom-4 right-4 flex flex-col gap-2 z-20">
        <Button
          size="sm"
          variant="secondary"
          className="w-10 h-10 p-0 bg-white shadow-md hover:bg-gray-50"
          onClick={() => {
            if (!mapInstance.current) return
            const z = mapInstance.current.getZoom?.() ?? 2
            mapInstance.current.setZoom(Math.min(20, z + 1))
          }}
        >
          <Plus className="w-4 h-4" />
        </Button>
        <Button
          size="sm"
          variant="secondary"
          className="w-10 h-10 p-0 bg-white shadow-md hover:bg-gray-50"
          onClick={() => {
            if (!mapInstance.current) return
            const z = mapInstance.current.getZoom?.() ?? 2
            mapInstance.current.setZoom(Math.max(2, z - 1))
          }}
        >
          <Minus className="w-4 h-4" />
        </Button>
        <Button
          size="sm"
          variant="secondary"
          className="w-10 h-10 p-0 bg-white shadow-md hover:bg-gray-50"
          onClick={() => {
            if (!mapInstance.current) return
            const centerTo = userMarkerRef.current?.getPosition?.()
            if (centerTo) {
              mapInstance.current.panTo(centerTo)
              mapInstance.current.setZoom(12)
            } else if (navigator.geolocation) {
              navigator.geolocation.getCurrentPosition((pos) => {
                const loc = { lat: pos.coords.latitude, lng: pos.coords.longitude }
                mapInstance.current!.panTo(loc)
                mapInstance.current!.setZoom(12)
              })
            }
          }}
        >
          <Navigation className="w-4 h-4" />
        </Button>
      </div>

      {/* Filter toggle button + Pop menu */}
      <div className="absolute top-16 right-4 z-20 flex flex-col items-end gap-2" ref={filterMenuRef}>
        <Button
          size="sm"
          variant={showFilterMenu ? "default" : "secondary"}
          className={`w-auto h-10 px-3 bg-white shadow-md hover:bg-gray-50 rounded-full ${showFilterMenu ? "bg-black/80 text-white hover:bg-black/90" : ""}`}
          onClick={() => setShowFilterMenu((v) => !v)}
          title="Map filters"
        >
          <SlidersHorizontal className="w-4 h-4 mr-2" /> Filters
        </Button>

        {showFilterMenu && (
          <div ref={filterMenuRef} className="w-60 bg-white/95 backdrop-blur-md border border-white/30 rounded-2xl shadow-xl p-2 space-y-1">
          <div
            className="flex items-center gap-2 px-2 py-2 rounded-lg hover:bg-gray-50 cursor-pointer"
            onClick={() => {
              const next = !filterAll
              setFilterAll(next)
              if (next) setFilterTypes({ stay: true, restaurant: true, attraction: true })
            }}
          >
            <div className={`w-4 h-4 rounded-full border ${filterAll ? "bg-black" : "bg-white"}`} />
            <span className="text-sm flex-1">All</span>
          </div>

          <div className="h-px bg-gray-100 my-1" />

          {([
            { key: "stay", label: "Stays" },
            { key: "restaurant", label: "Restaurants" },
            { key: "attraction", label: "Attractions" },
          ] as const).map((t) => (
            <div
              key={t.key}
              className="flex items-center gap-2 px-2 py-2 rounded-lg hover:bg-gray-50 cursor-pointer"
              onClick={() => {
                setFilterAll(false)
                setFilterTypes((prev) => ({ ...prev, [t.key]: !prev[t.key] }))
              }}
            >
              <input type="checkbox" checked={filterAll ? true : filterTypes[t.key]} readOnly className="accent-black" />
              <span className="text-sm flex-1">{t.label} <span className="text-gray-400">{` ${pois.filter(p=>p.type===t.key).length}`}</span></span>
            </div>
          ))}

          <div className="h-px bg-gray-100 my-1" />

          <div
            className="flex items-center gap-2 px-2 py-2 rounded-lg hover:bg-gray-50 cursor-pointer"
            onClick={() => setAddedOnly((v) => !v)}
          >
            <input type="checkbox" checked={addedOnly} readOnly className="accent-black" />
            <span className="text-sm flex-1">Added only</span>
          </div>
          <div
            className="flex items-center gap-2 px-2 py-2 rounded-lg hover:bg-gray-50 cursor-pointer"
            onClick={() => setSavedOnly((v) => !v)}
          >
            <input type="checkbox" checked={savedOnly} readOnly className="accent-black" />
            <span className="text-sm flex-1">Saved only {savedIds ? <span className="text-gray-400">{` ${savedIds.size}`}</span> : null}</span>
          </div>

          <div className="h-px bg-gray-100 my-1" />

          <div
            className="flex items-center gap-2 px-2 py-2 rounded-lg hover:bg-gray-50 cursor-pointer"
            onClick={() => setShowPrices((v) => !v)}
          >
            <input type="checkbox" checked={showPrices} readOnly className="accent-black" />
            <span className="text-sm flex-1">Show prices</span>
          </div>
          </div>
        )}
      </div>

      {/* Google Maps Attribution */}
      <div className="absolute bottom-2 right-2 text-xs text-gray-500 bg-white/80 px-2 py-1 rounded">
        Map Data ©2025 Google
      </div>

    </div>
  )
}
