import { NextRequest, NextResponse } from 'next/server'

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const photo_reference = searchParams.get('photo_reference')
  const maxwidth = searchParams.get('maxwidth') || '400'
  const key = searchParams.get('key')
  
  if (!photo_reference || !key) {
    return NextResponse.json(
      { error: 'photo_reference and key required' }, 
      { status: 400 }
    )
  }

  try {
    const photoUrl = `https://maps.googleapis.com/maps/api/place/photo?photo_reference=${photo_reference}&maxwidth=${maxwidth}&key=${key}`
    
    // Add timeout and abort controller for fetch
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 10000) // 10 second timeout
    
    const response = await fetch(photoUrl, {
      signal: controller.signal,
      headers: {
        'User-Agent': 'Roameo/1.0'
      }
    })
    
    clearTimeout(timeoutId)
    
    if (!response.ok) {
      console.error(`Photo proxy error: ${response.status} ${response.statusText}`)
      return NextResponse.json(
        { error: `Photo service error: ${response.status}` }, 
        { status: response.status }
      )
    }

    // Stream the image response
    const imageData = await response.arrayBuffer()
    const contentType = response.headers.get('content-type') || 'image/jpeg'
    
    return new NextResponse(imageData, {
      headers: {
        'Content-Type': contentType,
        'Cache-Control': 'public, max-age=3600',
        'Access-Control-Allow-Origin': '*'
      }
    })
  } catch (error: any) {
    console.error('[proxy] Error fetching photo:', error)
    
    // Handle timeout errors specifically
    if (error?.name === 'AbortError') {
      return NextResponse.json(
        { error: 'Photo fetch timeout' }, 
        { status: 408 }
      )
    }
    
    // Handle network errors
    if (error?.code === 'ETIMEDOUT' || error?.code === 'ECONNREFUSED') {
      return NextResponse.json(
        { error: 'Photo service unavailable' }, 
        { status: 503 }
      )
    }
    
    return NextResponse.json(
      { error: 'Internal server error' }, 
      { status: 500 }
    )
  }
}