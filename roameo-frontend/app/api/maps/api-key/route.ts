import { NextRequest, NextResponse } from 'next/server'

export async function GET(request: NextRequest) {
  const apiKey = process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY
  
  if (!apiKey) {
    return NextResponse.json(
      { error: 'Google Maps API key not configured' }, 
      { status: 500 }
    )
  }
  
  // Validate API key format (Google Maps API keys are 39 characters and start with AIza)
  if (apiKey.length !== 39 || !apiKey.startsWith('AIza')) {
    console.error('Invalid Google Maps API key format:', apiKey.substring(0, 10) + '...')
    return NextResponse.json(
      { error: 'Invalid Google Maps API key format' }, 
      { status: 500 }
    )
  }
  
  return NextResponse.json({ apiKey })
}