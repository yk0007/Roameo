#!/usr/bin/env node

/**
 * Test script for immediate POI search functionality
 * Tests that POI search happens immediately when destinations are extracted
 */

import fetch from 'node-fetch';

const API_BASE = 'http://localhost:4000';

async function testDestinationSearch(message, expectedIntent = 'DESTINATION_SEARCH') {
  console.log(`\n🧪 Testing: "${message}"`);
  console.log(`Expected intent: ${expectedIntent}`);
  
  try {
    const response = await fetch(`${API_BASE}/api/chat/send`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        sessionId: 'test-session-' + Date.now(),
        message: message,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const result = await response.json();
    console.log('✅ Response received');
    
    // Check if we got events back
    if (result.events && Array.isArray(result.events)) {
      console.log(`📊 Received ${result.events.length} events`);
      
      // Look for POI search results
      const poiEvent = result.events.find(event => event.type === 'search.results');
      const mapEvent = result.events.find(event => event.type === 'map.update');
      const chatEvent = result.events.find(event => event.type === 'chat.append');
      const navbarEvent = result.events.find(event => event.type === 'navbar.update');
      
      if (poiEvent) {
        console.log('🎯 POI search event found!');
        const pois = [...(poiEvent.data.stays || []), ...(poiEvent.data.restaurants || []), ...(poiEvent.data.attractions || [])];
        console.log(`   Found ${pois.length} POIs total`);
        console.log(`   - Stays: ${poiEvent.data.stays?.length || 0}`);
        console.log(`   - Restaurants: ${poiEvent.data.restaurants?.length || 0}`);
        console.log(`   - Attractions: ${poiEvent.data.attractions?.length || 0}`);
      } else {
        console.log('❌ No POI search event found');
      }
      
      if (mapEvent) {
        console.log('🗺️  Map update event found!');
      }
      
      if (chatEvent) {
        console.log('💬 Chat response:', chatEvent.data.content.substring(0, 100) + '...');
      }
      
      if (navbarEvent) {
        console.log('🧭 Navbar update:', navbarEvent.data);
      }
    } else {
      console.log('❌ No events received');
    }
    
  } catch (error) {
    console.error('❌ Test failed:', error.message);
  }
}

async function runTests() {
  console.log('🚀 Testing Immediate POI Search Functionality');
  console.log('==================================================');
  
  // Test cases for destination search (should trigger immediate POI search)
  await testDestinationSearch('show me places in ooty');
  await testDestinationSearch('what to do in coonoor');
  await testDestinationSearch('ooty attractions');
  await testDestinationSearch('find hotels in mumbai');
  await testDestinationSearch('restaurants in goa');
  await testDestinationSearch('I want to visit kerala');
  
  // Test case for trip planning (should not use immediate POI search)
  await testDestinationSearch('plan a 3 day trip to ooty', 'PLAN_TRIP');
  
  // Test case for general chat (should not trigger POI search)
  await testDestinationSearch('what is your name?', 'CHAT');
  
  console.log('\n✨ Testing complete!');
}

runTests().catch(console.error);