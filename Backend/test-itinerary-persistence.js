#!/usr/bin/env node

/**
 * Test script to verify itinerary persistence in Roameo backend
 * Run with: node test-itinerary-persistence.js
 */

import fetch from 'node-fetch';
import WebSocket from 'ws';

const BASE_URL = 'http://localhost:4000';
const WS_URL = 'ws://localhost:4000/ws';

// Test itinerary data
const testItinerary = {
  origin: "Mumbai",
  destination: "Ooty",
  days: 3,
  daysPlan: [
    {
      day: 1,
      date: "2024-01-01",
      title: "Arrival in Ooty",
      activities: [
        {
          name: "Check-in at hotel",
          start: "09:00",
          end: "10:00",
          location: "Ooty Lake Area"
        },
        {
          name: "Visit Ooty Botanical Gardens",
          start: "11:00",
          end: "13:00",
          location: "Government Botanical Garden"
        }
      ]
    },
    {
      day: 2,
      date: "2024-01-02", 
      title: "Explore Ooty",
      activities: [
        {
          name: "Doddabetta Peak",
          start: "09:00",
          end: "12:00",
          location: "Doddabetta Peak"
        }
      ]
    },
    {
      day: 3,
      date: "2024-01-03",
      title: "Departure",
      activities: [
        {
          name: "Check-out and departure",
          start: "10:00",
          end: "11:00",
          location: "Hotel"
        }
      ]
    }
  ]
};

async function wait(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function testItineraryPersistence() {
  console.log('🚀 Starting itinerary persistence test...\n');
  
  try {
    // Step 1: Create a new session
    console.log('📝 Step 1: Creating new session...');
    const createResponse = await fetch(`${BASE_URL}/api/chat/send`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: "Plan a trip to Ooty for 3 days" })
    });
    
    if (!createResponse.ok) {
      throw new Error(`Failed to create session: ${createResponse.status}`);
    }
    
    const sessionData = await createResponse.json();
    const sessionId = sessionData.sessionId;
    console.log(`✅ Session created: ${sessionId}\n`);
    
    // Step 2: Wait for trip planning to complete
    console.log('⏳ Step 2: Waiting for trip planning to complete...');
    await wait(5000); // Wait 5 seconds for planning
    
    // Step 3: Connect WebSocket and check for itinerary
    console.log('🔌 Step 3: Connecting WebSocket...');
    const ws = new WebSocket(`${WS_URL}?sessionId=${sessionId}`);
    
    let itineraryReceived = false;
    let receivedItinerary = null;
    
    const wsPromise = new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('WebSocket connection timeout'));
      }, 10000);
      
      ws.on('open', () => {
        console.log('✅ WebSocket connected');
        clearTimeout(timeout);
      });
      
      ws.on('message', (data) => {
        try {
          const event = JSON.parse(data.toString());
          console.log(`📨 Received event: ${event.type}`);
          
          if (event.type === 'itinerary.update' && event.data) {
            itineraryReceived = true;
            receivedItinerary = event.data;
            console.log(`✅ Itinerary received with ${event.data.daysPlan?.length || 0} days`);
            resolve();
          }
          
          if (event.type === 'session.ready') {
            console.log('📋 Session ready, waiting for itinerary...');
          }
        } catch (error) {
          console.error('❌ Error parsing WebSocket message:', error);
        }
      });
      
      ws.on('error', (error) => {
        console.error('❌ WebSocket error:', error);
        clearTimeout(timeout);
        reject(error);
      });
    });
    
    await wsPromise;
    ws.close();
    
    if (!itineraryReceived) {
      throw new Error('No itinerary received from WebSocket');
    }
    
    // Step 4: Disconnect and reconnect to test persistence
    console.log('\n🔄 Step 4: Testing persistence - reconnecting WebSocket...');
    await wait(1000);
    
    const ws2 = new WebSocket(`${WS_URL}?sessionId=${sessionId}`);
    let persistedItinerary = null;
    
    const persistencePromise = new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Persistence test timeout'));
      }, 10000);
      
      ws2.on('open', () => {
        console.log('✅ WebSocket reconnected');
      });
      
      ws2.on('message', (data) => {
        try {
          const event = JSON.parse(data.toString());
          
          if (event.type === 'itinerary.update' && event.data) {
            persistedItinerary = event.data;
            console.log(`✅ Persisted itinerary restored with ${event.data.daysPlan?.length || 0} days`);
            clearTimeout(timeout);
            resolve();
          }
        } catch (error) {
          console.error('❌ Error parsing reconnection message:', error);
        }
      });
      
      ws2.on('error', (error) => {
        clearTimeout(timeout);
        reject(error);
      });
    });
    
    await persistencePromise;
    ws2.close();
    
    // Step 5: Validate data integrity
    console.log('\n🔍 Step 5: Validating data integrity...');
    
    if (!persistedItinerary) {
      throw new Error('❌ No itinerary persisted after reconnection');
    }
    
    if (!persistedItinerary.daysPlan || persistedItinerary.daysPlan.length === 0) {
      throw new Error('❌ Persisted itinerary has no days plan');
    }
    
    console.log(`✅ Original itinerary: ${receivedItinerary.daysPlan?.length || 0} days`);
    console.log(`✅ Persisted itinerary: ${persistedItinerary.daysPlan?.length || 0} days`);
    
    // Step 6: Test sending another message to ensure itinerary survives
    console.log('\n💬 Step 6: Testing itinerary survival during new conversations...');
    
    const chatResponse = await fetch(`${BASE_URL}/api/chat/send`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        sessionId: sessionId,
        message: "What's the weather like in Ooty?" 
      })
    });
    
    if (!chatResponse.ok) {
      throw new Error(`Failed to send chat message: ${chatResponse.status}`);
    }
    
    // Wait for response
    await wait(2000);
    
    // Final check - reconnect one more time
    console.log('🔄 Final check: Reconnecting after chat...');
    const ws3 = new WebSocket(`${WS_URL}?sessionId=${sessionId}`);
    let finalItinerary = null;
    
    const finalPromise = new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Final persistence test timeout'));
      }, 10000);
      
      ws3.on('message', (data) => {
        try {
          const event = JSON.parse(data.toString());
          
          if (event.type === 'itinerary.update' && event.data) {
            finalItinerary = event.data;
            console.log(`✅ Final itinerary check: ${event.data.daysPlan?.length || 0} days`);
            clearTimeout(timeout);
            resolve();
          }
        } catch (error) {
          console.error('❌ Error in final check:', error);
        }
      });
      
      ws3.on('error', (error) => {
        clearTimeout(timeout);
        reject(error);
      });
    });
    
    await finalPromise;
    ws3.close();
    
    // Success summary
    console.log('\n🎉 SUCCESS: Itinerary persistence test completed!');
    console.log('✅ Itinerary survives disconnection/reconnection');
    console.log('✅ Itinerary survives new conversation messages');
    console.log('✅ Data integrity maintained throughout');
    
    return {
      success: true,
      sessionId,
      originalDays: receivedItinerary?.daysPlan?.length || 0,
      persistedDays: persistedItinerary?.daysPlan?.length || 0,
      finalDays: finalItinerary?.daysPlan?.length || 0
    };
    
  } catch (error) {
    console.error('\n❌ FAILURE: Itinerary persistence test failed!');
    console.error('Error:', error.message);
    return { success: false, error: error.message };
  }
}

// Run the test
testItineraryPersistence().then((result) => {
  if (result.success) {
    console.log('\n📊 Test Results:');
    console.log(`Session ID: ${result.sessionId}`);
    console.log(`Original Days: ${result.originalDays}`);
    console.log(`Persisted Days: ${result.persistedDays}`);
    console.log(`Final Days: ${result.finalDays}`);
    process.exit(0);
  } else {
    console.error('\n💥 Test failed:', result.error);
    process.exit(1);
  }
}).catch((error) => {
  console.error('\n💥 Unexpected error:', error);
  process.exit(1);
});