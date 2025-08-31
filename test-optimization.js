#!/usr/bin/env node

/**
 * Test script to verify the optimization changes:
 * 1. Destination agent is more selective
 * 2. Parallel processing works correctly
 * 3. Chat responses remain visible after planning
 */

const fetch = (...args) => import('node-fetch').then(({default: fetch}) => fetch(...args));

const BASE_URL = 'http://localhost:4000';

async function wait(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function testMessage(message) {
  console.log(`\n🧪 Testing: "${message}"`);
  
  try {
    const response = await fetch(`${BASE_URL}/api/chat/send`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const result = await response.json();
    
    console.log(`📍 Session: ${result.sessionId}`);
    console.log(`🎯 Events: ${result.events?.length || 0}`);
    
    // Analyze events to understand agent behavior
    if (result.events) {
      const intentEvents = result.events.filter(e => e.type === 'intent.detected');
      const chatEvents = result.events.filter(e => e.type === 'chat.append');
      const planningEvents = result.events.filter(e => e.type === 'planning.status');
      const itineraryEvents = result.events.filter(e => e.type === 'itinerary.update');
      
      if (intentEvents.length > 0) {
        console.log(`🎯 Intent detected: ${intentEvents[0].data.intent}`);
      }
      
      if (chatEvents.length > 0) {
        console.log(`💬 Chat responses: ${chatEvents.length}`);
        chatEvents.forEach((event, i) => {
          const preview = event.data.content.substring(0, 100).replace(/\n/g, ' ');
          console.log(`   ${i + 1}. ${preview}...`);
        });
      }
      
      if (planningEvents.length > 0) {
        console.log(`⏳ Planning status: ${planningEvents[0].data.status}`);
      }
      
      if (itineraryEvents.length > 0) {
        console.log(`📅 Itinerary created: ${itineraryEvents[0].data?.daysPlan?.length || 0} days`);
      }
    }
    
    return result;
  } catch (error) {
    console.error(`❌ Error testing "${message}":`, error.message);
    return null;
  }
}

async function runTests() {
  console.log('🚀 Testing Destination Agent Optimization\n');
  
  // Test 1: Vague statements should be CHAT (not trigger destination agent)
  console.log('='.repeat(60));
  console.log('TEST 1: Vague statements should NOT trigger destination agent');
  console.log('='.repeat(60));
  
  await testMessage("I want to visit kerala");
  await wait(1000);
  await testMessage("Tell me about goa");
  await wait(1000);
  await testMessage("Kerala looks nice");
  await wait(1000);
  
  // Test 2: Specific destination searches should work
  console.log('\n' + '='.repeat(60));
  console.log('TEST 2: Specific searches should trigger destination agent');
  console.log('='.repeat(60));
  
  await testMessage("show me places in ooty");
  await wait(1000);
  await testMessage("what are the top attractions in delhi");
  await wait(1000);
  await testMessage("find hotels in mumbai");
  await wait(1000);
  
  // Test 3: Trip planning should work and show responses
  console.log('\n' + '='.repeat(60));
  console.log('TEST 3: Trip planning should work with visible responses');
  console.log('='.repeat(60));
  
  await testMessage("plan a 3 day trip to coonoor");
  await wait(3000); // Give time for planning to complete
  
  // Test 4: General chat after planning
  console.log('\n' + '='.repeat(60));
  console.log('TEST 4: General chat should work after planning');
  console.log('='.repeat(60));
  
  await testMessage("What's the weather like?");
  await wait(1000);
  await testMessage("Thanks for the help");
  await wait(1000);
  
  console.log('\n✅ All tests completed!');
  console.log('\nExpected behavior:');
  console.log('- Test 1: Should show CHAT intent, no destination search');
  console.log('- Test 2: Should show DESTINATION_SEARCH intent with POI results');
  console.log('- Test 3: Should show PLAN_TRIP intent with itinerary and multiple chat responses');
  console.log('- Test 4: Should show CHAT intent with immediate responses');
}

runTests().catch(console.error);