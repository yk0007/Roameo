#!/usr/bin/env node

/**
 * Test script to demonstrate enhanced AI-powered chat responses
 * with POI hover functionality and bold formatting
 */

import fetch from 'node-fetch';

const API_BASE = 'http://localhost:4000';

async function testEnhancedResponse(message) {
  console.log(`\n🧪 Testing: "${message}"`);
  console.log('='.repeat(60));
  
  try {
    const response = await fetch(`${API_BASE}/api/chat/send`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        sessionId: 'test-enhanced-' + Date.now(),
        message: message,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const result = await response.json();
    
    // Find the chat response
    const chatEvent = result.events?.find(event => event.type === 'chat.append');
    if (chatEvent) {
      console.log('💬 Enhanced AI Response:');
      console.log(chatEvent.data.content);
      
      // Check for formatting features
      const content = chatEvent.data.content;
      const hasBoldFormatting = content.includes('**');
      const hasHoverCards = content.includes('data-poi-hover=');
      const boldCount = (content.match(/\*\*/g) || []).length / 2;
      const hoverCount = (content.match(/data-poi-hover=/g) || []).length;
      
      console.log('\n✨ Features Detected:');
      console.log(`   📝 Bold formatting: ${hasBoldFormatting ? '✅' : '❌'} (${boldCount} bold items)`);
      console.log(`   🎯 Hover cards: ${hasHoverCards ? '✅' : '❌'} (${hoverCount} hover POIs)`);
      
      // Extract POI names with hover functionality
      const hoverMatches = content.match(/data-poi-hover="([^"]+)"/g);
      if (hoverMatches) {
        console.log('   🏨 POIs with hover:');
        hoverMatches.forEach(match => {
          const poiName = match.match(/data-poi-hover="([^"]+)"/)[1];
          console.log(`      • ${poiName}`);
        });
      }
      
    } else {
      console.log('❌ No chat response found');
    }
    
  } catch (error) {
    console.error('❌ Test failed:', error.message);
  }
}

async function runTests() {
  console.log('🚀 Testing Enhanced AI-Powered Chat Responses');
  console.log('🎯 Features: Auto-bold formatting + POI hover cards');
  console.log('🤖 Powered by: Gemini AI with intelligent formatting prompts');
  console.log('='.repeat(80));
  
  // Test various destination search scenarios
  await testEnhancedResponse('hotels in ooty');
  await testEnhancedResponse('restaurants in mumbai');
  await testEnhancedResponse('show me attractions in coonoor');
  await testEnhancedResponse('find places to stay in kerala');
  await testEnhancedResponse('what to do in goa');
  
  console.log('\n✨ Testing complete!');
  console.log('\n🎉 Benefits of Enhanced Responses:');
  console.log('   • POI names are automatically formatted as **bold**');
  console.log('   • Destination names are highlighted in **bold**');
  console.log('   • POI names trigger hover cards with data-poi-hover attributes');
  console.log('   • AI generates contextual, engaging responses');
  console.log('   • No manual formatting required - everything is automated!');
}

runTests().catch(console.error);