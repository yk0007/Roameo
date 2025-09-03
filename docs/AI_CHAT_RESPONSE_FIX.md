# AI Chat Response Visibility Fix

## Issue Summary

The user reported that AI chat responses were not visible for PLAN_TRIP intent, while general chat intent was working fine. The logs showed:

```
[router] Intent detected: PLAN_TRIP for message: "plan a trip to ooty"
[planner] Trip details extraction response: ```json
{"destination": "Ooty"}
```
[planner] Extracted trip details: { destination: 'Ooty' }
[persist] Flushing trip patch for session a42ea186-9ecb-4ff7-a5a6-d1661c82bef9: {
"destination": "Ooty",
"days": 0,
"title": "[gemini-2.5-flash] error 503: {\n \"error\": {\n \"code\": 503,"
}
```

The problem was that Gemini API errors (specifically 503/400 errors) were causing:
1. **Title generation failures** that returned error text as the title
2. **Chat response generation failures** that could break the response flow
3. **No proper error handling** for API configuration issues

## Root Cause

The issue occurred in multiple places:
1. `generateSessionTitle()` function was returning Gemini error messages as titles
2. `plannerAgent()` wasn't handling all types of Gemini errors gracefully
3. `GeminiClient` was returning error messages instead of throwing exceptions
4. No fallback mechanisms for when AI services are unavailable

## Fixes Applied

### 1. Improved Title Generation Error Handling
**File:** `Backend/src/agents/title.ts`

- Added rejection of error responses containing "error", "503", etc.
- Enhanced fallback handling with better logging
- Ensured fallback titles are always used when Gemini fails

```typescript
// Check for error responses or invalid content
if (!title || title.startsWith("[gemini:") || title.includes("error") || title.includes("503")) {
  console.log(`[title] Gemini returned invalid response: ${title}, using fallback`);
  return fallback;
}
```

### 2. Enhanced Planner Agent Error Handling
**File:** `Backend/src/agents/planner.ts`

- Added comprehensive error checking for chat response generation
- Enhanced catch block to handle specific error types (API config, 503, 429, etc.)
- Improved fallback responses based on available information

```typescript
// Check for error responses or invalid content
if (!chatResponse?.trim() || 
    chatResponse.startsWith("[gemini:") || 
    chatResponse.includes("error 503") || 
    chatResponse.includes("error 500") ||
    chatResponse.includes("error 429")) {
  // ... use appropriate fallback response
}
```

### 3. Graph-Level Error Handling
**File:** `Backend/src/graph/graph.ts`

- Added try-catch around title generation calls
- Implemented fallback title creation when generation fails
- Ensured trip planning continues even if title generation fails

```typescript
// Generate title with fallback handling
let title: string;
try {
  title = await generateSessionTitle({...});
} catch (error: any) {
  console.log(`[planner] Title generation failed: ${error?.message || error}, using fallback`);
  const sessionSuffix = Math.random().toString(36).substring(2, 5).toUpperCase();
  title = res.destination ? `✨ ${res.destination} Adventure #${sessionSuffix}` : `✨ Dream Trip #${sessionSuffix}`;
}
```

### 4. Improved Gemini Client Error Handling
**File:** `Backend/src/tools/gemini.ts`

- Enhanced retry logic to handle all 5xx server errors
- Better error message sanitization for user-facing responses
- Proper exception throwing instead of returning error text

```typescript
// For user-facing errors, don't expose raw API errors
if (res.status === 400 && text.includes("API key expired")) {
  throw new Error("API configuration issue");
}

// Retry on 5xx server errors
if ((res.status >= 500 && res.status < 600) && attempt < maxRetries) {
  // ... retry logic
}

// For non-retriable errors, throw a generic error
throw new Error(`API request failed with status ${res.status}`);
```

### 5. Enhanced Chat Agent Error Handling
**File:** `Backend/src/agents/chat.ts`

- Added specific handling for API configuration issues
- Better fallback messages for users

### 6. Intent Detection Fallback (Partial)
**File:** `Backend/src/agents/intent.ts`

- Added heuristic fallback when Gemini fails
- Pattern matching for planning keywords + destinations
- Graceful degradation to simple keyword detection

## Testing Results

After applying the fixes:

1. **Error Responses are User-Friendly**: Instead of showing raw Gemini error messages, users now see appropriate messages like:
   - "I'm experiencing some technical difficulties right now. Please try again in a few minutes!"
   - "I'd be happy to help you plan your 3-day trip to Ooty! I'm having some technical difficulties..."

2. **No More Broken Responses**: The system no longer returns null or breaks completely when Gemini fails

3. **Proper Event Flow**: Chat responses are always generated, ensuring the frontend receives proper events

4. **Graceful Degradation**: When AI services fail, the system falls back to sensible defaults and helpful error messages

## Key Improvements

### Before Fix:
- Raw API errors shown to users: `"[gemini-2.5-flash] error 503: {...}"`
- No response generated when title generation failed
- System could break completely on Gemini failures
- Error messages saved as trip titles, corrupting the data flow

### After Fix:
- User-friendly error messages: `"I'm experiencing some technical difficulties..."`
- Always generates appropriate fallback responses
- Continues operation even when individual components fail
- Proper error isolation prevents cascading failures

## Technical Benefits

1. **Resilience**: System continues to function even when AI services are unavailable
2. **User Experience**: Users get helpful messages instead of cryptic error codes
3. **Data Integrity**: Error messages don't corrupt trip data or titles
4. **Monitoring**: Better error logging for debugging and monitoring
5. **Graceful Degradation**: Fallback mechanisms ensure core functionality remains available

## Conclusion

The original issue was that Gemini API failures (especially 503 errors) were breaking the trip planning flow by:
1. Returning error messages as titles instead of proper titles
2. Not generating chat responses when AI services failed
3. Propagating errors up the call stack without proper handling

The fixes ensure that:
1. **Error messages never reach users as content**
2. **Appropriate fallback responses are always generated**
3. **The system continues to function even when AI services fail**
4. **Users receive helpful, contextual error messages**

This makes the system much more robust and provides a better user experience during API outages or configuration issues.