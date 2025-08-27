# Memcached Cache Layer Implementation

This document describes the Memcached cache layer implementation for the Roameo backend application.

## Overview

The cache layer provides distributed caching capabilities using Memcached to improve application performance by reducing database load and speeding up data retrieval operations.

## Features

- **Session Caching**: Automatic caching of user sessions with TTL management
- **Search Results Caching**: Cache search queries and results to reduce API calls
- **POI Data Caching**: Cache Points of Interest data for faster retrieval
- **Trip & Itinerary Caching**: Cache trip planning data and itineraries
- **Health Monitoring**: Built-in health checks and connection monitoring
- **Graceful Degradation**: Falls back to database when cache is unavailable
- **Batch Operations**: Support for batch session retrieval
- **Cache Warming**: Proactive cache population for frequently accessed data

## Installation

### Prerequisites

1. **Install Memcached Server**:

   **On macOS (using Homebrew):**
   ```bash
   brew install memcached
   brew services start memcached
   ```

   **On Ubuntu/Debian:**
   ```bash
   sudo apt-get update
   sudo apt-get install memcached
   sudo systemctl start memcached
   sudo systemctl enable memcached
   ```

   **On CentOS/RHEL:**
   ```bash
   sudo yum install memcached
   sudo systemctl start memcached
   sudo systemctl enable memcached
   ```

   **Using Docker:**
   ```bash
   docker run -d --name memcached -p 11211:11211 memcached:latest
   ```

2. **Node.js Dependencies** (already installed):
   ```bash
   npm install memcached @types/memcached
   ```

## Configuration

### Environment Variables

Add the following to your `.env` file:

```bash
# Memcached Configuration
MEMCACHED_SERVERS=localhost:11211
# For multiple servers: server1:11211,server2:11211,server3:11211

# Optional: Admin token for cache management endpoints
ADMIN_TOKEN=your-secure-admin-token-here
```

### Multiple Servers

For high availability, configure multiple Memcached servers:

```bash
MEMCACHED_SERVERS=cache1.example.com:11211,cache2.example.com:11211,cache3.example.com:11211
```

## Usage

### Backend Integration

The cache layer is automatically integrated into the backend via the `CachedDb` class, which wraps the existing `WriteThroughDb`:

```typescript
import { CachedDb } from './cache/cached-db.js';

// Automatic cache integration
const db = new CachedDb(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);
```

### Cache Operations

#### Session Management
```typescript
// Get session (checks cache first, then database)
const session = await db.getSession(sessionId);

// Update session (updates database and cache)
const updatedSession = await db.upsertSession(sessionId, updateData);
```

#### Search Results
```typescript
// Cache search results
await db.setSearchResults(query, results, sessionId);

// Retrieve cached search results
const cachedResults = await db.getSearchResults(query, sessionId);
```

#### POI Data
```typescript
// Cache POI data
await db.setPoiData(poiId, poiData);

// Retrieve cached POI data
const cachedPoi = await db.getPoiData(poiId);
```

## API Endpoints

### Cache Management

The backend provides REST API endpoints for cache management:

#### Get Cache Statistics
```http
GET /api/cache/stats
```

Response:
```json
{
  "success": true,
  "stats": {
    "connected": true,
    "stats": {
      "server1:11211": {
        "pid": 12345,
        "uptime": 3600,
        "curr_items": 150,
        "total_items": 1000,
        "bytes": 524288,
        "curr_connections": 5,
        "cmd_get": 1500,
        "cmd_set": 500,
        "get_hits": 1200,
        "get_misses": 300
      }
    }
  },
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

#### Cache Health Check
```http
GET /api/cache/health
```

Response:
```json
{
  "healthy": true,
  "message": "Memcached is healthy",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

#### Flush Cache (Admin Only)
```http
POST /api/cache/flush
Authorization: Bearer your-admin-token
```

#### Warm Cache
```http
POST /api/cache/warm
Content-Type: application/json

{
  "sessionIds": ["session1", "session2", "session3"]
}
```

#### Session Operations
```http
# Get cached session info
GET /api/cache/session/:sessionId

# Delete session from cache and database (Admin only)
DELETE /api/cache/session/:sessionId
Authorization: Bearer your-admin-token
```

## Performance Benefits

### Cache Hit Ratios

Expected performance improvements:

- **Session Retrieval**: 90%+ cache hit ratio, ~10ms vs ~100ms database query
- **Search Results**: 70%+ cache hit ratio for repeated searches
- **POI Data**: 80%+ cache hit ratio for popular destinations
- **Trip Data**: 85%+ cache hit ratio for active planning sessions

### TTL (Time To Live) Settings

- **Sessions**: 1 hour (3600 seconds)
- **Search Results**: 30 minutes (1800 seconds)
- **POI Data**: 2 hours (7200 seconds)
- **Trip/Itinerary Data**: 1 hour (3600 seconds)

## Monitoring

### Health Checks

The cache layer includes comprehensive health monitoring:

1. **Connection Health**: Automatic connection monitoring with retry logic
2. **Performance Monitoring**: Tracks slow operations and logs warnings
3. **Error Handling**: Graceful degradation when cache is unavailable
4. **Statistics**: Real-time cache statistics and hit/miss ratios

### Logging

Cache operations are logged with appropriate levels:

```bash
[memcached] Connected successfully
[memcached] Connection issue: Server timeout
[cached-db] Cache read failed for session abc123: Connection timeout
[cached-db] Warming cache for 50 sessions
```

## Troubleshooting

### Common Issues

1. **Connection Refused**
   ```bash
   # Check if Memcached is running
   telnet localhost 11211
   
   # Or check with netstat
   netstat -an | grep 11211
   ```

2. **Memory Issues**
   ```bash
   # Check Memcached memory usage
   echo "stats" | nc localhost 11211
   ```

3. **High Memory Usage**
   ```bash
   # Flush cache if needed
   echo "flush_all" | nc localhost 11211
   ```

### Performance Tuning

1. **Increase Memory Limit**:
   ```bash
   # Start Memcached with 512MB memory
   memcached -m 512 -p 11211 -d
   ```

2. **Configure Multiple Instances**:
   ```bash
   # Run multiple instances on different ports
   memcached -m 256 -p 11211 -d
   memcached -m 256 -p 11212 -d
   ```

3. **Monitor Cache Efficiency**:
   ```bash
   # Check cache statistics regularly
   curl http://localhost:3000/api/cache/stats
   ```

## Security Considerations

1. **Network Security**: Memcached doesn't have built-in authentication. Use firewall rules to restrict access.

2. **Admin Endpoints**: Cache management endpoints require admin token authentication.

3. **Data Sensitivity**: Avoid caching sensitive personal data. The implementation focuses on session metadata and application data.

## Deployment

### Production Recommendations

1. **Use Multiple Servers**: Deploy at least 2-3 Memcached instances for redundancy
2. **Monitor Memory Usage**: Set up alerts for memory usage above 80%
3. **Network Configuration**: Use private networks for cache communication
4. **Regular Health Checks**: Monitor cache health and performance metrics

### Docker Deployment

```yaml
version: '3.8'
services:
  memcached:
    image: memcached:latest
    ports:
      - "11211:11211"
    command: memcached -m 512
    restart: unless-stopped
    
  backend:
    build: ./Backend
    environment:
      - MEMCACHED_SERVERS=memcached:11211
    depends_on:
      - memcached
```

## Migration

### From No Cache

The cache layer is designed to work alongside the existing database. No migration is required - the cache will be populated as data is accessed.

### Cache Invalidation

The system automatically handles cache invalidation when data is updated:

- Session updates invalidate session cache
- Message additions invalidate session cache
- Trip updates invalidate related caches

## Development

### Running Locally

1. Start Memcached:
   ```bash
   brew services start memcached
   # or
   memcached -p 11211 -d
   ```

2. Start the backend:
   ```bash
   cd Backend
   npm run dev
   ```

3. Check cache health:
   ```bash
   curl http://localhost:3000/api/cache/health
   ```

### Testing

```bash
# Test cache connectivity
npm run test:cache

# Load test cache performance
npm run test:cache:performance
```

## Future Enhancements

1. **Redis Migration**: Plan to support Redis as an alternative cache backend
2. **Compression**: Add compression for large cached objects
3. **Cache Warming Strategies**: Implement intelligent cache warming based on usage patterns
4. **Metrics Integration**: Integration with monitoring systems like Prometheus
5. **Cache Partitioning**: Implement cache partitioning strategies for better performance