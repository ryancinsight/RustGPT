# RustGPT Web UI API Documentation

## Overview

The RustGPT Web UI provides a RESTful API for interacting with trained language models. The API is designed to be compatible with OpenAI's API specification where applicable.

## Base URL

```
http://localhost:8080
```

## Authentication

When authentication is enabled, include your API key in the Authorization header:

```
Authorization: Bearer YOUR_API_KEY
```

## Endpoints

### Health Check

```
GET /health
```

Returns the server health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1704067200
}
```

### Server Status

```
GET /v1/status
```

Returns detailed server status and statistics.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "current_model": "model-name",
  "uptime_seconds": 3600,
  "stats": {
    "total_requests": 100,
    "successful_requests": 95,
    "failed_requests": 5,
    "requests_per_second": 2.5,
    "avg_latency_ms": 150.0,
    "total_tokens_generated": 5000,
    "success_rate": 95.0
  }
}
```

### List Models

```
GET /v1/models
```

Returns a list of available models.

**Response:**
```json
{
  "data": [
    {
      "id": "model-name",
      "name": "model-name",
      "architecture": "Autoregressive",
      "config": {
        "embedding_dim": 128,
        "hidden_dim": 256,
        "num_layers": 3,
        "num_heads": 4,
        "max_seq_len": 80
      },
      "path": "models/model-name.bin",
      "size_bytes": 100000000,
      "size_human": "95.37 MB",
      "created_at": "2024-01-01T00:00:00Z",
      "is_loaded": true
    }
  ],
  "total": 1
}
```

### Get Model

```
GET /v1/models/:model_id
```

Returns information about a specific model.

### Load Model

```
POST /v1/models/load
```

Load a model into memory.

**Request Body:**
```json
{
  "model": "model-name"
}
```

**Response:**
```json
{
  "success": true,
  "model": { ... },
  "load_time_ms": 500
}
```

### Unload Model

```
POST /v1/models/unload
```

Unload the current model from memory.

### Chat Completions

```
POST /v1/chat/completions
```

Create a chat completion (OpenAI-compatible).

**Request Body:**
```json
{
  "model": "model-name",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "stream": false,
  "conversation_id": "conv-123"
}
```

**Response:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1704067200,
  "model": "model-name",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  },
  "stats": {
    "time_to_first_token_ms": 50.0,
    "total_time_ms": 200.0,
    "tokens_per_second": 100.0
  }
}
```

### Text Completions

```
POST /v1/completions
```

Create a text completion (legacy OpenAI-compatible).

**Request Body:**
```json
{
  "model": "model-name",
  "prompt": "Once upon a time",
  "max_tokens": 256,
  "temperature": 0.7
}
```

### List Conversations

```
GET /v1/conversations
```

Returns a list of conversation sessions.

### Get Conversation

```
GET /v1/conversations/:id
```

Returns details of a specific conversation.

### Delete Conversation

```
DELETE /v1/conversations/:id
```

Delete a conversation session.

## Streaming

Set `stream: true` in chat completion requests to receive server-sent events (SSE) with incremental responses.

## Error Handling

Errors follow the standard HTTP status codes:

- `200` - Success
- `400` - Bad Request
- `401` - Unauthorized
- `404` - Not Found
- `500` - Internal Server Error

Error responses include a JSON body:

```json
{
  "error": {
    "message": "Error description",
    "type": "error_type",
    "code": "500"
  }
}
```
