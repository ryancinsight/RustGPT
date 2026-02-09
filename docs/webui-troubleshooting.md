# RustGPT WebUI Troubleshooting Guide

## Client-Side Diagnostics

### 1. Browser Console Errors

Open Developer Tools (F12) → Console tab and look for errors when:
- Clicking "New chat" button
- Selecting a different model from dropdown
- Sending a message

**Common errors:**
| Error | Likely Cause | Solution |
|-------|--------------|----------|
| `Failed to fetch` | Network/Server down | Restart webui server |
| `CORS policy` error | CORS not enabled | Enable CORS in config |
| `TypeError` null property | JavaScript bug | Clear cache, reload |

### 2. Browser Cache Issues

1. Hard refresh: `Ctrl+Shift+R` (Windows) or `Cmd+Shift+R` (Mac)
2. Clear browser cache:
   - Chrome: Settings → Privacy → Clear browsing data
   - Firefox: Options → Privacy → Clear Data
3. Try incognito/private window mode
4. Disable browser extensions temporarily

### 3. Network Tab Analysis

Open Developer Tools → Network tab → Attempt failing action

**Response codes:**
| Code | Meaning | Action |
|------|---------|--------|
| 200 | Success | Check UI rendering |
| 400 | Bad request | Check request format |
| 401/403 | Auth issue | Check API keys |
| 500 | Server error | Check server logs |
| 502/503 | Service unavailable | Restart server |

---

## Server-Side Diagnostics

### 1. Server Health Check

```bash
# Test basic connectivity
curl http://127.0.0.1:8080/health

# Expected response:
# {"status":"healthy","timestamp":...}
```

### 2. API Endpoint Tests

```bash
# List available models
curl http://127.0.0.1:8080/v1/models

# Create a chat completion (mock)
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"hello"}]}'
```

### 3. Check Server Logs

Look for ERROR level logs:
```bash
2026-02-09T15:08:31Z ERROR tower_http::trace::on_failure: response failed classification=Status code: 500
```

---

## Common Issues & Solutions

### Issue: "New Chat" button does nothing

**Diagnosis:**
- Check console for JS errors
- Check Network tab for `/v1/conversations` API calls

**Solutions:**
```bash
# Restart the server
pkill -f webui
cargo run --bin webui

# Clear localStorage (stored conversations)
# Open browser console:
localStorage.clear()
```

### Issue: Cannot switch models

**Diagnosis:**
```bash
# Check models directory exists
ls -la models/

# Test model loading API
curl http://127.0.0.1:8080/v1/models
```

**Solutions:**
```bash
# Add model files to models/ directory
# Supported formats: .bin, .safetensors, .pt, .ckpt

# Restart server after adding models
```

### Issue: 500 Internal Server Error

**Solutions:**
```bash
# Check if port is already in use
netstat -ano | findstr :8080

# Kill any existing processes
taskkill /PID <PID> /F

# Restart server
cargo run --bin webui
```

---

## Quick Fixes to Try First

1. **Restart the server:**
   ```bash
   pkill -f webui && cargo run --bin webui
   ```

2. **Clear browser data:**
   - Clear cookies and cache
   - Hard refresh the page (Ctrl+Shift+R)

3. **Check model directory:**
   ```bash
   ls models/
   # If empty, add model files
   ```

4. **Verify no port conflicts:**
   ```bash
   # Windows
   netstat -ano | findstr :8080
   ```

---

## Verification Steps

After applying fixes, verify:

1. [ ] Health endpoint returns 200: `curl http://127.0.0.1:8080/health`
2. [ ] Models endpoint returns list: `curl http://127.0.0.1:8080/v1/models`
3. [ ] WebUI loads without console errors
4. [ ] "New chat" button creates new conversation
5. [ ] Model dropdown shows available models
6. [ ] Chat completion works with selected model
