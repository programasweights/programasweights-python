# Browser Inference

Run PAW programs directly in the browser via WebAssembly. No custom server, no API key, no build setup.

Programs compiled with the compact interpreter (GPT-2 124M) run entirely client-side. The base model (134 MB) downloads once and is cached; each program adds ~12 MB total (~5 MB adapter + ~7 MB prefix cache).

## Quick Start

### CDN (no build tools)

```html
<script type="module">
  import paw from 'https://cdn.jsdelivr.net/npm/@programasweights/web';

  const fn = await paw.function('email-triage-browser');
  const result = await fn('Urgent: server is down!');
  console.log(result); // "immediate"
</script>
```

### npm

```bash
npm install @programasweights/web
```

```javascript
import paw from '@programasweights/web';

const fn = await paw.function('email-triage-browser', {
  onProgress: ({ loaded, total, stage }) => {
    console.log(`${stage}: ${Math.round(loaded/total*100)}%`);
  },
});

const result = await fn('Check this urgent message');
console.log(result);

// Clean up adapter (base model stays cached)
await fn.free();
```

## How It Works

1. **Base model** — Compact interpreter (GPT-2 124M, 134 MB) downloads from Hugging Face CDN and is cached in the browser after first load.
2. **LoRA adapter** — Each program is a ~5 MB Q4_0 GGUF LoRA adapter that specializes the base model for a specific task.
3. **Prefix cache** — A precomputed KV cache (~7 MB) eliminates the prompt prefill step, making the first inference call fast.
4. **Inference** — Runs via WebAssembly (llama.cpp compiled to WASM with SIMD). ~200ms per call on Chrome.

Multiple programs share the cached base model. Loading a second program only downloads ~12 MB (adapter + prefix cache).

If you want browser inference to stay independent of the PAW API at runtime, load by program ID rather than slug. Slugs still need one API call for resolution.

## API Reference

### `paw.function(slugOrId, options?)`

Load a program and return a callable function.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `slugOrId` | `string` | required | Program slug (for example `"programasweights/email-triage-browser"`) or program ID hash |
| `options.onProgress` | `function` | — | Progress callback: `({ loaded, total, stage }) => void` |
| `options.maxTokens` | `number` | `512` | Maximum output tokens |
| `options.temperature` | `number` | `0` | Sampling temperature (0 = greedy) |

**Returns:** `Promise<(input: string) => Promise<string>>`

The returned function also has:

- `.free()` — releases the LoRA adapter (base model stays cached)
- `.spec` — the program's original specification
- `.programId` — the program's content-addressable ID

### `paw.configure(config)`

Set global configuration.

```javascript
paw.configure({
  apiUrl: 'https://your-server.com/api/v1',
});
```

## Download Sizes

| Component | Size | When |
|-----------|------|------|
| Base model (GPT-2 Q8_0) | 134 MB | First program load (cached) |
| LoRA adapter | ~5 MB | Per program |
| Prefix cache | ~7 MB | Per program |
| **First load total** | **~146 MB** | |
| **Switching programs** | **~12 MB** | |

## Browser Compatibility

| Feature | Chrome 92+ | Firefox 100+ | Edge 92+ | Safari 15.2+ |
|---------|-----------|-------------|---------|-------------|
| WASM SIMD | Yes | Yes | Yes | Yes |
| Multi-threaded WASM | Yes | Yes | Yes | Yes |
| Single-thread fallback | Yes | Yes | Yes | Yes |

Multi-threading requires `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: credentialless` headers on your web server.

## Performance

Expected performance with prefix cache:

| Metric | Chrome | Firefox |
|--------|--------|---------|
| First inference | ~200ms | ~500ms |
| Subsequent calls | ~200ms | ~500ms |

Performance depends on the output length — classification tasks (1-2 tokens) are fastest. Longer outputs (JSON repair, text extraction) scale linearly with output token count.

## Multi-threading Headers

For best performance, add these response headers to your web server:

**nginx:**
```nginx
add_header Cross-Origin-Opener-Policy "same-origin" always;
add_header Cross-Origin-Embedder-Policy "credentialless" always;
```

**Express.js:**
```javascript
app.use((req, res, next) => {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'credentialless');
  next();
});
```

Without these headers, the SDK falls back to single-threaded WASM (still functional, but slower).

## Limitations

- Only programs compiled with the **compact** interpreter (GPT-2 124M) are supported. Programs compiled with the standard interpreter (Qwen3 0.6B) are too large for browser inference (~594 MB base model).
- The 134 MB base model download may be slow on mobile connections.
- Performance varies by browser (Chrome is fastest).
