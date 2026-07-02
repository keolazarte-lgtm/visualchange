# 🚀 MagicVisual — Guía de Deploy

App de edición de fotos con IA: subís tu foto → IA genera outfit → preserva tu rostro. 100% gratis.

## ⚠️ Antes de empezar: necesitás una API key de Z.ai

1. Andá a **https://z.ai/manage-apikey/apikey-list**
2. Logeate con tu cuenta de Z.ai (la misma que usás para chatear conmigo)
3. Click en "Create API Key"
4. Copiá el key (formato: `xxxxxxxx.xxxxxxxxxxxxxxxx`)
5. Guardalo — lo vas a necesitar en todos los deploys

Costo: Z.ai tiene free tier generoso (miles de requests gratis por mes). Para uso intensivo de todo el día, probablemente necesites el plan de pago (~$5-20/mes según uso).

---

## 🏆 Recomendación: Railway ($5/mes, 24/7 siempre activo)

**Por qué Railway:**
- ✅ Bun nativo (no Docker necesario)
- ✅ 24/7 siempre activo (no se recicla como el sandbox de Z.ai)
- ✅ Sin timeout (podés generar imágenes de 30s+ sin problema)
- ✅ Free tier: $5 de crédito por mes (~1000 imágenes)
- ✅ Deploy en 2 minutos

### Pasos:

1. Andá a **https://railway.app** y logeate con GitHub
2. Click en **"New Project"** → **"Deploy from GitHub repo"**
3. Pero primero, subí el código a GitHub:
   - Creá un repo nuevo en GitHub (ej: `magicvisual`)
   - Subí todos los archivos de esta carpeta al repo
4. En Railway, seleccioná tu repo `magicvisual`
5. Railway detecta `bun` automáticamente desde `package.json`
6. En la pestaña **"Variables"**, agregá:
   ```
   ZAI_API_KEY=tu_api_key_aqui
   RATE_LIMIT_MAX=0
   ```
7. Click en **"Deploy"**
8. En 1-2 min te da un URL pública tipo `https://magicvisual-production.up.railway.app/`
9. ¡Listo! Entrá desde cualquier dispositivo, sin límites

**Comando alternativo (CLI):**
```bash
npm i -g @railway/cli
railway login
cd magicvisual-deploy
railway init
railway up
```

---

## 🥈 Alternativa 2: Render (free tier disponible)

**Pros:** Free tier (con sleep después de 15 min inactivo)
**Contras:** Se duerme, primer request tarda 30s en despertar

### Pasos:

1. Subí el código a GitHub (igual que Railway)
2. Andá a **https://dashboard.render.com**
3. Click **"New +"** → **"Web Service"**
4. Conectá tu repo de GitHub
5. Render detecta el `Dockerfile` automáticamente
6. En **"Environment"**, agregá:
   ```
   ZAI_API_KEY=tu_api_key_aqui
   RATE_LIMIT_MAX=0
   ```
7. Click **"Create Web Service"**
8. URL: `https://magicvisual.onrender.com/`

Para 24/7 sin sleep: upgrade a **Starter plan** ($7/mes)

---

## 🥉 Alternativa 3: Vercel (con advertencia importante)

**⚠️ ADVERTENCIA:** Vercel free tier tiene **timeout de 10 segundos**. Nuestro pipeline toma **25 segundos**. Va a fallar en el plan free.

Solo funcionaría en **Vercel Pro** ($20/mes) con timeout de 300s.

### Si igual querés Vercel:

1. El código actual usa `Bun.serve()` que NO es compatible con Vercel directamente
2. Necesitarías refactorizar a Next.js API Routes
3. Te cobran $20/mes para que ande el timeout
4. **No recomendado para este caso de uso**

---

## 🏗 Arquitectura del deploy

```
MagicVisual
├── index.ts          # Server Bun + pipeline + UI inline (1458 líneas)
├── package.json      # Deps: z-ai-web-dev-sdk, @gradio/client, sharp
├── Dockerfile        # Universal (Railway, Render, Fly.io, DO, AWS)
├── railway.toml      # Config específica Railway
├── render.yaml       # Config específica Render
├── .env.example      # Template de variables
├── .gitignore
└── README.md         # Esta guía
```

## 🔧 Endpoints del server deployado

- `GET /` → UI completa (HTML)
- `GET /health` → Status (`{"ok":true, "uptime_sec":...}`)
- `POST /api/edit` → Pipeline completo (SSE stream, ~25s)
- `POST /api/describe` → Solo VLM analysis
- `POST /api/suggest-outfits` → LLM sugiere outfits

## 💰 Comparativa de costos

| Plataforma | Plan | Costo/mes | Características |
|------------|------|-----------|-----------------|
| Railway | Hobby | $5 | 24/7, sin timeout, Bun nativo ⭐ |
| Render | Free | $0 | Se duerle, 30s wake-up |
| Render | Starter | $7 | 24/7, sin timeout |
| Vercel | Free | $0 | ❌ No funciona (10s timeout) |
| Vercel | Pro | $20 | Funciona con 300s timeout |
| Fly.io | Free | $0 | 24/7 con limitaciones |
| Z.ai sandbox | Free | $0 | Se recicla cada horas ❌ |

## ✅ Testing post-deploy

Después de deployar, verificá:

```bash
# 1. Health check
curl https://tu-url-deployada.com/health

# 2. UI carga
curl -s https://tu-url-deployada.com/ | head -5

# 3. Pipeline completo (con una foto de prueba)
curl -X POST https://tu-url-deployada.com/api/describe \
  -F "photo=@test.jpg"
```

## 🆘 Troubleshooting

**"Build failed: Cannot find module z-ai-web-dev-sdk"**
→ Asegurate de tener `"z-ai-web-dev-sdk": "latest"` en package.json

**"Sharp: libvips not found"**
→ En Docker, ya está en el Dockerfile. En Railway nativo, debería funcionar out-of-the-box.

**"ZAI_API_KEY invalid"**
→ Verificá que la key no tenga espacios en blanco. Formato correcto: `xxxxxxxx.xxxxxxxxxxxxxxxx`

**"HF Space timeout"**
→ HuggingFace a veces se pone lento. El código tiene 60s timeout + fallback a sharp composite automático.

**"InsightFace returns null"**
→ El space `felixrosberg/face-swap` está orientado a anonimización. A veces no devuelve nada. Probá con otra foto (mejor iluminada, cara más frontal).

## 📞 Soporte

Si algo no anda, mandame el log de error y te ayudo a debuggear.
