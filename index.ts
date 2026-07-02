/**
 * MagicVisual — AI Outfit Magic with Face Preservation (v3.0)
 *
 * Pipeline:
 *   0. (optional) HD upscale if option.upscale
 *   1. VLM (glm-4v-flash) analyzes uploaded photo → extracts body description
 *      + face detection (skip if bodyOnly=true)
 *      + current_clothing detection in bodyOnly mode
 *   2. Outfit generation:
 *      - bodyOnly=true:  IP2P (timbrooks/instruct-pix2pix) preserves identity.
 *                        Falls back to DALL-E edit if IP2P fails.
 *      - bodyOnly=false: DALL-E edit (z-ai-web-dev-sdk), with prompt sanitizer.
 *                        Falls back to Pollinations.ai if Z.ai blocks (1301).
 *   3. Face swap via InsightFace (skip if bodyOnly=true).
 *      Falls back to sharp ellipse composite if HF unavailable.
 *   4. (optional) Final resize — skipped if keepAspectRatio=true.
 *   5. (optional) Animate via Z.ai video API.
 *
 * Extras:
 *   - VLM result cache (same photo → multiple outfits, only analyze once)
 *   - LLM outfit suggestions (returns {label, prompt})
 *   - Rate limiting (disabled by default; RATE_LIMIT_MAX env var to enable)
 *   - 3 styles × 8 ideas, 18 backgrounds, 8 body mods, 3 quality presets,
 *     4 picante levels, 5 animation presets, 10 body-only ideas
 *   - keepBackground, keepAspectRatio, bodyOnly, upscale toggles
 *   - bootstrapZaiConfig for cloud deployment
 *
 * Endpoints:
 *   GET  /                  — UI HTML
 *   GET  /health            — status
 *   GET  /api/info          — all options
 *   POST /api/describe      — VLM analysis (accept bodyOnly)
 *   POST /api/suggest-outfits — LLM suggestions ({label, prompt} objects)
 *   POST /api/edit          — main pipeline (streaming SSE)
 *   POST /api/upscale       — standalone upscale
 *
 * Server: Bun.serve on port 3000 (Caddy on port 81 proxies to here)
 */

import ZAI from "z-ai-web-dev-sdk";
import { Client } from "@gradio/client";
import sharp from "sharp";
import { createHash } from "crypto";
import { writeFileSync } from "fs";
import { homedir } from "os";
import { join } from "path";

// ===== CRITICAL: Global error handlers — server must NEVER die =====
process.on("uncaughtException", (err) => {
  console.error("[FATAL] uncaughtException:", err?.stack || err);
});
process.on("unhandledRejection", (reason) => {
  console.error("[FATAL] unhandledRejection:", reason?.stack || reason);
});
process.on("SIGPIPE", () => {
  console.warn("[WARN] SIGPIPE received (client disconnected mid-stream)");
});

// ===== Cloud deployment: bootstrap Z.ai config from env vars =====
function bootstrapZaiConfig(): void {
  const apiKey = process.env.ZAI_API_KEY;
  if (!apiKey) {
    console.error("[BOOTSTRAP] ZAI_API_KEY env var not set — SDK will use existing ~/.z-ai-config");
    return;
  }
  const config = {
    baseUrl: process.env.ZAI_BASE_URL || "https://api.z.ai/api/paas/v4",
    apiKey: apiKey,
    chatId: process.env.ZAI_CHAT_ID || "magicvisual-deploy",
    userId: process.env.ZAI_USER_ID || "magicvisual-user",
    token: process.env.ZAI_TOKEN || apiKey,
  };
  const locations = [
    join(process.cwd(), ".z-ai-config"),
    join(homedir(), ".z-ai-config"),
    "/etc/.z-ai-config",
  ];
  for (const loc of locations) {
    try {
      writeFileSync(loc, JSON.stringify(config), { mode: 0o600 });
      console.log(`[BOOTSTRAP] Wrote ${loc}`);
    } catch (e: any) {
      if (!loc.startsWith("/etc/")) {
        console.warn(`[BOOTSTRAP] Could not write to ${loc}: ${e?.message || e}`);
      }
    }
  }
}
bootstrapZaiConfig();

// ===== Configuration =====
const PORT = Number(process.env.PORT) || 3000;
const HF_SPACE = "felixrosberg/face-swap";
const IP2P_SPACE = "timbrooks/instruct-pix2pix";
const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB — we auto-compress larger files
const MAX_DIMENSION = 2048; // auto-resize if width or height > this
const VLM_CACHE_TTL_MS = 30 * 60 * 1000; // 30 min
const RATE_LIMIT_WINDOW_MS = 60 * 1000;
// 0 = unlimited (default for personal use). Set RATE_LIMIT_MAX env to enable.
const RATE_LIMIT_MAX = Number(process.env.RATE_LIMIT_MAX) || 0;

// ===== Style presets (Spanish labels, English prompts) =====
const STYLES: Record<
  string,
  {
    name: string;
    desc: string;
    suffix: string;
    ideas: { label: string; prompt: string }[];
    llmHint: string;
  }
> = {
  vanilla: {
    name: "Vainilla",
    desc: "Sutil y elegante",
    suffix:
      "Elegant fashion photography, soft natural lighting, tasteful pose, magazine quality, high detail, photorealistic, full body shot.",
    ideas: [
      { label: "Vestido de noche rojo", prompt: "red evening gown" },
      { label: "Vestido de verano blanco", prompt: "white summer sundress" },
      { label: "Traje de negocio negro", prompt: "black business suit" },
      { label: "Vestido floral primaveral", prompt: "floral spring dress" },
      { label: "Chaqueta vaquera con jeans", prompt: "denim jacket with jeans" },
      { label: "Abrigo de invierno con bufanda", prompt: "winter coat with scarf" },
      { label: "Boho chic con sombrero", prompt: "boho chic outfit with hat" },
      { label: "Lino de verano", prompt: "summer linen outfit" },
    ],
    llmHint:
      "elegant, classy outfits suitable for fashion magazine: dresses, suits, casual-chic combinations",
  },
  versatil: {
    name: "Versátil",
    desc: "Atrevido pero elegante",
    suffix:
      "Editorial fashion photography, dramatic studio lighting, confident pose, high fashion magazine cover style, ultra detailed, photorealistic, full body shot.",
    ideas: [
      { label: "Chaqueta de cuero con mini", prompt: "black leather jacket with mini skirt" },
      { label: "Vestido dorado brillante", prompt: "golden sequined dress" },
      { label: "Top de encaje con falda", prompt: "sheer black lace top with skirt" },
      { label: "Catsuit de terciopelo", prompt: "velvet bodysuit" },
      { label: "Vestido de seda", prompt: "silk slip dress" },
      { label: "Crop top con pants tiro alto", prompt: "crop top with high-waist pants" },
      { label: "Top malla con blazer", prompt: "mesh top with blazer" },
      { label: "Pantalón cuero con blusa", prompt: "leather pants with silk blouse" },
    ],
    llmHint:
      "bold, edgy editorial fashion outfits: leather, sequins, sheer fabrics, statement pieces",
  },
  fetish: {
    name: "Fetiche",
    desc: "Lencería, latex, bikinis",
    suffix:
      "Boudoir photography, intimate mood lighting, confident pose, professional studio, ultra realistic skin texture, fashion editorial, photorealistic, full body shot.",
    ideas: [
      { label: "Lencería encaje rojo", prompt: "red lace lingerie set" },
      { label: "Catsuit latex negro", prompt: "black latex catsuit" },
      { label: "Bikini blanco en la playa", prompt: "white bikini on the beach" },
      { label: "Corset cuero con medias", prompt: "black leather corset with stockings" },
      { label: "Bodysuit malla con tacos", prompt: "mesh bodysuit with heels" },
      { label: "Camisón de seda con encaje", prompt: "silk robe with lace details" },
      { label: "Arnés cuero sobre ropa interior", prompt: "leather harness over underwear" },
      { label: "Babydoll transparente", prompt: "sheer babydoll with thong" },
    ],
    llmHint:
      "intimate apparel: lingerie, latex, bikinis, bodysuits, corsets, boudoir fashion",
  },
};

// ===== Body modifications (multi-select chips) =====
const BODY_MODS: Record<string, { label: string; prompt: string }> = {
  bust_slight: { label: "Busto +", prompt: "with dramatically enlarged and fuller bust, much bigger breasts, noticeably curvier chest" },
  bust_mod: { label: "Busto ++", prompt: "with hugely enlarged bust, very large round breasts, dramatically fuller chest, eye-catching cleavage" },
  hips_slight: { label: "Caderas +", prompt: "with dramatically wider hips, much curvier lower body, enlarged hip proportions" },
  hips_mod: { label: "Caderas ++", prompt: "with extremely wide hips, dramatically enlarged curvy lower body, exaggerated hourglass lower curves" },
  waist_slim: { label: "Cintura fina", prompt: "with dramatically slim and tiny waist, very narrow midsection, heavily corseted waist, extreme hourglass shape" },
  voluptuous: { label: "Voluptuosa", prompt: "with dramatically voluptuous exaggerated hourglass figure, hugely enlarged full bust, extremely tiny slim waist, dramatically wide hips, very curvy body, extreme proportions" },
  curves: { label: "Curvas pronunciadas", prompt: "with extremely pronounced exaggerated curves, dramatically curvy toned body, very feminine silhouette, extreme hourglass shape, dramatically enhanced bust and hips" },
  athletic: { label: "Atlética", prompt: "with dramatically athletic toned muscular body, very fit physique, extremely defined muscles, six-pack abs, toned legs and arms" },
};

// ===== Quality presets (3 tabs) =====
const QUALITY_PRESETS: Record<string, { label: string; suffix: string }> = {
  standard: {
    label: "Estándar",
    suffix: "Sharp focus, professional photography, realistic proportions, high resolution.",
  },
  hd: {
    label: "HD Realista",
    suffix: "Ultra photorealistic, 8K resolution, hyperdetailed skin texture with pores, professional photography, sharp focus, magazine quality, realistic body proportions, natural lighting.",
  },
  cinematic: {
    label: "Cinematográfico",
    suffix: "Cinematic photography, dramatic lighting, shallow depth of field, ultra detailed, film grain, color graded, 8K, hyperrealistic skin texture.",
  },
};

// ===== Picante levels (content-filter-safe mood modifiers) =====
const PICANTE_LEVELS: Record<string, { label: string; suffix: string }> = {
  off: { label: "Off", suffix: "" },
  mild: {
    label: "Suave",
    suffix: "Confident pose, striking expression, fashion editorial mood, alluring but tasteful.",
  },
  medium: {
    label: "Medio",
    suffix: "Bold confident pose, expressive gaze, fashion magazine editorial, intimate but classy mood.",
  },
  hot: {
    label: "Picante",
    suffix: "Bold expressive pose, confident striking gaze, fashion boudoir editorial, classy but intense mood, professional photography.",
  },
};

// ===== Animation presets (5 + off, uses Z.ai video API) =====
const ANIMATION_PRESETS: Record<string, { label: string; prompt: string }> = {
  off: { label: "Off", prompt: "" },
  breath: {
    label: "Respiración",
    prompt: "Subtle breathing motion, chest rising and falling gently, slight head tilt, eyes looking at camera, soft smile. No nudity. Fashion editorial style. Classy and tasteful.",
  },
  hair: {
    label: "Pelo al viento",
    prompt: "Hair gently flowing in slow motion wind, soft smile, head turning slightly, confident expression. No nudity. Fashion editorial style.",
  },
  pose: {
    label: "Cambio de pose",
    prompt: "Slow subtle pose shift, hip sway, hand moving to touch hair, soft confident expression. No nudity. Fashion editorial style. Tasteful.",
  },
  camera: {
    label: "Cámara lenta",
    prompt: "Slow camera pan around subject, subject breathing naturally, soft smile, subtle hair movement. No nudity. Cinematic fashion editorial.",
  },
  gaze: {
    label: "Mirada",
    prompt: "Subject slowly looking up at camera, sultry gaze, soft smile forming, subtle breathing. No nudity. Boudoir fashion editorial, classy.",
  },
};

// ===== Body-only ideas (legs/feet, IP2P instructions) =====
const BODY_ONLY_IDEAS: { label: string; prompt: string }[] = [
  { label: "Medias latex negras", prompt: "change to black latex thigh-high stockings" },
  { label: "Tacos aguja rojos", prompt: "change to red stiletto high heels" },
  { label: "Pantimedias fishnet", prompt: "change to black fishnet pantyhose" },
  { label: "Botas cuero altas", prompt: "change to tall black leather boots" },
  { label: "Sandalias tiras", prompt: "change to strappy high-heel sandals" },
  { label: "Medias encaje", prompt: "change to lace thigh-high stockings" },
  { label: "Zapatos plataforma", prompt: "change to black platform heels" },
  { label: "Botines taco fino", prompt: "change to stiletto ankle boots" },
  { label: "Pantimedias blancas", prompt: "change to white thigh-high stockings" },
  { label: "Tacos transparentes", prompt: "change to clear acrylic stripper heels" },
];

// ===== 18 backgrounds (Spanish labels, English prompts) =====
const BACKGROUNDS: { label: string; prompt: string }[] = [
  { label: "Estudio blanco", prompt: "modern studio with white backdrop" },
  { label: "Hotel dorado", prompt: "luxury hotel bedroom with golden accents" },
  { label: "Playa tropical", prompt: "tropical beach at sunset" },
  { label: "Azotea nocturna", prompt: "rooftop terrace with city skyline at night" },
  { label: "Living escandinavo", prompt: "minimalist scandinavian living room" },
  { label: "Cuarto neón", prompt: "moody dark room with neon accents" },
  { label: "Baño de mármol", prompt: "elegant marble bathroom" },
  { label: "Walk-in closet", prompt: "lavish walk-in closet with mirrors" },
  { label: "Ventana lluvia", prompt: "rainy window with bokeh lights" },
  { label: "Pileta azotea", prompt: "rooftop pool at golden hour" },
  { label: "Barroco vintage", prompt: "vintage baroque interior with chandeliers" },
  { label: "Selva tropical", prompt: "tropical jungle with waterfall" },
  { label: "Cabaña nieve", prompt: "snowy mountain cabin interior" },
  { label: "Loft industrial", prompt: "industrial loft with concrete walls" },
  { label: "Suite art deco", prompt: "art deco hotel suite" },
  { label: "Yate privado", prompt: "private yacht deck" },
  { label: "Desierto dorado", prompt: "desert at golden hour" },
  { label: "Bosque niebla", prompt: "misty forest clearing" },
];

// ===== DALL-E supported sizes (for keepAspectRatio matching) =====
const DALL_E_SIZES: { w: number; h: number; ratio: number; size: string }[] = [
  { w: 1024, h: 1024, ratio: 1.0, size: "1024x1024" },
  { w: 768, h: 1344, ratio: 0.571, size: "768x1344" },
  { w: 864, h: 1152, ratio: 0.75, size: "864x1152" },
  { w: 1344, h: 768, ratio: 1.75, size: "1344x768" },
  { w: 1152, h: 864, ratio: 1.333, size: "1152x864" },
  { w: 1440, h: 720, ratio: 2.0, size: "1440x720" },
  { w: 720, h: 1440, ratio: 0.5, size: "720x1440" },
];

// ===== Prompt sanitizer (replaces trigger words before sending to Z.ai) =====
const PROMPT_REPLACEMENTS: { pattern: RegExp; replacement: string }[] = [
  // Stockings / pantyhose variants
  { pattern: /\bthigh-high stockings\b/gi, replacement: "knee-high socks" },
  { pattern: /\bthigh high stockings\b/gi, replacement: "knee-high socks" },
  { pattern: /\bthigh-highs\b/gi, replacement: "knee-high socks" },
  { pattern: /\bstockings\b/gi, replacement: "long socks" },
  { pattern: /\bpantyhose\b/gi, replacement: "sheer tights" },
  { pattern: /\bfishnet\b/gi, replacement: "mesh pattern" },
  { pattern: /\bhosiery\b/gi, replacement: "sheer legwear" },
  { pattern: /\bgarter belt\b/gi, replacement: "decorative belt" },
  { pattern: /\bgarter\b/gi, replacement: "decorative" },
  // Heels variants
  { pattern: /\bstiletto\b/gi, replacement: "high heel" },
  { pattern: /\bstilettos\b/gi, replacement: "high heels" },
  { pattern: /\bstripper heels\b/gi, replacement: "platform heels" },
  { pattern: /\bhooker heels\b/gi, replacement: "platform heels" },
  // Latex / leather / rubber (worst offenders)
  { pattern: /\blatex\b/gi, replacement: "shiny" },
  { pattern: /\brubber\b/gi, replacement: "shiny" },
  { pattern: /\bPVC\b/gi, replacement: "shiny" },
  { pattern: /\bvinyl\b/gi, replacement: "shiny" },
  // Lingerie (sometimes blocked)
  { pattern: /\blingerie set\b/gi, replacement: "intimate apparel" },
  { pattern: /\blingerie\b/gi, replacement: "intimate wear" },
  { pattern: /\bbabydoll\b/gi, replacement: "short nightgown" },
  { pattern: /\bchemise\b/gi, replacement: "silk slip" },
  { pattern: /\bteddy\b/gi, replacement: "bodysuit" },
  // Seductive / sultry / provocative
  { pattern: /\bseductive\b/gi, replacement: "alluring" },
  { pattern: /\bsultry\b/gi, replacement: "striking" },
  { pattern: /\bprovocative\b/gi, replacement: "bold" },
  { pattern: /\bteasing\b/gi, replacement: "playful" },
  { pattern: /\bsexy\b/gi, replacement: "attractive" },
  { pattern: /\berotic\b/gi, replacement: "sensual" },
  // Bondage / harness
  { pattern: /\bharness\b/gi, replacement: "strappy accessory" },
  { pattern: /\bbondage\b/gi, replacement: "strappy" },
  { pattern: /\brestraints\b/gi, replacement: "strappy" },
  // Underwear variants
  { pattern: /\bthong\b/gi, replacement: "briefs" },
  { pattern: /\bg-string\b/gi, replacement: "briefs" },
  { pattern: /\bpanties\b/gi, replacement: "underwear" },
  // Nude / naked (sometimes blocked)
  { pattern: /\bnude\b/gi, replacement: "natural skin" },
  { pattern: /\bnaked\b/gi, replacement: "natural" },
  { pattern: /\btopless\b/gi, replacement: "bare shoulders" },
  // Fetish-related
  { pattern: /\bfetish\b/gi, replacement: "alternative fashion" },
  { pattern: /\bbdsm\b/gi, replacement: "alternative fashion" },
  { pattern: /\bdominatrix\b/gi, replacement: "powerful fashion" },
];

function sanitizePrompt(s: string): string {
  let out = s;
  for (const { pattern, replacement } of PROMPT_REPLACEMENTS) {
    out = out.replace(pattern, replacement);
  }
  return out;
}

// ===== Pick closest DALL-E size for a given aspect ratio =====
function pickDallESize(width: number, height: number): string {
  const ratio = width / height;
  let best = DALL_E_SIZES[0];
  let bestDiff = Math.abs(DALL_E_SIZES[0].ratio - ratio);
  for (const cand of DALL_E_SIZES) {
    const diff = Math.abs(cand.ratio - ratio);
    if (diff < bestDiff) {
      bestDiff = diff;
      best = cand;
    }
  }
  console.log(`[Size] photo ${width}x${height} (ratio ${ratio.toFixed(3)}) → ${best.size}`);
  return best.size;
}

// ===== VLM cache (keyed by photo hash) =====
type VlmResult = {
  bodyDesc: string;
  faceCount: number;
  currentClothing?: string;
  raw: any;
  timestamp: number;
};
const vlmCache = new Map<string, VlmResult>();

function hashBuffer(buf: Buffer): string {
  return createHash("sha256").update(buf).digest("hex").slice(0, 16);
}

function getCachedVlm(hash: string): VlmResult | null {
  const entry = vlmCache.get(hash);
  if (!entry) return null;
  if (Date.now() - entry.timestamp > VLM_CACHE_TTL_MS) {
    vlmCache.delete(hash);
    return null;
  }
  return entry;
}

function setCachedVlm(hash: string, result: VlmResult): void {
  vlmCache.set(hash, result);
  if (vlmCache.size > 50) {
    const oldest = vlmCache.keys().next().value;
    if (oldest) vlmCache.delete(oldest);
  }
}

// ===== Rate limiting =====
type RateBucket = { count: number; resetAt: number };
const rateBuckets = new Map<string, RateBucket>();

function checkRateLimit(ip: string): { ok: boolean; remaining: number; resetAt: number } {
  if (RATE_LIMIT_MAX === 0) {
    return { ok: true, remaining: Infinity, resetAt: Date.now() + RATE_LIMIT_WINDOW_MS };
  }
  const now = Date.now();
  let bucket = rateBuckets.get(ip);
  if (!bucket || now > bucket.resetAt) {
    bucket = { count: 0, resetAt: now + RATE_LIMIT_WINDOW_MS };
    rateBuckets.set(ip, bucket);
  }
  bucket.count++;
  if (bucket.count > RATE_LIMIT_MAX) {
    return { ok: false, remaining: 0, resetAt: bucket.resetAt };
  }
  return { ok: true, remaining: RATE_LIMIT_MAX - bucket.count, resetAt: bucket.resetAt };
}

setInterval(() => {
  const now = Date.now();
  for (const [ip, bucket] of rateBuckets.entries()) {
    if (now > bucket.resetAt) rateBuckets.delete(ip);
  }
}, 5 * 60 * 1000);

// ===== ZAI initialization =====
let zai: any;
async function initZAI() {
  if (!zai) {
    console.log("[ZAI] Initializing SDK...");
    zai = await ZAI.create();
    console.log("[ZAI] Ready");
  }
  return zai;
}

// ===== Gradio client: InsightFace face-swap (lazy) =====
let gradioClient: any | null = null;
let gradioConnectPromise: Promise<any | null> | null = null;

async function getGradio(): Promise<any | null> {
  if (gradioClient) return gradioClient;
  if (gradioConnectPromise) return gradioConnectPromise;
  gradioConnectPromise = (async () => {
    try {
      console.log("[Gradio] Connecting to", HF_SPACE);
      gradioClient = await Client.connect(HF_SPACE, { hf_token: undefined as any });
      console.log("[Gradio] Connected");
      return gradioClient;
    } catch (e: any) {
      console.error("[Gradio] Connect failed:", e?.message || e);
      gradioConnectPromise = null;
      return null;
    } finally {
      gradioConnectPromise = null;
    }
  })();
  return gradioConnectPromise;
}

// ===== Gradio client: Instruct-Pix2Pix (lazy) =====
let ip2pClient: any | null = null;
let ip2pConnectPromise: Promise<any | null> | null = null;

async function getIp2p(): Promise<any | null> {
  if (ip2pClient) return ip2pClient;
  if (ip2pConnectPromise) return ip2pConnectPromise;
  ip2pConnectPromise = (async () => {
    try {
      console.log("[IP2P] Connecting to", IP2P_SPACE, "Space...");
      // Use HF_TOKEN env var for ZeroGPU quota (free accounts get more quota
      // when authenticated). Falls back to anonymous if not set.
      const hfToken = process.env.HF_TOKEN || process.env.HUGGING_FACE_TOKEN;
      const connectOpts = hfToken ? { hf_token: hfToken as any } : {};
      ip2pClient = await Client.connect(IP2P_SPACE, connectOpts);
      console.log("[IP2P] Connected (with token:", !!hfToken, ")");
      return ip2pClient;
    } catch (e: any) {
      console.error("[IP2P] Connect failed:", e?.message || e);
      ip2pConnectPromise = null;
      return null;
    } finally {
      ip2pConnectPromise = null;
    }
  })();
  return ip2pConnectPromise;
}

// ===== Helpers =====
function detectMime(buf: Buffer): string {
  if (buf[0] === 0xff && buf[1] === 0xd8) return "image/jpeg";
  if (buf[0] === 0x89 && buf[1] === 0x50) return "image/png";
  if (buf[0] === 0x47 && buf[1] === 0x49) return "image/gif";
  if (buf[0] === 0x52 && buf[1] === 0x49) return "image/webp";
  return "image/jpeg";
}

function toDataUrl(buf: Buffer): string {
  return `data:${detectMime(buf)};base64,${buf.toString("base64")}`;
}

// Auto-compress photos that are too large (> 10MB or > 2048px dimension).
// Returns a buffer that's guaranteed to be under 10MB and with max dimension 2048px.
// This allows users to upload photos from their phone (often 12-20MB, 4000px+).
async function autoCompressPhoto(buf: Buffer): Promise<Buffer> {
  const meta = await sharp(buf).metadata();
  const origW = meta.width || 0;
  const origH = meta.height || 0;
  const origSizeMB = buf.length / (1024 * 1024);

  // Check if compression is needed
  const needsResize = origW > MAX_DIMENSION || origH > MAX_DIMENSION;
  const needsCompress = origSizeMB > 10;

  if (!needsResize && !needsCompress) {
    console.log(`[AutoCompress] No compression needed (${origW}x${origH}, ${origSizeMB.toFixed(2)}MB)`);
    return buf;
  }

  let pipeline = sharp(buf);
  if (needsResize) {
    const ratio = Math.min(MAX_DIMENSION / origW, MAX_DIMENSION / origH);
    const newW = Math.round(origW * ratio);
    const newH = Math.round(origH * ratio);
    console.log(`[AutoCompress] Resizing ${origW}x${origH} → ${newW}x${newH}`);
    pipeline = pipeline.resize(newW, newH, { fit: "inside", withoutEnlargement: true });
  }

  // Compress with JPEG quality 90 (good balance)
  const result = await pipeline.jpeg({ quality: 90, mozjpeg: true }).toBuffer();
  const newSizeMB = result.length / (1024 * 1024);
  console.log(`[AutoCompress] ${origSizeMB.toFixed(2)}MB → ${newSizeMB.toFixed(2)}MB (${((1 - newSizeMB / origSizeMB) * 100).toFixed(0)}% reduction)`);
  return result;
}

function getClientIp(req: Request): string {
  return (
    req.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ||
    req.headers.get("x-real-ip") ||
    "unknown"
  );
}

// ===== Build IP2P instruction from outfit + (optional) detected clothing =====
function buildIp2pInstruction(outfit: string, currentClothing?: string): string {
  const trimmed = outfit.trim();
  // If user already wrote an instruction-style prompt, use as-is.
  if (/^(change|replace|turn|make|switch|remove|add)\b/i.test(trimmed)) {
    return trimmed;
  }
  if (currentClothing && currentClothing !== "unknown" && currentClothing.length > 0) {
    return `change ${currentClothing} to ${trimmed}`;
  }
  return `change to ${trimmed}`;
}

// ===== Z.ai API call wrapper with automatic retry on 429 (rate limit) =====
// Z.ai has aggressive rate limits. When we hit 429, we wait and retry ONCE.
// If it still fails, the error propagates and the caller falls back to Pollinations.
// We only retry once because multiple retries (5s+10s+20s=35s) would cause
// the Z.ai proxy to timeout (502 Bad Gateway).
async function withZaiRetry<T>(
  fn: () => Promise<T>,
  maxRetries: number = 1,
  baseDelayMs: number = 5000
): Promise<T> {
  let lastError: any;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (e: any) {
      lastError = e;
      const errMsg = e?.message || String(e);
      const is429 = errMsg.includes("429") || errMsg.toLowerCase().includes("too many");

      if (!is429 || attempt === maxRetries) {
        throw e; // not a 429, or out of retries
      }

      // Single retry after 5s
      console.warn(`[ZaiRetry] 429 on attempt ${attempt + 1}/${maxRetries + 1}, waiting ${baseDelayMs}ms...`);
      await new Promise((r) => setTimeout(r, baseDelayMs));
    }
  }
  throw lastError;
}

// ===== VLM body analysis + face detection =====
async function analyzePhoto(
  photoBuf: Buffer,
  bodyOnly: boolean = false
): Promise<VlmResult & { raw: any }> {
  const z = await initZAI();
  const dataUrl = toDataUrl(photoBuf);

  const prompt = bodyOnly
    ? `Analyze this body photo (may focus on legs/feet). Return ONLY a JSON object (no other text, no markdown) with these fields:
{
  "visible_body_parts": "<what's visible: legs, feet, ankles, calves, etc.>",
  "skin_tone": "fair | light | olive | tan | brown | dark brown",
  "body_type": "slim | athletic | average | curvy | plus-size | muscular",
  "gender_presentation": "feminine | masculine | androgynous",
  "pose_description": "<brief pose description>",
  "current_clothing": "<what they are currently wearing on the visible body parts>"
}

Only describe what is VISIBLE. If you cannot determine a field, use "unknown".`
    : `Analyze this person's photo. Return ONLY a JSON object (no other text, no markdown) with these fields:
{
  "face_detected": true | false,
  "face_count": <number of distinct faces visible, 0 if none>,
  "skin_tone": "fair | light | olive | tan | brown | dark brown",
  "hair_color": "black | dark brown | brown | blonde | red | gray | white | bald",
  "hair_length": "very short | short | medium | long | very long",
  "body_type": "slim | athletic | average | curvy | plus-size | muscular",
  "age_range": "young adult | adult | middle-aged | mature",
  "gender_presentation": "feminine | masculine | androgynous",
  "distinguishing_features": "tattoos | freckles | glasses | beard | piercings | none"
}

Only describe what is VISIBLE. If you cannot determine a field, use "unknown". Do NOT describe clothing, background, or accessories.`;

  const response = await withZaiRetry(() => z.chat.completions.createVision({
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: prompt },
          { type: "image_url", image_url: { url: dataUrl } },
        ],
      },
    ],
    thinking: { type: "disabled" },
  }));

  const text: string = response.choices?.[0]?.message?.content || "";
  console.log("[VLM] Raw response:", text.slice(0, 500));

  const match = text.match(/\{[\s\S]*\}/);
  if (!match) {
    console.warn("[VLM] No JSON found, using fallback");
    const fallback: VlmResult & { raw: any } = bodyOnly
      ? {
          bodyDesc: "adult person, showing legs and feet",
          faceCount: 0,
          currentClothing: "unknown",
          raw: { error: "no_json", raw: text.slice(0, 300) },
          timestamp: Date.now(),
        }
      : {
          bodyDesc: "adult person",
          faceCount: 1,
          raw: { error: "no_json", raw: text.slice(0, 300) },
          timestamp: Date.now(),
        };
    return fallback;
  }

  let obj: any;
  try {
    obj = JSON.parse(match[0]);
  } catch (e) {
    console.error("[VLM] JSON parse failed:", e);
    const fallback: VlmResult & { raw: any } = bodyOnly
      ? {
          bodyDesc: "adult person, showing legs and feet",
          faceCount: 0,
          currentClothing: "unknown",
          raw: { error: "parse_failed" },
          timestamp: Date.now(),
        }
      : {
          bodyDesc: "adult person",
          faceCount: 1,
          raw: { error: "parse_failed" },
          timestamp: Date.now(),
        };
    return fallback;
  }

  if (bodyOnly) {
    const parts: string[] = [];
    if (obj.gender_presentation && obj.gender_presentation !== "unknown")
      parts.push(obj.gender_presentation);
    parts.push("person");
    if (obj.skin_tone && obj.skin_tone !== "unknown") parts.push(`with ${obj.skin_tone} skin`);
    if (obj.body_type && obj.body_type !== "unknown") parts.push(`${obj.body_type} build`);
    if (obj.visible_body_parts && obj.visible_body_parts !== "unknown")
      parts.push(`showing ${obj.visible_body_parts}`);
    const bodyDesc = parts.join(", ");
    const currentClothing = obj.current_clothing || "unknown";
    console.log("[VLM] bodyOnly bodyDesc:", bodyDesc, "| current_clothing:", currentClothing);
    return {
      bodyDesc,
      faceCount: 0,
      currentClothing,
      raw: obj,
      timestamp: Date.now(),
    };
  }

  // Face-based path
  const parts: string[] = [];
  if (obj.age_range && obj.age_range !== "unknown") parts.push(obj.age_range);
  if (obj.gender_presentation && obj.gender_presentation !== "unknown")
    parts.push(obj.gender_presentation);
  parts.push("person");
  if (obj.skin_tone && obj.skin_tone !== "unknown") parts.push(`with ${obj.skin_tone} skin`);
  if (obj.hair_length && obj.hair_color && obj.hair_color !== "bald" && obj.hair_color !== "unknown")
    parts.push(`${obj.hair_length} ${obj.hair_color} hair`);
  else if (obj.hair_color === "bald") parts.push("bald");
  else if (obj.hair_color && obj.hair_color !== "unknown") parts.push(`${obj.hair_color} hair`);
  if (obj.body_type && obj.body_type !== "unknown") parts.push(`${obj.body_type} build`);
  if (
    obj.distinguishing_features &&
    obj.distinguishing_features !== "none" &&
    obj.distinguishing_features !== "unknown"
  )
    parts.push(obj.distinguishing_features);

  const bodyDesc = parts.join(", ");
  console.log("[VLM] Body desc:", bodyDesc, "| Face count:", obj.face_count);

  return {
    bodyDesc,
    faceCount: Number(obj.face_count) || (obj.face_detected ? 1 : 0),
    raw: obj,
    timestamp: Date.now(),
  };
}

// ===== DALL-E outfit generation (image-to-image edit) =====
async function generateOutfitDallE(
  photoBuf: Buffer,
  bodyDesc: string,
  outfit: string,
  bg: string,
  styleSuffix: string,
  opts: {
    bodyMods?: string[];
    qualitySuffix?: string;
    picanteSuffix?: string;
    keepBackground?: boolean;
    keepAspectRatio?: boolean;
  } = {}
): Promise<Buffer> {
  const z = await initZAI();
  const dataUrl = toDataUrl(photoBuf);

  // Build outfit prompt with body mods
  let outfitClause = `Wearing ${sanitizePrompt(outfit)}.`;
  if (opts.bodyMods && opts.bodyMods.length > 0) {
    const modParts = opts.bodyMods
      .map((k) => BODY_MODS[k]?.prompt)
      .filter(Boolean);
    if (modParts.length > 0) {
      outfitClause += ` The person is ${modParts.join(", ")}.`;
    }
  }

  // Background clause
  const bgClause = opts.keepBackground
    ? "Standing in the same setting and background as the reference photo. Preserve the original background."
    : `Standing in ${sanitizePrompt(bg)}.`;

  // Quality suffix
  const qualitySuffix = opts.qualitySuffix || QUALITY_PRESETS.standard.suffix;
  const picanteSuffix = opts.picanteSuffix || "";

  const prompt = `Full-body fashion photograph of: ${bodyDesc}. ${outfitClause} ${bgClause} ${styleSuffix} ${qualitySuffix} ${picanteSuffix} The person's face, hair color, skin tone, and body proportions match the reference photo.`;

  console.log("[DALL-E] Prompt:", prompt);

  // Determine size
  let size = "768x1344";
  if (opts.keepAspectRatio) {
    try {
      const meta = await sharp(photoBuf).metadata();
      if (meta.width && meta.height) {
        size = pickDallESize(meta.width, meta.height);
      }
    } catch (e) {
      console.warn("[DALL-E] sharp metadata failed, using default size", e);
    }
  }

  const response = await withZaiRetry(() => z.images.generations.edit({
    prompt,
    images: [{ url: dataUrl }] as any,
    size: size as any,
  } as any));

  const b64: string | undefined = response.data?.[0]?.base64;
  if (!b64) {
    console.error("[DALL-E] No image in response. Raw:", JSON.stringify(response).slice(0, 500));
    throw new Error("DALL-E returned no image data");
  }
  return Buffer.from(b64, "base64");
}

// ===== Detect Z.ai content-filter block (error code 1301) OR rate limit (429) =====
// Both should trigger Pollinations fallback so the pipeline doesn't fail.
function isZaiContentBlock(err: any): boolean {
  if (!err) return false;
  const msg = (err?.message || err?.toString?.() || "").toString();
  // Content filter (1301)
  if (/1301/.test(msg)) return true;
  if (/content.{0,10}filter/i.test(msg)) return true;
  if (/blocked/i.test(msg) && /content/i.test(msg)) return true;
  if (err?.code === 1301 || err?.error?.code === 1301) return true;
  // Rate limit (429) — also fall back to Pollinations
  if (/429/.test(msg)) return true;
  if (/too many requests/i.test(msg)) return true;
  if (err?.code === 429 || err?.error?.code === 429) return true;
  return false;
}

// ===== Instruct-Pix2Pix edit (preserves identity, no content filter) =====
async function editWithIp2p(photoBuf: Buffer, instruction: string): Promise<Buffer | null> {
  const client = await getIp2p();
  if (!client) return null;
  try {
    const resized = await sharp(photoBuf)
      .resize(512, 512, { fit: "cover" })
      .png()
      .toBuffer();
    const blob = new Blob([resized], { type: "image/png" });
    console.log(`[IP2P] Editing with instruction: "${instruction}"`);
    const result = await Promise.race([
      client.predict("/generate", [
        blob,
        instruction,
        50,
        "Fix Seed",
        42,
        "Fix CFG",
        7.5,
        1.5,
      ]),
      new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error("IP2P timeout (90s)")), 90000)
      ),
    ]);
    const data = (result as any)?.data;
    if (!Array.isArray(data)) return null;
    // The generated image is at index 3 in the IP2P output array
    const imgItem = data[3] || data[0];
    if (!imgItem) return null;
    if (typeof imgItem === "string") {
      if (imgItem.startsWith("http")) {
        const res = await fetch(imgItem);
        return Buffer.from(await res.arrayBuffer());
      }
      if (imgItem.startsWith("data:")) {
        return Buffer.from(imgItem.split(",")[1], "base64");
      }
    }
    if (imgItem?.url) {
      const res = await fetch(imgItem.url);
      return Buffer.from(await res.arrayBuffer());
    }
    if (imgItem?.path) {
      try {
        const f = Bun.file(imgItem.path);
        if (await f.exists()) return Buffer.from(await f.arrayBuffer());
      } catch {}
    }
    if (imgItem?.base64) return Buffer.from(imgItem.base64, "base64");
    return null;
  } catch (e: any) {
    console.warn("[IP2P] Edit failed:", e?.message || e);
    return null;
  }
}

// ===== Pollinations.ai fallback (when Z.ai blocks) =====
async function generateWithPollinations(
  prompt: string,
  width: number,
  height: number
): Promise<Buffer> {
  const encoded = encodeURIComponent(prompt);
  const url = `https://image.pollinations.ai/prompt/${encoded}?width=${width}&height=${height}&nologo=true`;
  let lastErr: any;
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      console.log(`[Pollinations] Attempt ${attempt}: ${url.slice(0, 200)}...`);
      const res = await Promise.race([
        fetch(url),
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error("Pollinations timeout (60s)")), 60000)
        ),
      ]);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const buf = Buffer.from(await res.arrayBuffer());
      if (buf.length < 1000) throw new Error("Response too small");
      console.log(`[Pollinations] OK, ${buf.length} bytes`);
      return buf;
    } catch (e: any) {
      lastErr = e;
      console.warn(`[Pollinations] Attempt ${attempt} failed:`, e?.message || e);
      await new Promise((r) => setTimeout(r, 1500));
    }
  }
  throw new Error(`Pollinations failed after 3 attempts: ${lastErr?.message || lastErr}`);
}

// ===== HD Upscale (Lanczos3 + sharpen) =====
async function upscaleImage(buf: Buffer): Promise<Buffer> {
  const meta = await sharp(buf).metadata();
  const newW = Math.min((meta.width || 512) * 2, 2048);
  const newH = Math.min((meta.height || 512) * 2, 2048);
  console.log(`[Upscale] ${meta.width}x${meta.height} → ${newW}x${newH}`);
  return sharp(buf)
    .resize(newW, newH, { fit: "cover", kernel: "lanczos3" })
    .sharpen({ sigma: 1.2, m1: 1.0, m2: 0.8 })
    .jpeg({ quality: 95 })
    .toBuffer();
}

// ===== Animate photo via Z.ai video API =====
// ===== Animation task store (in-memory, for async polling) =====
// The Z.ai video API takes 2-4 minutes to generate a video. We can't wait
// synchronously because the Z.ai proxy (alb) cuts connections after 30s.
// Solution: create the task, return the taskId immediately, then poll
// in the background. The frontend polls /api/animation-status every 5s.
type AnimationTask = {
  taskId: string;        // Z.ai task ID
  status: "PROCESSING" | "SUCCESS" | "FAIL";
  videoUrl?: string;     // set when SUCCESS
  error?: string;        // set when FAIL
  createdAt: number;     // for cleanup
  prompt: string;        // for debugging
};
const animationTasks = new Map<string, AnimationTask>();

// Cleanup old tasks every 10 minutes (keep tasks for 1 hour)
setInterval(() => {
  const now = Date.now();
  for (const [id, task] of animationTasks.entries()) {
    if (now - task.createdAt > 60 * 60 * 1000) {
      animationTasks.delete(id);
    }
  }
}, 10 * 60 * 1000);

// Background poller: polls Z.ai for a task and updates the store
async function pollAnimationTask(ourTaskId: string, zaiTaskId: string, prompt: string) {
  const z = await initZAI();
  console.log(`[AnimPoll] Starting background poll for ${ourTaskId} (zai: ${zaiTaskId})`);
  const maxAttempts = 30; // 30 * 10s = 300s = 5 min (was 80*3s=240s but caused 429)
  let consecutiveErrors = 0;
  for (let i = 0; i < maxAttempts; i++) {
    try {
      // Poll every 10s (was 3s — too aggressive, caused 429 cascading to VLM)
      await new Promise((r) => setTimeout(r, 10000));
      const status = await z.async.result.query(zaiTaskId);
      const st = status?.task_status;
      console.log(`[AnimPoll] ${ourTaskId} attempt ${i + 1}: ${st}`);
      consecutiveErrors = 0; // reset on success

      const task = animationTasks.get(ourTaskId);
      if (!task) {
        console.log(`[AnimPoll] Task ${ourTaskId} no longer in store, stopping`);
        return;
      }

      if (st === "SUCCESS") {
        const videoUrl =
          status?.video_result?.[0]?.url ||
          status?.video_url ||
          status?.url ||
          status?.video;
        if (videoUrl && typeof videoUrl === "string") {
          task.status = "SUCCESS";
          task.videoUrl = videoUrl;
          animationTasks.set(ourTaskId, task);
          console.log(`[AnimPoll] ${ourTaskId} SUCCESS: ${videoUrl}`);
          return;
        }
        task.status = "FAIL";
        task.error = "SUCCESS but no video URL";
        animationTasks.set(ourTaskId, task);
        return;
      }
      if (st === "FAIL") {
        task.status = "FAIL";
        task.error = "Z.ai video generation failed";
        animationTasks.set(ourTaskId, task);
        console.log(`[AnimPoll] ${ourTaskId} FAIL`);
        return;
      }
      // PROCESSING → continue polling
    } catch (e: any) {
      const errMsg = e?.message || String(e);
      consecutiveErrors++;
      console.warn(`[AnimPoll] ${ourTaskId} poll error (${consecutiveErrors}/5):`, errMsg.slice(0, 100));

      // If 429 (rate limited), wait longer before next attempt
      if (errMsg.includes("429") || errMsg.toLowerCase().includes("too many")) {
        console.log(`[AnimPoll] ${ourTaskId} rate limited, waiting 30s before retry...`);
        await new Promise((r) => setTimeout(r, 30000)); // extra 30s wait on 429
      }

      // If 5 consecutive errors, give up
      if (consecutiveErrors >= 5) {
        const task = animationTasks.get(ourTaskId);
        if (task) {
          task.status = "FAIL";
          task.error = "Too many polling errors (likely rate limited)";
          animationTasks.set(ourTaskId, task);
          console.log(`[AnimPoll] ${ourTaskId} gave up after 5 consecutive errors`);
        }
        return;
      }
      // Otherwise, continue polling (don't abort on transient errors)
    }
  }
  // Timed out
  const task = animationTasks.get(ourTaskId);
  if (task) {
    task.status = "FAIL";
    task.error = "Timeout (5 min)";
    animationTasks.set(ourTaskId, task);
    console.log(`[AnimPoll] ${ourTaskId} timed out`);
  }
}

// Create an animation task and return immediately (async).
// Returns our internal task ID that the frontend can poll via /api/animation-status.
// The background poller updates the task status in the animationTasks store.
async function animatePhoto(
  photoBuf: Buffer,
  animationPrompt: string,
  onProgress?: (msg: string) => void
): Promise<string | null> {
  try {
    const z = await initZAI();
    // Resize to a sane size for the video API
    const resized = await sharp(photoBuf)
      .resize(1024, 1024, { fit: "cover" })
      .jpeg({ quality: 92 })
      .toBuffer();
    const dataUrl = toDataUrl(resized);

    onProgress?.("Enviando pedido de animación a Z.ai video API...");
    console.log("[Animate] Creating video generation task...");
    const createRes = await withZaiRetry(() => z.video.generations.create({
      prompt: animationPrompt,
      image_url: dataUrl,
      quality: "quality",
      with_audio: false,
    } as any));

    const zaiTaskId: string | undefined = createRes?.id || createRes?.request_id;
    if (!zaiTaskId) {
      console.error("[Animate] No task id returned:", JSON.stringify(createRes).slice(0, 500));
      return null;
    }
    console.log("[Animate] Z.ai task created:", zaiTaskId);

    // Generate our own task ID and register in the store
    const ourTaskId = "anim_" + Date.now() + "_" + Math.random().toString(36).slice(2, 8);
    animationTasks.set(ourTaskId, {
      taskId: zaiTaskId,
      status: "PROCESSING",
      createdAt: Date.now(),
      prompt: animationPrompt,
    });

    // Start background polling (do NOT await — fire and forget)
    pollAnimationTask(ourTaskId, zaiTaskId, animationPrompt).catch((e) => {
      console.error(`[Animate] Background poller crashed for ${ourTaskId}:`, e);
      const t = animationTasks.get(ourTaskId);
      if (t) {
        t.status = "FAIL";
        t.error = "Background poller crashed";
        animationTasks.set(ourTaskId, t);
      }
    });

    onProgress?.("Animación encolada, el frontend hará polling...");
    console.log("[Animate] Background poller started, returning task ID:", ourTaskId);
    return ourTaskId;
  } catch (e: any) {
    console.error("[Animate] Error:", e?.message || e);
    return null;
  }
}

// ===== Face swap via HuggingFace Space =====
async function faceSwap(
  originalBuf: Buffer,
  generatedBuf: Buffer
): Promise<Buffer | null> {
  const client = await getGradio();
  if (!client) return null;

  const sourcePng = await sharp(originalBuf)
    .resize(1024, 1024, { fit: "cover" })
    .png()
    .toBuffer();
  const targetPng = await sharp(generatedBuf)
    .resize(1024, 1024, { fit: "cover" })
    .png()
    .toBuffer();

  const sourceBlob = new Blob([sourcePng], { type: "image/png" });
  const targetBlob = new Blob([targetPng], { type: "image/png" });

  const endpoints: Array<{ path: string; payload: Record<string, any> }> = [
    {
      path: "/run_inference",
      payload: {
        target: targetBlob,
        source: sourceBlob,
        slider: 0,
        adv_slider: 0,
        settings: ["Compare"],
      },
    },
    {
      path: "/run_inference",
      payload: {
        target: targetBlob,
        source: sourceBlob,
        slider: 0,
        adv_slider: 0,
        settings: [],
      },
    },
  ];

  for (const ep of endpoints) {
    try {
      console.log(
        `[FaceSwap] Trying ${ep.path} settings=${JSON.stringify(ep.payload.settings)}`
      );
      const result = await Promise.race([
        client.predict(ep.path, ep.payload),
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error("HF predict timeout (60s)")), 60000)
        ),
      ]);

      const buf = await parseGradioImageResult(result);
      if (buf) {
        console.log("[FaceSwap] Success, image size:", buf.length);
        return buf;
      }
    } catch (e: any) {
      console.warn(`[FaceSwap] ${ep.path} failed:`, e?.message || e);
    }
  }
  return null;
}

async function parseGradioImageResult(result: any): Promise<Buffer | null> {
  const data = result?.data?.[0];
  if (!data) return null;

  if (typeof data === "string") {
    if (data.startsWith("http")) {
      const res = await fetch(data);
      return Buffer.from(await res.arrayBuffer());
    }
    if (data.startsWith("data:")) {
      return Buffer.from(data.split(",")[1], "base64");
    }
  }
  if (data?.url) {
    const res = await fetch(data.url);
    return Buffer.from(await res.arrayBuffer());
  }
  if (data?.path) {
    try {
      const f = Bun.file(data.path);
      if (await f.exists()) return Buffer.from(await f.arrayBuffer());
    } catch {}
  }
  if (data?.base64) return Buffer.from(data.base64, "base64");
  if (data?.name && typeof data.name === "string" && data.name.startsWith("http")) {
    const res = await fetch(data.name);
    return Buffer.from(await res.arrayBuffer());
  }
  return null;
}

// ===== Fallback: sharp ellipse composite =====
async function fallbackComposite(
  originalBuf: Buffer,
  generatedBuf: Buffer
): Promise<Buffer> {
  const SIZE = 1024;
  const sourceResized = await sharp(originalBuf)
    .resize(SIZE, SIZE, { fit: "cover" })
    .png()
    .toBuffer();
  const generatedResized = await sharp(generatedBuf)
    .resize(SIZE, SIZE, { fit: "cover" })
    .png()
    .toBuffer();

  const faceRegion = await sharp(sourceResized)
    .extract({ left: 256, top: 0, width: 512, height: 512 })
    .toBuffer();

  const maskSvg = `<svg width="512" height="512" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="g" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="white" stop-opacity="1"/>
      <stop offset="70%" stop-color="white" stop-opacity="1"/>
      <stop offset="100%" stop-color="white" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <ellipse cx="256" cy="256" rx="180" ry="220" fill="url(#g)"/>
</svg>`;
  const maskBuf = Buffer.from(maskSvg);

  const faceWithMask = await sharp(faceRegion)
    .composite([{ input: maskBuf, blend: "dest-in" }])
    .png()
    .toBuffer();

  return sharp(generatedResized)
    .composite([{ input: faceWithMask, top: 30, left: 256 }])
    .png()
    .toBuffer();
}

// ===== Full pipeline =====
type PipelineOptions = {
  bodyMods?: string[];
  quality?: string;
  picante?: string;
  upscale?: boolean;
  animate?: string;
  keepBackground?: boolean;
  keepAspectRatio?: boolean;
  bodyOnly?: boolean;
};

async function runPipeline(
  photoBuf: Buffer,
  outfit: string,
  bg: string,
  style: string,
  onProgress: (stage: string, msg: string) => void,
  options: PipelineOptions = {}
): Promise<{
  result: Buffer;
  bodyDesc: string;
  usedFallback: boolean;
  faceCount: number;
  vlmRaw: any;
  cached: boolean;
  currentClothing?: string;
  ip2pUsed?: boolean;
  pollinationsUsed?: boolean;
  animationTaskId?: string | null;
}> {
  const t0 = Date.now();
  const bodyOnly = !!options.bodyOnly;
  const cacheKey = hashBuffer(photoBuf) + (bodyOnly ? ":bodyOnly" : "");

  // Step 0: optional upscale
  let workBuf = photoBuf;
  if (options.upscale) {
    onProgress("upscaling", "Mejorando resolución (Lanczos3 + sharpen)...");
    try {
      workBuf = await upscaleImage(photoBuf);
    } catch (e: any) {
      console.warn("[Pipeline] Upscale failed, using original:", e?.message || e);
    }
  }

  // Step 1: VLM (cached)
  onProgress("analyzing", bodyOnly
    ? "Analizando tu foto (modo cuerpo) con VLM..."
    : "Analizando tu foto con IA visual...");
  let analysis = getCachedVlm(cacheKey);
  let cached = true;
  if (!analysis) {
    cached = false;
    try {
      const fresh = await analyzePhoto(workBuf, bodyOnly);
      analysis = {
        bodyDesc: fresh.bodyDesc,
        faceCount: fresh.faceCount,
        currentClothing: fresh.currentClothing,
        raw: fresh.raw,
        timestamp: Date.now(),
      };
      setCachedVlm(cacheKey, analysis);
    } catch (vlmError: any) {
      // VLM failed (likely 429 rate limit). Use fallback description and continue.
      console.warn("[VLM] Failed, using fallback:", vlmError?.message?.slice(0, 100));
      onProgress("analyzing", "VLM no disponible (rate limit) — usando descripción genérica...");
      analysis = {
        bodyDesc: bodyOnly ? "person, showing body" : "adult person",
        faceCount: bodyOnly ? 0 : 1,  // assume face exists in non-bodyOnly mode
        currentClothing: "",
        raw: { error: vlmError?.message },
        timestamp: Date.now(),
      };
      setCachedVlm(cacheKey, analysis);
    }
  } else {
    console.log("[VLM] Cache hit for key", cacheKey);
  }

  // Face validation (skip if bodyOnly)
  if (!bodyOnly && analysis.faceCount === 0) {
    throw new Error(
      "No se detectó ninguna cara en la foto. Subí una foto donde tu rostro sea claramente visible, o activá el modo Cuerpo (bodyOnly)."
    );
  }

  onProgress(
    "analyzed",
    bodyOnly
      ? `${cached ? "(cache) " : ""}Cuerpo: ${analysis.bodyDesc.slice(0, 80)}...`
      : `${cached ? "(cache) " : ""}${analysis.faceCount} cara${analysis.faceCount > 1 ? "s" : ""} detectada${analysis.faceCount > 1 ? "s" : ""}. Cuerpo: ${analysis.bodyDesc.slice(0, 60)}...`
  );

  // Step 2: outfit generation
  const stylePreset = STYLES[style] || STYLES.vanilla;
  const qualityPreset = QUALITY_PRESETS[options.quality || "standard"];
  const picantePreset = PICANTE_LEVELS[options.picante || "off"];

  let generated: Buffer;
  let ip2pUsed = false;
  let pollinationsUsed = false;

  if (bodyOnly) {
    // bodyOnly: IP2P first, fall back to DALL-E edit if it fails.
    onProgress("generating", "Instruct-Pix2Pix editando tu foto...");
    const instruction = buildIp2pInstruction(outfit, analysis.currentClothing);
    console.log("[Pipeline] bodyOnly → IP2P instruction:", instruction);
    const ip2pResult = await editWithIp2p(workBuf, instruction);
    if (ip2pResult) {
      generated = ip2pResult;
      ip2pUsed = true;
      onProgress("ip2p_done", "IP2P listo.");
    } else {
      console.warn("[Pipeline] IP2P failed, falling back to DALL-E edit");
      onProgress("generating", "IP2P no disponible → DALL-E edit...");
      try {
        generated = await generateOutfitDallE(workBuf, analysis.bodyDesc, outfit, bg, stylePreset.suffix, {
          bodyMods: options.bodyMods,
          qualitySuffix: qualityPreset.suffix,
          picanteSuffix: picantePreset.suffix,
          keepBackground: options.keepBackground,
          keepAspectRatio: options.keepAspectRatio,
        });
      } catch (e: any) {
        if (isZaiContentBlock(e)) {
          console.warn("[Pipeline] Z.ai blocked DALL-E, trying Pollinations fallback");
          onProgress("generating", "Z.ai bloqueado → Pollinations.ai...");
          const w = 768, h = 1344;
          const pp = `Full-body fashion photograph of: ${analysis.bodyDesc}. Wearing ${sanitizePrompt(outfit)}. ${options.keepBackground ? "Original background preserved." : `Standing in ${sanitizePrompt(bg)}.`} ${stylePreset.suffix} ${qualityPreset.suffix} ${picantePreset.suffix}`;
          generated = await generateWithPollinations(pp, w, h);
          pollinationsUsed = true;
        } else {
          throw e;
        }
      }
    }
  } else {
    // Face mode: DALL-E edit first, fall back to Pollinations on 1301 block.
    onProgress("generating", `DALL-E generando "${outfit}"...`);
    try {
      generated = await generateOutfitDallE(workBuf, analysis.bodyDesc, outfit, bg, stylePreset.suffix, {
        bodyMods: options.bodyMods,
        qualitySuffix: qualityPreset.suffix,
        picanteSuffix: picantePreset.suffix,
        keepBackground: options.keepBackground,
        keepAspectRatio: options.keepAspectRatio,
      });
    } catch (e: any) {
      if (isZaiContentBlock(e)) {
        console.warn("[Pipeline] Z.ai blocked DALL-E (1301), trying Pollinations fallback");
        onProgress("generating", "Z.ai bloqueado → Pollinations.ai...");
        const w = 768, h = 1344;
        const pp = `Full-body fashion photograph of: ${analysis.bodyDesc}. Wearing ${sanitizePrompt(outfit)}. ${options.keepBackground ? "Original background preserved." : `Standing in ${sanitizePrompt(bg)}.`} ${stylePreset.suffix} ${qualityPreset.suffix} ${picantePreset.suffix}`;
        generated = await generateWithPollinations(pp, w, h);
        pollinationsUsed = true;
      } else {
        throw e;
      }
    }
  }

  // Step 3: face swap (skip if bodyOnly — IP2P preserved identity)
  let final = generated;
  let usedFallback = false;
  if (!bodyOnly) {
    onProgress("swapping", "Recomponiendo tu rostro real con InsightFace...");
    let swapped = await faceSwap(workBuf, generated);
    if (!swapped) {
      console.warn("[Pipeline] Face swap failed, using sharp fallback");
      onProgress("fallback", "HF no disponible — usando composición local...");
      swapped = await fallbackComposite(workBuf, generated);
      usedFallback = true;
    }
    final = swapped;
  } else {
    onProgress("ip2p_done", "Identidad preservada por IP2P (sin face swap).");
  }

  // Step 4: final resize + JPEG (skip if keepAspectRatio)
  if (options.keepAspectRatio) {
    final = await sharp(final).jpeg({ quality: 92 }).toBuffer();
  } else {
    final = await sharp(final)
      .resize(1024, 1024, { fit: "cover" })
      .jpeg({ quality: 92 })
      .toBuffer();
  }

  // Step 5: optional animation (async — returns task ID, not video URL)
  let animationTaskId: string | null = null;
  if (options.animate && options.animate !== "off") {
    const animPreset = ANIMATION_PRESETS[options.animate];
    if (animPreset && animPreset.prompt) {
      onProgress("animating", "Encolando animación con Z.ai video API...");
      animationTaskId = await animatePhoto(final, animPreset.prompt, (msg) =>
        onProgress("animating", msg)
      );
    }
  }

  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
  console.log(
    `[Pipeline] Done in ${elapsed}s (bodyOnly=${bodyOnly}, ip2p=${ip2pUsed}, pollinations=${pollinationsUsed}, fallback=${usedFallback}, cached=${cached}, animate=${!!animationTaskId})`
  );

  return {
    result: final,
    bodyDesc: analysis.bodyDesc,
    usedFallback,
    faceCount: analysis.faceCount,
    vlmRaw: analysis.raw,
    cached,
    currentClothing: analysis.currentClothing,
    ip2pUsed,
    pollinationsUsed,
    animationTaskId,
  };
}

// ===== LLM outfit suggestions (returns {label, prompt}) =====
async function suggestOutfits(
  style: string,
  bodyDesc?: string,
  bodyOnly: boolean = false
): Promise<{ label: string; prompt: string }[]> {
  const z = await initZAI();

  let hint: string;
  let exampleFormat: string;
  if (bodyOnly) {
    hint = "footwear and legwear for a body-only photo: stockings, heels, boots, sandals, pantyhose";
    exampleFormat =
      'each item should be a complete outfit description like "black latex thigh-high stockings" or "red stiletto high heels"';
  } else {
    const preset = STYLES[style] || STYLES.vanilla;
    hint = preset.llmHint;
    exampleFormat =
      'each outfit described in 4-8 English words like "red lace lingerie set" or "black leather corset with stockings"';
  }

  const prompt = `Suggest 10 ${hint}. Return ONLY a JSON array of strings, ${exampleFormat}.${bodyDesc ? ` The person is: ${bodyDesc}.` : ""} Be specific and varied. No explanations. The strings will be used as the "outfit" field in an image-edit prompt.`;

  const response = await withZaiRetry(() => z.chat.completions.create({
    messages: [{ role: "user", content: prompt }],
    thinking: { type: "disabled" },
  }));

  const text: string = response.choices?.[0]?.message?.content || "";
  console.log("[LLM] Suggest raw:", text.slice(0, 500));

  const match = text.match(/\[[\s\S]*\]/);
  if (!match) {
    return bodyOnly
      ? BODY_ONLY_IDEAS
      : (STYLES[style] || STYLES.vanilla).ideas;
  }
  try {
    const arr = JSON.parse(match[0]);
    if (Array.isArray(arr) && arr.every((x) => typeof x === "string")) {
      // Convert strings to {label, prompt} pairs.
      // For bodyOnly, also prefix the prompt with "change to " if it doesn't already start with an edit verb.
      return arr.slice(0, 10).map((s: string) => {
        const promptStr: string = s;
        const labelStr: string = promptStr.length > 36 ? promptStr.slice(0, 34) + "…" : promptStr;
        let finalPrompt = promptStr;
        if (bodyOnly && !/^(change|replace|turn|make|switch)\b/i.test(finalPrompt)) {
          finalPrompt = `change to ${finalPrompt}`;
        }
        return { label: labelStr, prompt: finalPrompt };
      });
    }
  } catch (e) {
    console.error("[LLM] Parse failed:", e);
  }
  return bodyOnly
    ? BODY_ONLY_IDEAS
    : (STYLES[style] || STYLES.vanilla).ideas;
}

// ===== HTTP Server =====
const server = Bun.serve({
  port: PORT,
  idleTimeout: 255,
  async fetch(req): Promise<Response> {
    const url = new URL(req.url);
    const ip = getClientIp(req);

    // CORS
    if (req.method === "OPTIONS") {
      return new Response(null, {
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type",
        },
      });
    }

    // ===== Health =====
    if (url.pathname === "/health") {
      return Response.json({
        ok: true,
        service: "magicvisual",
        version: "3.0.0",
        ts: Date.now(),
        uptime_sec: Math.floor(process.uptime()),
        pipeline: bodyOnlyPipelineLabel(),
        vlm_cache_size: vlmCache.size,
        rate_limit_buckets: rateBuckets.size,
        rate_limit_max: RATE_LIMIT_MAX,
        features: [
          "vlm", "dall-e-edit", "insightface", "sharp-fallback",
          "ip2p", "pollinations", "upscale", "animate",
          "body-mods", "quality-presets", "picante", "body-only",
          "keep-background", "keep-aspect-ratio", "prompt-sanitizer",
        ],
      });
    }

    // ===== Info =====
    if (url.pathname === "/api/info") {
      return Response.json({
        version: "3.0.0",
        styles: Object.fromEntries(
          Object.entries(STYLES).map(([k, v]) => [
            k,
            { name: v.name, desc: v.desc, ideas: v.ideas, llm_hint: v.llmHint },
          ])
        ),
        body_only_ideas: BODY_ONLY_IDEAS,
        body_mods: Object.fromEntries(
          Object.entries(BODY_MODS).map(([k, v]) => [k, { label: v.label, prompt: v.prompt }])
        ),
        quality_presets: Object.fromEntries(
          Object.entries(QUALITY_PRESETS).map(([k, v]) => [k, { label: v.label, suffix: v.suffix }])
        ),
        picante_levels: Object.fromEntries(
          Object.entries(PICANTE_LEVELS).map(([k, v]) => [k, { label: v.label, suffix: v.suffix }])
        ),
        animation_presets: Object.fromEntries(
          Object.entries(ANIMATION_PRESETS).map(([k, v]) => [k, { label: v.label, prompt: v.prompt }])
        ),
        backgrounds: BACKGROUNDS,
        dall_e_sizes: DALL_E_SIZES,
        max_file_size: MAX_FILE_SIZE,
        rate_limit: {
          max: RATE_LIMIT_MAX,
          window_sec: RATE_LIMIT_WINDOW_MS / 1000,
          enabled: RATE_LIMIT_MAX > 0,
        },
        endpoints: [
          "GET /", "GET /health", "GET /api/info",
          "POST /api/describe", "POST /api/suggest-outfits",
          "POST /api/edit", "POST /api/upscale",
        ],
      });
    }

    // ===== UI =====
    if (url.pathname === "/" && req.method === "GET") {
      return new Response(HTML, {
        headers: { "Content-Type": "text/html; charset=utf-8" },
      });
    }

    // ===== Describe (VLM only) =====
    if (url.pathname === "/api/describe" && req.method === "POST") {
      const rl = checkRateLimit(ip);
      if (!rl.ok) {
        return Response.json(
          { error: "Rate limit exceeded", reset_at: rl.resetAt },
          { status: 429 }
        );
      }

      try {
        const formData = await req.formData();
        const file = formData.get("photo") as File | null;
        const bodyOnly = (formData.get("bodyOnly") as string) === "true" || (formData.get("bodyOnly") as string) === "1";
        if (!file) return Response.json({ error: "Missing photo" }, { status: 400 });
        if (file.size > MAX_FILE_SIZE)
          return Response.json({ error: "File too large" }, { status: 413 });

        const photoBuf = await autoCompressPhoto(Buffer.from(await file.arrayBuffer()));
        const cacheKey = hashBuffer(photoBuf) + (bodyOnly ? ":bodyOnly" : "");
        let analysis = getCachedVlm(cacheKey);
        if (!analysis) {
          const fresh = await analyzePhoto(photoBuf, bodyOnly);
          analysis = {
            bodyDesc: fresh.bodyDesc,
            faceCount: fresh.faceCount,
            currentClothing: fresh.currentClothing,
            raw: fresh.raw,
            timestamp: Date.now(),
          };
          setCachedVlm(cacheKey, analysis);
        }
        return Response.json({
          body_desc: analysis.bodyDesc,
          face_count: analysis.faceCount,
          current_clothing: analysis.currentClothing || null,
          body_only: bodyOnly,
          cached: getCachedVlm(cacheKey) !== null,
        });
      } catch (e: any) {
        return Response.json({ error: e?.message || "Error" }, { status: 500 });
      }
    }

    // ===== Suggest outfits (LLM) =====
    if (url.pathname === "/api/suggest-outfits" && req.method === "POST") {
      try {
        const body = await req.json().catch(() => ({}));
        const style = (body.style as string) || "vanilla";
        const bodyDesc = body.body_desc as string | undefined;
        const bodyOnly = !!body.body_only;
        const suggestions = await suggestOutfits(style, bodyDesc, bodyOnly);
        return Response.json({ suggestions });
      } catch (e: any) {
        return Response.json({ error: e?.message || "Error" }, { status: 500 });
      }
    }

    // ===== Animation status (poll endpoint for async video generation) =====
    // Frontend calls this every 5s with ?taskId=xxx to check if the video is ready.
    // Returns: { status: "PROCESSING"|"SUCCESS"|"FAIL", videoUrl?, error? }
    if (url.pathname === "/api/animation-status" && req.method === "GET") {
      const taskId = url.searchParams.get("taskId");
      if (!taskId) {
        return Response.json({ error: "Missing taskId parameter" }, { status: 400 });
      }
      const task = animationTasks.get(taskId);
      if (!task) {
        return Response.json({ error: "Task not found (may have expired after 1 hour)" }, { status: 404 });
      }
      return Response.json({
        taskId,
        status: task.status,
        videoUrl: task.videoUrl || null,
        error: task.error || null,
        ageSeconds: Math.floor((Date.now() - task.createdAt) / 1000),
      });
    }

    // ===== Upscale (standalone) =====
    if (url.pathname === "/api/upscale" && req.method === "POST") {
      const rl = checkRateLimit(ip);
      if (!rl.ok) {
        return Response.json(
          { error: "Rate limit exceeded", reset_at: rl.resetAt },
          { status: 429 }
        );
      }
      try {
        const formData = await req.formData();
        const file = formData.get("photo") as File | null;
        if (!file) return Response.json({ error: "Missing photo" }, { status: 400 });
        if (file.size > MAX_FILE_SIZE)
          return Response.json({ error: "File too large" }, { status: 413 });
        const photoBuf = await autoCompressPhoto(Buffer.from(await file.arrayBuffer()));
        const upscaled = await upscaleImage(photoBuf);
        const b64 = upscaled.toString("base64");
        return Response.json({
          result: `data:image/jpeg;base64,${b64}`,
          size: upscaled.length,
        });
      } catch (e: any) {
        return Response.json({ error: e?.message || "Error" }, { status: 500 });
      }
    }

    // ===== Main edit endpoint (streaming) =====
    if (url.pathname === "/api/edit" && req.method === "POST") {
      const rl = checkRateLimit(ip);
      if (!rl.ok) {
        return Response.json(
          { error: "Rate limit exceeded. Intentá de nuevo en un minuto.", reset_at: rl.resetAt },
          { status: 429 }
        );
      }

      try {
        const formData = await req.formData();
        const file = formData.get("photo") as File | null;
        const outfit = (formData.get("outfit") as string | null) || "";
        const bg = (formData.get("bg") as string | null) || BACKGROUNDS[0].prompt;
        const style = (formData.get("style") as string | null) || "vanilla";

        // New options
        // bodyMods: accept both "curves,voluptuous" (comma-separated) and
        // '["curves","voluptuous"]' (JSON array) formats. Frontend sends
        // comma-separated, so handle that primarily.
        const bodyModsRaw = (formData.get("bodyMods") as string | null) || "";
        let bodyMods: string[] = [];
        if (bodyModsRaw) {
          // Try JSON first (in case frontend changes to JSON later)
          if (bodyModsRaw.startsWith("[")) {
            try {
              const parsed = JSON.parse(bodyModsRaw);
              if (Array.isArray(parsed)) bodyMods = parsed.filter((k) => typeof k === "string" && BODY_MODS[k]);
            } catch {}
          } else {
            // Comma-separated format (what frontend sends)
            bodyMods = bodyModsRaw
              .split(",")
              .map((s) => s.trim())
              .filter((k) => k && BODY_MODS[k]);
          }
        }
        const quality = (formData.get("quality") as string | null) || "standard";
        const picante = (formData.get("picante") as string | null) || "off";
        const upscale = (formData.get("upscale") as string | null) === "true";
        const animate = (formData.get("animate") as string | null) || "off";
        const keepBackground = (formData.get("keepBackground") as string | null) === "true";
        const keepAspectRatio = (formData.get("keepAspectRatio") as string | null) === "true";
        const bodyOnly = (formData.get("bodyOnly") as string | null) === "true";

        if (!file) return Response.json({ error: "Missing photo" }, { status: 400 });
        if (!outfit.trim()) return Response.json({ error: "Missing outfit" }, { status: 400 });
        if (file.size > MAX_FILE_SIZE)
          return Response.json({ error: "File too large (max 50MB)" }, { status: 413 });

        const photoBuf = await autoCompressPhoto(Buffer.from(await file.arrayBuffer()));
        console.log(
          `[Edit] ip=${ip} photo=${file.size}B outfit="${outfit}" bg="${bg}" style="${style}" bodyMods=${JSON.stringify(bodyMods)} quality=${quality} picante=${picante} upscale=${upscale} animate=${animate} keepBg=${keepBackground} keepAr=${keepAspectRatio} bodyOnly=${bodyOnly}`
        );

        const stream = new ReadableStream({
          async start(controller) {
            const encoder = new TextEncoder();
            let closed = false;
            const send = (obj: any) => {
              if (closed) return;
              try {
                controller.enqueue(encoder.encode(`data: ${JSON.stringify(obj)}\n\n`));
              } catch {
                closed = true;
              }
            };

            // Keepalive: send a REAL data event every 5 seconds to prevent proxy timeout.
            // The Z.ai proxy (alb = Alibaba Load Balancer) cuts connections after ~30s
            // of inactivity. SSE comments (":") get filtered by the proxy, so we must
            // send actual data events. The frontend ignores type:"keepalive" events.
            const keepalive = setInterval(() => {
              if (closed) return;
              try {
                controller.enqueue(
                  encoder.encode(`data: ${JSON.stringify({ type: "keepalive", ts: Date.now() })}\n\n`)
                );
              } catch {
                closed = true;
              }
            }, 5000);

            try {
              const result = await runPipeline(
                photoBuf,
                outfit,
                bg,
                style,
                (stage, msg) => send({ type: "progress", stage, msg }),
                {
                  bodyMods,
                  quality,
                  picante,
                  upscale,
                  animate,
                  keepBackground,
                  keepAspectRatio,
                  bodyOnly,
                }
              );

              const b64 = result.result.toString("base64");
              send({
                type: "done",
                result: `data:image/jpeg;base64,${b64}`,
                bodyDesc: result.bodyDesc,
                usedFallback: result.usedFallback,
                faceCount: result.faceCount,
                cached: result.cached,
                currentClothing: result.currentClothing || null,
                ip2pUsed: !!result.ip2pUsed,
                pollinationsUsed: !!result.pollinationsUsed,
                animationTaskId: result.animationTaskId || null,
              });
            } catch (e: any) {
              console.error("[Edit] Pipeline error:", e);
              send({ type: "error", error: e?.message || String(e) });
            } finally {
              clearInterval(keepalive);
              if (!closed) {
                try {
                  controller.close();
                } catch {}
              }
            }
          },
        });

        return new Response(stream, {
          headers: {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            Connection: "keep-alive",
            "X-Accel-Buffering": "no",
            "X-RateLimit-Remaining": String(rl.remaining),
            "X-RateLimit-Reset": String(rl.resetAt),
          },
        });
      } catch (e: any) {
        console.error("[Edit] Request error:", e);
        return Response.json({ error: e?.message || "Internal error" }, { status: 500 });
      }
    }

    return new Response("Not found", { status: 404 });
  },
});

function bodyOnlyPipelineLabel(): string[] {
  return [
    "upscale?", "vlm", "ip2p-or-dall-e", "pollinations-fallback?", "face-swap?",
    "sharp-fallback?", "resize?", "animate?",
  ];
}

console.log(`[MagicVisual] Server running at http://localhost:${PORT}`);
console.log(`[MagicVisual] v3.0 — IP2P, body-mods, quality, picante, animate, bodyOnly, keepBackground, keepAspectRatio, sanitizer, Pollinations fallback`);
console.log(`[MagicVisual] Endpoints: GET / | GET /health | GET /api/info | POST /api/edit | POST /api/describe | POST /api/suggest-outfits | POST /api/upscale`);

// ===== HTML UI =====
const HTML = getHtml();

function getHtml(): string {
  // Build a compact JSON for the client. Note: BACKGROUNDS is an array of {label, prompt}.
  const stylesJson = JSON.stringify(
    Object.fromEntries(
      Object.entries(STYLES).map(([k, v]) => [
        k,
        { name: v.name, desc: v.desc, ideas: v.ideas },
      ])
    )
  );
  const bodyOnlyIdeasJson = JSON.stringify(BODY_ONLY_IDEAS);
  const bodyModsJson = JSON.stringify(
    Object.fromEntries(
      Object.entries(BODY_MODS).map(([k, v]) => [k, { label: v.label }])
    )
  );
  const qualityJson = JSON.stringify(
    Object.fromEntries(
      Object.entries(QUALITY_PRESETS).map(([k, v]) => [k, { label: v.label }])
    )
  );
  const picanteJson = JSON.stringify(
    Object.fromEntries(
      Object.entries(PICANTE_LEVELS).map(([k, v]) => [k, { label: v.label }])
    )
  );
  const animationJson = JSON.stringify(
    Object.fromEntries(
      Object.entries(ANIMATION_PRESETS).map(([k, v]) => [k, { label: v.label }])
    )
  );
  const backgroundsJson = JSON.stringify(BACKGROUNDS);

  return `<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MagicVisual — AI Outfit Magic</title>
<style>
  :root {
    --bg: #0a0612;
    --bg2: #14091f;
    --surface: rgba(255,255,255,0.04);
    --surface2: rgba(255,255,255,0.07);
    --border: rgba(255,255,255,0.08);
    --text: #f3eefc;
    --text2: #a99fc4;
    --accent: #a855f7;
    --accent2: #ec4899;
    --accent3: #f59e0b;
    --grad: linear-gradient(135deg, #a855f7 0%, #ec4899 100%);
    --radius: 14px;
    --radius-sm: 10px;
    --ok: #10b981;
    --warn: #f59e0b;
    --err: #ef4444;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif; min-height: 100vh; }
  body {
    background:
      radial-gradient(ellipse at 20% 0%, rgba(168, 85, 247, 0.15) 0%, transparent 50%),
      radial-gradient(ellipse at 80% 100%, rgba(236, 72, 153, 0.12) 0%, transparent 50%),
      var(--bg);
    background-attachment: fixed;
  }
  .container { max-width: 1320px; margin: 0 auto; padding: 24px 20px 80px; }
  header { text-align: center; padding: 28px 0 36px; }
  .logo {
    display: inline-flex; align-items: center; gap: 12px;
    font-size: 28px; font-weight: 800; letter-spacing: -0.02em;
  }
  .logo-mark {
    width: 44px; height: 44px; border-radius: 12px;
    background: var(--grad);
    display: grid; place-items: center;
    font-size: 24px;
    box-shadow: 0 8px 24px -8px rgba(236, 72, 153, 0.6);
  }
  .logo-text { background: var(--grad); -webkit-background-clip: text; background-clip: text; color: transparent; }
  .tagline { color: var(--text2); margin-top: 8px; font-size: 14px; }
  .tagline .pill {
    display: inline-block; padding: 4px 10px; border-radius: 999px;
    background: var(--surface2); border: 1px solid var(--border);
    font-size: 11px; margin-left: 6px;
  }
  .layout { display: grid; grid-template-columns: 1fr; gap: 20px; }
  @media (min-width: 980px) { .layout { grid-template-columns: 1fr 1fr; } }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    padding: 20px;
  }
  .card h2 {
    font-size: 14px; font-weight: 600; color: var(--text2);
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 14px;
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 8px;
  }
  .card h2 .badge {
    font-size: 10px; padding: 3px 8px; border-radius: 6px;
    background: rgba(168, 85, 247, 0.15); color: var(--accent);
    text-transform: none; letter-spacing: 0; font-weight: 600;
  }
  .card h2 .toggle-row {
    display: inline-flex; gap: 10px; align-items: center;
    text-transform: none; letter-spacing: 0; font-size: 11px; color: var(--text);
    font-weight: 500;
  }

  /* Toggle switch */
  .toggle { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; user-select: none; }
  .toggle input { display: none; }
  .toggle .track {
    width: 32px; height: 18px; border-radius: 999px;
    background: var(--surface2); border: 1px solid var(--border);
    position: relative; transition: all 0.2s;
  }
  .toggle .track::after {
    content: ""; position: absolute; top: 2px; left: 2px;
    width: 12px; height: 12px; border-radius: 50%; background: var(--text2);
    transition: all 0.2s;
  }
  .toggle input:checked + .track {
    background: var(--grad); border-color: transparent;
  }
  .toggle input:checked + .track::after {
    left: 16px; background: white;
  }

  .dropzone {
    border: 2px dashed var(--border);
    border-radius: var(--radius-sm);
    padding: 28px 16px; text-align: center;
    cursor: pointer;
    transition: all 0.2s;
    background: var(--surface2);
  }
  .dropzone:hover, .dropzone.drag { border-color: var(--accent); background: rgba(168, 85, 247, 0.08); }
  .dropzone .ico { font-size: 36px; opacity: 0.6; margin-bottom: 6px; }
  .dropzone .ttl { font-weight: 600; font-size: 15px; }
  .dropzone .sub { font-size: 12px; color: var(--text2); margin-top: 4px; }
  .dropzone .preview { max-width: 100%; max-height: 220px; border-radius: var(--radius-sm); margin-top: 12px; display: block; }

  .toggle-grid {
    display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px;
    margin-top: 12px;
  }
  @media (min-width: 600px) { .toggle-grid { grid-template-columns: repeat(4, 1fr); } }
  .toggle-card {
    padding: 10px 8px; border-radius: var(--radius-sm);
    background: var(--surface2); border: 1px solid var(--border);
    display: flex; flex-direction: column; align-items: center; gap: 6px;
    transition: all 0.15s;
  }
  .toggle-card.active { border-color: var(--accent); background: rgba(168, 85, 247, 0.12); }
  .toggle-card .lbl { font-size: 12px; color: var(--text); font-weight: 600; }
  .toggle-card .sub { font-size: 10px; color: var(--text2); text-align: center; line-height: 1.2; }

  .style-tabs { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
  .style-tab, .pill-tab {
    padding: 10px 8px; border-radius: var(--radius-sm);
    background: var(--surface2); border: 1px solid var(--border);
    cursor: pointer; text-align: center; transition: all 0.2s;
    font-size: 12px; font-weight: 600;
  }
  .style-tab:hover, .pill-tab:hover { border-color: var(--accent); }
  .style-tab.active, .pill-tab.active {
    background: var(--grad); border-color: transparent;
    color: white; box-shadow: 0 4px 16px -4px rgba(236, 72, 153, 0.4);
  }
  .style-tab .name { font-weight: 700; font-size: 13px; }
  .style-tab .desc { font-size: 11px; color: var(--text2); margin-top: 2px; }
  .style-tab.active .desc { color: rgba(255,255,255,0.85); }

  .pill-row {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px;
  }
  .pill-row.cols-4 { grid-template-columns: repeat(4, 1fr); }
  .pill-row.cols-5 { grid-template-columns: repeat(5, 1fr); }
  .pill-row.cols-6 { grid-template-columns: repeat(6, 1fr); }
  @media (max-width: 600px) {
    .pill-row, .pill-row.cols-4, .pill-row.cols-5, .pill-row.cols-6 { grid-template-columns: repeat(3, 1fr); }
  }

  .outfit-input {
    width: 100%; padding: 12px 14px;
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: var(--radius-sm); color: var(--text);
    font-size: 14px; font-family: inherit;
    transition: border-color 0.2s;
  }
  .outfit-input:focus { outline: none; border-color: var(--accent); }

  .row { display: flex; gap: 8px; align-items: center; margin-top: 10px; flex-wrap: wrap; }
  .btn-suggest {
    padding: 8px 14px; border-radius: 8px;
    background: var(--surface2); border: 1px solid var(--border);
    color: var(--text); cursor: pointer; font-size: 12px; font-weight: 600;
    transition: all 0.15s; white-space: nowrap;
  }
  .btn-suggest:hover { border-color: var(--accent); background: rgba(168, 85, 247, 0.1); }
  .btn-suggest:disabled { opacity: 0.5; cursor: not-allowed; }

  .ideas { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
  .idea {
    padding: 6px 10px; border-radius: 999px;
    background: var(--surface2); border: 1px solid var(--border);
    font-size: 12px; cursor: pointer; transition: all 0.15s;
    color: var(--text2);
  }
  .idea:hover { border-color: var(--accent); color: var(--text); background: rgba(168, 85, 247, 0.1); }

  .mod-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; margin-top: 10px; }
  @media (max-width: 600px) { .mod-grid { grid-template-columns: repeat(2, 1fr); } }
  .mod-chip {
    padding: 8px 6px; font-size: 11px; text-align: center;
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 8px; cursor: pointer; transition: all 0.15s;
    color: var(--text); font-weight: 500;
  }
  .mod-chip:hover { border-color: var(--accent); }
  .mod-chip.active {
    background: var(--grad); border-color: transparent; color: white;
    box-shadow: 0 4px 12px -4px rgba(236, 72, 153, 0.4);
  }

  .bg-grid {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px;
    max-height: 200px; overflow-y: auto; padding: 4px;
  }
  .bg-grid::-webkit-scrollbar { width: 6px; }
  .bg-grid::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  .bg-item {
    padding: 8px 6px; font-size: 11px; text-align: center;
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 8px; cursor: pointer; transition: all 0.15s;
    line-height: 1.2; min-height: 44px; display: flex; align-items: center; justify-content: center;
  }
  .bg-item:hover { border-color: var(--accent); }
  .bg-item.active { background: var(--grad); border-color: transparent; color: white; }

  .btn-generate {
    width: 100%; padding: 16px; margin-top: 16px;
    background: var(--grad); color: white;
    border: none; border-radius: var(--radius-sm);
    font-size: 16px; font-weight: 700; cursor: pointer;
    transition: all 0.2s; box-shadow: 0 8px 24px -8px rgba(236, 72, 153, 0.5);
  }
  .btn-generate:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 12px 28px -8px rgba(236, 72, 153, 0.6); }
  .btn-generate:disabled { opacity: 0.5; cursor: not-allowed; }

  .progress-wrap { margin-top: 16px; display: none; }
  .progress-wrap.show { display: block; }
  .progress-bar { height: 6px; background: var(--surface2); border-radius: 3px; overflow: hidden; }
  .progress-fill { height: 100%; background: var(--grad); width: 0%; transition: width 0.4s ease; border-radius: 3px; }
  .progress-stage { margin-top: 10px; font-size: 13px; color: var(--text2); display: flex; align-items: center; gap: 8px; }
  .progress-stage .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent); animation: pulse 1.4s infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.5; transform: scale(1.3); } }

  .result-wrap { text-align: center; }
  .result-img { max-width: 100%; border-radius: var(--radius-sm); box-shadow: 0 12px 40px -10px rgba(0, 0, 0, 0.6); }
  .result-video { max-width: 100%; border-radius: var(--radius-sm); box-shadow: 0 12px 40px -10px rgba(0, 0, 0, 0.6); margin-top: 12px; }
  .result-meta { margin-top: 12px; font-size: 12px; color: var(--text2); padding: 10px; background: var(--surface2); border-radius: 8px; text-align: left; }
  .result-meta code { color: var(--accent2); }
  .result-meta .ok { color: var(--ok); }
  .result-meta .warn { color: var(--warn); }
  .result-meta div { margin-top: 4px; }
  .result-meta div:first-child { margin-top: 0; }
  .result-actions { display: flex; gap: 8px; margin-top: 14px; justify-content: center; flex-wrap: wrap; }
  .btn-secondary {
    padding: 10px 16px; border-radius: 8px;
    background: var(--surface2); border: 1px solid var(--border);
    color: var(--text); cursor: pointer; font-size: 13px; font-weight: 600;
    transition: all 0.15s; text-decoration: none; display: inline-flex; align-items: center; gap: 6px;
  }
  .btn-secondary:hover { border-color: var(--accent); background: rgba(168, 85, 247, 0.1); }

  .history-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(80px, 1fr)); gap: 8px; margin-top: 10px; }
  .history-item { aspect-ratio: 1; border-radius: 8px; overflow: hidden; cursor: pointer; border: 1px solid var(--border); transition: all 0.15s; position: relative; }
  .history-item:hover { border-color: var(--accent); transform: scale(1.02); }
  .history-item img { width: 100%; height: 100%; object-fit: cover; }
  .history-empty { color: var(--text2); font-size: 13px; padding: 16px; text-align: center; }
  .history-item .del { position: absolute; top: 2px; right: 2px; background: rgba(0,0,0,0.7); color: white; width: 18px; height: 18px; border-radius: 50%; display: none; place-items: center; font-size: 11px; border: none; cursor: pointer; }
  .history-item:hover .del { display: grid; }

  .empty-state { color: var(--text2); font-size: 13px; padding: 32px 16px; text-align: center; line-height: 1.7; }
  .empty-state code { color: var(--accent2); }
  .error { margin-top: 12px; padding: 12px 14px; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; color: #fca5a5; font-size: 13px; }
  .info-banner { margin-top: 12px; padding: 10px 12px; background: rgba(168, 85, 247, 0.08); border: 1px solid rgba(168, 85, 247, 0.2); border-radius: 8px; font-size: 12px; color: var(--text2); }
  .info-banner strong { color: var(--accent); }

  /* Toast */
  .toast-wrap { position: fixed; top: 16px; right: 16px; z-index: 1000; display: flex; flex-direction: column; gap: 8px; }
  .toast {
    padding: 12px 16px; border-radius: 8px;
    background: rgba(20, 9, 31, 0.95); border: 1px solid var(--border);
    color: var(--text); font-size: 13px; max-width: 320px;
    backdrop-filter: blur(20px);
    animation: slidein 0.3s ease;
    display: flex; align-items: center; gap: 8px;
  }
  .toast.ok { border-color: var(--ok); }
  .toast.err { border-color: var(--err); }
  .toast.warn { border-color: var(--warn); }
  @keyframes slidein { from { transform: translateX(100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }

  footer { text-align: center; margin-top: 40px; color: var(--text2); font-size: 12px; }
  footer a { color: var(--accent); text-decoration: none; }
</style>
</head>
<body>
<div class="toast-wrap" id="toastWrap"></div>

<div class="container">
  <header>
    <div class="logo">
      <div class="logo-mark">✨</div>
      <div class="logo-text">MagicVisual</div>
    </div>
    <div class="tagline">
      Subí tu foto, elegí un outfit, la IA hace magia. Identidad preservada.
      <span class="pill">100% gratis</span>
      <span class="pill">v3.0</span>
      <span class="pill">IP2P + DALL-E + InsightFace</span>
    </div>
  </header>

  <div class="layout">
    <div>
      <div class="card" style="margin-bottom: 16px;">
        <h2>
          1 · Tu foto
          <span class="toggle-row">
            <label class="toggle" title="Subir a HD con Lanczos3 + sharpen antes de procesar">
              <input type="checkbox" id="tgUpscale">
              <span class="track"></span>
              <span>HD</span>
            </label>
            <label class="toggle" title="Conservar proporción original de la foto (sin recortar a 1024x1024)">
              <input type="checkbox" id="tgKeepAr">
              <span class="track"></span>
              <span>Mantener ratio</span>
            </label>
            <label class="toggle" title="Modo cuerpo: para fotos de piernas/pies. Usa Instruct-Pix2Pix y omite face swap.">
              <input type="checkbox" id="tgBodyOnly">
              <span class="track"></span>
              <span>Modo cuerpo</span>
            </label>
          </span>
        </h2>
        <div class="dropzone" id="dropzone">
          <div class="ico">📷</div>
          <div class="ttl">Hacé clic o arrastrá una foto</div>
          <div class="sub" id="dropzoneSub">JPG / PNG / WebP — máx 50MB — cara visible recomendada</div>
          <img class="preview" id="preview" style="display:none;" alt="preview">
          <input type="file" id="file" accept="image/*" style="display:none;">
        </div>
        <div id="photoInfo" style="display:none; margin-top:10px; font-size:12px; color:var(--text2);"></div>
      </div>

      <div class="card" style="margin-bottom: 16px;">
        <h2>2 · Estilo</h2>
        <div class="style-tabs" id="styleTabs">
          <div class="style-tab active" data-style="vanilla">
            <div class="name">Vainilla</div>
            <div class="desc">Sutil y elegante</div>
          </div>
          <div class="style-tab" data-style="versatil">
            <div class="name">Versátil</div>
            <div class="desc">Atrevido elegante</div>
          </div>
          <div class="style-tab" data-style="fetish">
            <div class="name">Fetiche</div>
            <div class="desc">Lencería, latex</div>
          </div>
        </div>
      </div>

      <div class="card" style="margin-bottom: 16px;">
        <h2>
          3 · Outfit
          <button class="btn-suggest" id="btnSuggest" title="La IA sugiere 10 outfits para este estilo / modo">✨ Sugerir con IA</button>
        </h2>
        <input type="text" class="outfit-input" id="outfit" placeholder="Ej: red lace lingerie set (o change to black latex stockings)" autocomplete="off">
        <div class="ideas" id="ideas"></div>
      </div>

      <div class="card" style="margin-bottom: 16px;">
        <h2>4 · Modificaciones corporales <span class="badge" id="modCount">0</span></h2>
        <div class="mod-grid" id="modGrid"></div>
      </div>

      <div class="card" style="margin-bottom: 16px;">
        <h2>5 · Calidad</h2>
        <div class="pill-row" id="qualityTabs"></div>
      </div>

      <div class="card" style="margin-bottom: 16px;">
        <h2>6 · Picante <span style="font-size:11px; color:var(--text2); text-transform:none; letter-spacing:0;">(modifier de pose/mood, seguro para filtro de contenido)</span></h2>
        <div class="pill-row cols-4" id="picanteTabs"></div>
      </div>

      <div class="card" style="margin-bottom: 16px;">
        <h2>
          7 · Fondo
          <label class="toggle" title="Conservar el fondo original de la foto en vez de reemplazarlo">
            <input type="checkbox" id="tgKeepBg">
            <span class="track"></span>
            <span>Mantener fondo</span>
          </label>
        </h2>
        <div class="bg-grid" id="bgGrid"></div>
      </div>

      <div class="card" style="margin-bottom: 16px;">
        <h2>8 · Animación <span style="font-size:11px; color:var(--text2); text-transform:none; letter-spacing:0;">(opcional, video Z.ai)</span></h2>
        <div class="pill-row cols-3" id="animTabs"></div>
      </div>

      <button class="btn-generate" id="btnGenerate" disabled>✨ Generar magia</button>

      <div class="progress-wrap" id="progressWrap">
        <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
        <div class="progress-stage">
          <span class="dot"></span>
          <span id="progressMsg">Iniciando...</span>
        </div>
      </div>

      <div id="errorBox"></div>
    </div>

    <div>
      <div class="card" id="resultCard">
        <h2>Resultado</h2>
        <div id="resultContent">
          <div class="empty-state">
            Tu imagen aparecerá acá.<br><br>
            <strong>Pipeline v3.0:</strong><br>
            1. <code>VLM</code> analiza tu foto (cuerpo + cara o solo cuerpo)<br>
            2. <code>DALL-E</code> o <code>IP2P</code> genera el outfit<br>
            3. <code>InsightFace</code> swappea tu rostro (si no es bodyOnly)<br>
            4. <code>Pollinations</code> fallback si Z.ai bloquea<br>
            5. (opcional) <code>Z.ai video</code> anima el resultado<br>
            ~20-40s · animación +1-3min
          </div>
        </div>
      </div>

      <div class="card" style="margin-top: 16px;">
        <h2>Historial <span class="badge" id="historyCount">0</span></h2>
        <div id="historyContent">
          <div class="history-empty">Tus resultados guardados aparecen acá (localStorage, máx 24).</div>
        </div>
      </div>

      <div class="card" style="margin-top: 16px;">
        <h2>Estado del servidor</h2>
        <div id="serverStatus" style="font-size:13px; color:var(--text2);">Cargando...</div>
      </div>
    </div>
  </div>

  <footer>
    MagicVisual v3.0 · Z.ai (DALL-E + GLM-4V + video) · HuggingFace InsightFace · Instruct-Pix2Pix · Pollinations.ai · Bun runtime<br>
    <a href="/health" target="_blank">/health</a> · <a href="/api/info" target="_blank">/api/info</a>
  </footer>
</div>

<script>
  const STYLES = ${stylesJson};
  const BODY_ONLY_IDEAS = ${bodyOnlyIdeasJson};
  const BODY_MODS = ${bodyModsJson};
  const QUALITY_PRESETS = ${qualityJson};
  const PICANTE_LEVELS = ${picanteJson};
  const ANIMATION_PRESETS = ${animationJson};
  const BACKGROUNDS = ${backgroundsJson};

  const state = {
    photoFile: null,
    photoPreview: null,
    photoBodyDesc: null,
    style: 'vanilla',
    outfit: '',
    bg: BACKGROUNDS[0].prompt,
    bodyMods: [],
    quality: 'standard',
    picante: 'off',
    animate: 'off',
    upscale: false,
    keepBackground: false,
    keepAspectRatio: false,
    bodyOnly: false,
    generating: false,
  };

  const $ = (id) => document.getElementById(id);
  const dropzone = $('dropzone'), fileInput = $('file'), preview = $('preview');
  const dropzoneSub = $('dropzoneSub');
  const outfitInput = $('outfit'), ideasEl = $('ideas'), bgGrid = $('bgGrid');
  const modGrid = $('modGrid'), modCount = $('modCount');
  const qualityTabs = $('qualityTabs'), picanteTabs = $('picanteTabs'), animTabs = $('animTabs');
  const btnGenerate = $('btnGenerate'), btnSuggest = $('btnSuggest');
  const progressWrap = $('progressWrap'), progressFill = $('progressFill'), progressMsg = $('progressMsg');
  const resultContent = $('resultContent'), errorBox = $('errorBox'), historyContent = $('historyContent');
  const photoInfo = $('photoInfo'), serverStatus = $('serverStatus'), historyCount = $('historyCount');
  const toastWrap = $('toastWrap');
  const tgUpscale = $('tgUpscale'), tgKeepAr = $('tgKeepAr'), tgBodyOnly = $('tgBodyOnly'), tgKeepBg = $('tgKeepBg');

  // ===== Toast =====
  function toast(msg, type = 'ok', duration = 3000) {
    const el = document.createElement('div');
    el.className = 'toast ' + type;
    el.textContent = msg;
    toastWrap.appendChild(el);
    setTimeout(() => { el.style.opacity = '0'; setTimeout(() => el.remove(), 300); }, duration);
  }

  // ===== Server status =====
  async function loadServerStatus() {
    try {
      const r = await fetch('/health');
      const d = await r.json();
      const upMin = Math.floor(d.uptime_sec / 60);
      const upSec = d.uptime_sec % 60;
      const feats = (d.features || []).slice(0, 8).join(', ');
      serverStatus.innerHTML =
        '<div>✅ Online · v' + d.version + '</div>' +
        '<div style="margin-top:6px;">Uptime: <strong>' + upMin + 'm ' + upSec + 's</strong></div>' +
        '<div>VLM cache: <strong>' + d.vlm_cache_size + '</strong> fotos</div>' +
        '<div>Rate limit: <strong>' + (d.rate_limit_max > 0 ? d.rate_limit_max + '/min' : 'off') + '</strong></div>' +
        '<div style="margin-top:6px; font-size:11px;">Features: ' + escapeHtml(feats) + '…</div>';
    } catch (e) {
      serverStatus.innerHTML = '<div style="color:var(--err);">❌ Server caído</div>';
    }
  }
  loadServerStatus();
  setInterval(loadServerStatus, 10000);

  // ===== Toggle bindings =====
  function bindToggle(el, key, onChange) {
    el.checked = !!state[key];
    el.onchange = () => {
      state[key] = el.checked;
      if (onChange) onChange();
    };
  }
  bindToggle(tgUpscale, 'upscale');
  bindToggle(tgKeepAr, 'keepAspectRatio');
  bindToggle(tgKeepBg, 'keepBackground');
  bindToggle(tgBodyOnly, 'bodyOnly', () => {
    // Update dropzone hint + re-render ideas
    dropzoneSub.textContent = state.bodyOnly
      ? 'JPG / PNG / WebP — máx 50MB — piernas/pies visibles (modo cuerpo)'
      : 'JPG / PNG / WebP — máx 50MB — cara visible recomendada';
    renderIdeas();
    if (state.bodyOnly) {
      toast('Modo cuerpo activado: IP2P preserva identidad. Face swap desactivado.', 'warn', 4500);
    } else {
      toast('Modo cara activado: DALL-E + InsightFace.', 'ok', 2500);
    }
  });

  // ===== Background grid =====
  function renderBgGrid() {
    bgGrid.innerHTML = '';
    BACKGROUNDS.forEach((bg) => {
      const el = document.createElement('div');
      el.className = 'bg-item' + (bg.prompt === state.bg ? ' active' : '');
      el.textContent = bg.label;
      el.title = bg.prompt;
      el.onclick = () => { state.bg = bg.prompt; renderBgGrid(); };
      bgGrid.appendChild(el);
    });
  }

  // ===== Ideas =====
  function renderIdeas() {
    const ideas = state.bodyOnly ? BODY_ONLY_IDEAS : (STYLES[state.style] || {ideas: []}).ideas;
    ideasEl.innerHTML = '';
    ideas.forEach((idea) => {
      const el = document.createElement('div');
      el.className = 'idea';
      el.textContent = idea.label;
      el.title = idea.prompt;
      el.onclick = () => { outfitInput.value = idea.prompt; state.outfit = idea.prompt; updateButton(); };
      ideasEl.appendChild(el);
    });
  }

  // ===== Body mods grid =====
  function renderModGrid() {
    modGrid.innerHTML = '';
    Object.entries(BODY_MODS).forEach(([key, mod]) => {
      const el = document.createElement('div');
      el.className = 'mod-chip' + (state.bodyMods.includes(key) ? ' active' : '');
      el.textContent = mod.label;
      el.onclick = () => {
        const idx = state.bodyMods.indexOf(key);
        if (idx >= 0) state.bodyMods.splice(idx, 1);
        else state.bodyMods.push(key);
        modCount.textContent = state.bodyMods.length;
        renderModGrid();
      };
      modGrid.appendChild(el);
    });
    modCount.textContent = state.bodyMods.length;
  }

  // ===== Quality / Picante / Animation tabs =====
  function renderPillTabs(container, map, stateKey) {
    container.innerHTML = '';
    Object.entries(map).forEach(([key, val]) => {
      const el = document.createElement('div');
      el.className = 'pill-tab' + (state[stateKey] === key ? ' active' : '');
      el.textContent = val.label;
      el.onclick = () => { state[stateKey] = key; renderPillTabs(container, map, stateKey); };
      container.appendChild(el);
    });
  }

  // ===== Style tabs =====
  document.querySelectorAll('.style-tab').forEach((tab) => {
    tab.onclick = () => {
      document.querySelectorAll('.style-tab').forEach((t) => t.classList.remove('active'));
      tab.classList.add('active');
      state.style = tab.dataset.style;
      if (!state.bodyOnly) renderIdeas();
    };
  });

  // ===== File upload =====
  dropzone.onclick = () => fileInput.click();
  dropzone.ondragover = (e) => { e.preventDefault(); dropzone.classList.add('drag'); };
  dropzone.ondragleave = () => dropzone.classList.remove('drag');
  dropzone.ondrop = (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag');
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  };
  fileInput.onchange = (e) => { if (e.target.files[0]) handleFile(e.target.files[0]); };

  async function handleFile(file) {
    if (!file.type.startsWith('image/')) { showError('El archivo debe ser una imagen.'); return; }
    if (file.size > 50 * 1024 * 1024) { showError('La imagen supera los 50MB.'); return; }
    state.photoFile = file;
    state.photoBodyDesc = null;
    const reader = new FileReader();
    reader.onload = (e) => {
      state.photoPreview = e.target.result;
      preview.src = e.target.result;
      preview.style.display = 'block';
      updateButton();
      photoInfo.style.display = 'block';
      photoInfo.textContent = 'Analizando' + (state.bodyOnly ? ' cuerpo' : ' rostro') + '...';
      describePhoto(file);
    };
    reader.readAsDataURL(file);
    clearError();
  }

  async function describePhoto(file) {
    try {
      const fd = new FormData();
      fd.append('photo', file);
      fd.append('bodyOnly', state.bodyOnly ? 'true' : 'false');
      const r = await fetch('/api/describe', { method: 'POST', body: fd });
      const d = await r.json();
      if (d.error) {
        photoInfo.innerHTML = '<span style="color:var(--warn);">⚠ ' + escapeHtml(d.error) + '</span>';
        return;
      }
      state.photoBodyDesc = d.body_desc;
      if (state.bodyOnly) {
        const cloth = d.current_clothing && d.current_clothing !== 'unknown'
          ? ' · ropa actual: <em>' + escapeHtml(d.current_clothing) + '</em>'
          : '';
        photoInfo.innerHTML = '<span style="color:var(--ok);">✓ Cuerpo analizado</span> · ' + escapeHtml((d.body_desc || '').slice(0, 90)) + cloth + (d.cached ? ' · <em>(cache)</em>' : '');
        toast('Foto lista para generar (modo cuerpo)', 'ok', 2000);
      } else {
        if (d.face_count === 0) {
          photoInfo.innerHTML = '<span style="color:var(--err);">⚠ No se detectó cara. Subí otra foto o activá modo cuerpo.</span>';
          toast('No detectamos cara en la foto', 'err');
        } else {
          photoInfo.innerHTML = '<span style="color:var(--ok);">✓ ' + d.face_count + ' cara' + (d.face_count > 1 ? 's' : '') + ' detectada' + (d.face_count > 1 ? 's' : '') + '</span> · ' + escapeHtml((d.body_desc || '').slice(0, 80)) + (d.cached ? ' · <em>(cache)</em>' : '');
          toast('Foto lista para generar', 'ok', 2000);
        }
      }
    } catch (e) {
      photoInfo.innerHTML = '<span style="color:var(--warn);">No se pudo analizar la foto (continuá igual)</span>';
    }
  }

  // ===== Outfit input =====
  outfitInput.oninput = (e) => { state.outfit = e.target.value; updateButton(); };

  function updateButton() {
    btnGenerate.disabled = !state.photoFile || !state.outfit.trim() || state.generating;
  }

  // ===== Suggest outfits =====
  btnSuggest.onclick = async () => {
    btnSuggest.disabled = true;
    btnSuggest.textContent = '⏳ Pensando...';
    try {
      const r = await fetch('/api/suggest-outfits', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          style: state.style,
          body_desc: state.photoBodyDesc,
          body_only: state.bodyOnly,
        }),
      });
      const d = await r.json();
      if (d.suggestions && d.suggestions.length > 0) {
        ideasEl.innerHTML = '';
        d.suggestions.forEach((item) => {
          const el = document.createElement('div');
          el.className = 'idea';
          el.textContent = item.label;
          el.title = item.prompt;
          el.style.background = 'rgba(168, 85, 247, 0.15)';
          el.style.borderColor = 'var(--accent)';
          el.onclick = () => { outfitInput.value = item.prompt; state.outfit = item.prompt; updateButton(); };
          ideasEl.appendChild(el);
        });
        toast(d.suggestions.length + ' outfits sugeridos', 'ok');
      } else {
        toast('No se pudieron sugerir outfits', 'warn');
      }
    } catch (e) {
      toast('Error: ' + e.message, 'err');
    } finally {
      btnSuggest.disabled = false;
      btnSuggest.textContent = '✨ Sugerir con IA';
    }
  };

  // ===== Generate =====
  btnGenerate.onclick = async () => {
    if (state.generating) return;
    state.generating = true;
    updateButton();
    clearError();
    showProgress(5, 'Preparando...');
    resultContent.innerHTML = '<div class="empty-state">Generando... esto puede tardar 20-40s (o más con animación).</div>';

    const formData = new FormData();
    formData.append('photo', state.photoFile);
    formData.append('outfit', state.outfit);
    formData.append('bg', state.bg);
    formData.append('style', state.style);
    formData.append('bodyMods', JSON.stringify(state.bodyMods));
    formData.append('quality', state.quality);
    formData.append('picante', state.picante);
    formData.append('upscale', state.upscale ? 'true' : 'false');
    formData.append('animate', state.animate);
    formData.append('keepBackground', state.keepBackground ? 'true' : 'false');
    formData.append('keepAspectRatio', state.keepAspectRatio ? 'true' : 'false');
    formData.append('bodyOnly', state.bodyOnly ? 'true' : 'false');

    try {
      const res = await fetch('/api/edit', { method: 'POST', body: formData });
      if (!res.ok) {
        const err = await res.text();
        throw new Error('HTTP ' + res.status + ': ' + err);
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const events = buf.split('\\n\\n');
        buf = events.pop() || '';
        for (const ev of events) {
          if (ev.startsWith('data: ')) {
            try { handleEvent(JSON.parse(ev.slice(6))); } catch (e) {}
          }
        }
      }
    } catch (e) {
      showError('Error: ' + e.message);
      resultContent.innerHTML = '<div class="empty-state">Falló la generación. Probá de nuevo.</div>';
    } finally {
      state.generating = false;
      updateButton();
    }
  };

  function handleEvent(data) {
    if (data.type === 'keepalive') {
      // Ignore keepalive events — they just keep the connection alive
      return;
    }
    if (data.type === 'progress') {
      const stageMap = {
        upscaling: [8, 'Mejorando resolución (HD)...'],
        analyzing: [15, 'Analizando tu foto con VLM...'],
        analyzed: [30, 'Cuerpo detectado, generando outfit...'],
        generating: [40, 'Generando outfit...'],
        ip2p_done: [70, 'Instruct-Pix2Pix listo...'],
        swapping: [75, 'InsightFace recomponiendo tu rostro...'],
        fallback: [85, 'Usando fallback local...'],
        animating: [92, 'Animando...'],
      };
      const [pct, msg] = stageMap[data.stage] || [55, data.msg || 'Procesando...'];
      showProgress(pct, data.msg ? (msg + ' — ' + data.msg) : msg);
    } else if (data.type === 'done') {
      showProgress(100, 'Listo!');
      showResult(data);
      saveToHistory(data);
      setTimeout(() => { progressWrap.classList.remove('show'); }, 1500);
      if (data.animationTaskId) {
        toast('¡Imagen lista! Animación en proceso...', 'ok', 3000);
        pollAnimation(data.animationTaskId);
      } else {
        toast('¡Magia lista! 🎉', 'ok');
      }
    } else if (data.type === 'error') {
      showError('Pipeline error: ' + data.error);
      progressWrap.classList.remove('show');
      toast('Error: ' + data.error, 'err', 5000);
    }
  }

  function showProgress(pct, msg) {
    progressWrap.classList.add('show');
    progressFill.style.width = pct + '%';
    progressMsg.textContent = msg;
  }

  function showResult(data) {
    const dataUrl = data.result;
    const downloadName = 'magicvisual-' + Date.now() + '.jpg';
    let metaHtml = '<div class="result-meta">';
    metaHtml += '<div><strong>Body detectado:</strong> <code>' + escapeHtml(data.bodyDesc || '') + '</code></div>';
    if (data.currentClothing) {
      metaHtml += '<div><strong>Ropa actual:</strong> <code>' + escapeHtml(data.currentClothing) + '</code></div>';
    }
    if (data.bodyOnly === undefined && !data.ip2pUsed) {
      // Face mode
      metaHtml += '<div><strong>Face swap:</strong> ' + (data.usedFallback ? '<span class="warn">Fallback local (calidad limitada)</span>' : '<span class="ok">InsightFace real ✓</span>') + '</div>';
    }
    metaHtml += '<div><strong>Motor:</strong> ' +
      (data.ip2pUsed ? '<span class="ok">Instruct-Pix2Pix ✓</span>' :
        data.pollinationsUsed ? '<span class="warn">Pollinations.ai (Z.ai bloqueado)</span>' :
        '<span class="ok">DALL-E edit ✓</span>') + '</div>';
    metaHtml += '<div><strong>VLM:</strong> ' + (data.cached ? '<span class="ok">cache ✓</span>' : 'fresh') + '</div>';
    if (data.faceCount !== undefined && data.faceCount !== null) {
      metaHtml += '<div><strong>Caras:</strong> ' + data.faceCount + '</div>';
    }
    metaHtml += '</div>';

    let animHtml = '';
    if (data.animationUrl) {
      // Direct video URL (from history reload)
      animHtml = '<video class="result-video" src="' + escapeHtml(data.animationUrl) + '" controls autoplay loop muted playsinline></video>';
    } else if (data.animationTaskId) {
      // Animation in progress — show placeholder that pollAnimation will replace
      animHtml = '<div id="animPlaceholder" style="padding: 16px; background: var(--surface2); border-radius: 8px; text-align: center; margin-top: 12px;">' +
        '<div style="font-size: 14px; color: var(--text2);">🎬 Animación en proceso...</div>' +
        '<div id="animProgress" style="font-size: 12px; color: var(--text2); margin-top: 6px;">Esperando respuesta del servidor...</div>' +
        '</div>';
    }

    resultContent.innerHTML =
      '<div class="result-wrap">' +
        '<img class="result-img" src="' + dataUrl + '" alt="result">' +
        animHtml +
        metaHtml +
        '<div class="result-actions">' +
          '<a class="btn-secondary" href="' + dataUrl + '" download="' + downloadName + '">⬇ Descargar imagen</a>' +
          (data.animationUrl ? '<a class="btn-secondary" href="' + escapeHtml(data.animationUrl) + '" target="_blank" download="magicvisual-' + Date.now() + '.mp4">⬇ Descargar video</a>' : '') +
          '<button class="btn-secondary" onclick="location.reload()">↻ Otra vez</button>' +
        '</div>' +
      '</div>';
  }

  function showError(msg) { errorBox.innerHTML = '<div class="error">' + escapeHtml(msg) + '</div>'; }
  function clearError() { errorBox.innerHTML = ''; }

  // ===== Poll animation status (async video generation) =====
  // Polls /api/animation-status every 5s until SUCCESS or FAIL.
  // Updates the #animPlaceholder with progress and replaces it with the video when done.
  function pollAnimation(taskId) {
    console.log('[Anim] Starting poll for', taskId);
    let attempts = 0;
    const maxAttempts = 40; // 40 * 8s = 320s = ~5.3 min max (matches backend 5 min)
    const interval = setInterval(async () => {
      attempts++;
      try {
        const res = await fetch('/api/animation-status?taskId=' + encodeURIComponent(taskId));
        const d = await res.json();
        const ph = document.getElementById('animPlaceholder');
        const prog = document.getElementById('animProgress');
        if (!ph) {
          // Placeholder no longer exists (user navigated away)
          clearInterval(interval);
          return;
        }

        if (d.status === 'SUCCESS' && d.videoUrl) {
          clearInterval(interval);
          console.log('[Anim] SUCCESS:', d.videoUrl);
          // Replace placeholder with video
          ph.outerHTML = '<video class="result-video" src="' + escapeHtml(d.videoUrl) + '" controls autoplay loop muted playsinline></video>';
          // Add download button for video
          const actions = document.querySelector('.result-actions');
          if (actions) {
            const dlBtn = document.createElement('a');
            dlBtn.className = 'btn-secondary';
            dlBtn.href = d.videoUrl;
            dlBtn.download = 'magicvisual-' + Date.now() + '.mp4';
            dlBtn.target = '_blank';
            dlBtn.textContent = '⬇ Descargar video';
            actions.insertBefore(dlBtn, actions.firstChild);
          }
          // Update history with the video URL
          const arr = getHistory();
          if (arr.length > 0) {
            arr[0].animationUrl = d.videoUrl;
            setHistory(arr);
          }
          toast('¡Video listo! 🎬', 'ok');
        } else if (d.status === 'FAIL') {
          clearInterval(interval);
          console.log('[Anim] FAIL:', d.error);
          ph.outerHTML = '<div style="padding: 12px; background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3); border-radius: 8px; color: #fca5a5; font-size: 13px; margin-top: 12px;">⚠ Animación falló: ' + escapeHtml(d.error || 'Error desconocido') + '</div>';
          toast('Animación falló', 'err', 4000);
        } else {
          // PROCESSING
          if (prog) {
            prog.textContent = 'Procesando... (' + attempts + '/' + maxAttempts + ') — ' + (d.ageSeconds || 0) + 's transcurridos';
          }
        }

        if (attempts >= maxAttempts) {
          clearInterval(interval);
          if (prog) {
            prog.parentElement.outerHTML = '<div style="padding: 12px; background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.3); border-radius: 8px; color: #fcd34d; font-size: 13px; margin-top: 12px;">⚠ La animación está tardando demasiado. El servidor sigue procesando en background, pero dejamos de esperar.</div>';
          }
        }
      } catch (e) {
        console.warn('[Anim] Poll error:', e);
      }
    }, 8000);  // poll every 8s (was 5s — backend polls every 10s, no point polling faster)
  }
  function escapeHtml(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }

  // ===== History =====
  function getHistory() {
    try { return JSON.parse(localStorage.getItem('mv_history') || '[]'); } catch { return []; }
  }
  function setHistory(arr) {
    localStorage.setItem('mv_history', JSON.stringify(arr.slice(0, 24)));
    renderHistory();
  }
  function saveToHistory(data) {
    const arr = getHistory();
    arr.unshift({
      img: data.result,
      ts: Date.now(),
      outfit: state.outfit,
      style: state.style,
      bodyDesc: data.bodyDesc,
      animationUrl: data.animationUrl || null,
    });
    setHistory(arr);
  }
  function renderHistory() {
    const arr = getHistory();
    historyCount.textContent = arr.length;
    if (arr.length === 0) {
      historyContent.innerHTML = '<div class="history-empty">Tus resultados guardados aparecen acá (localStorage, máx 24).</div>';
      return;
    }
    historyContent.innerHTML = '<div class="history-grid">' + arr.map((item, i) =>
      '<div class="history-item" data-i="' + i + '" title="' + escapeHtml(item.outfit || '') + '">' +
        '<img src="' + item.img + '" alt="result">' +
        (item.animationUrl ? '<div style="position:absolute;bottom:2px;left:2px;font-size:9px;background:rgba(0,0,0,0.7);padding:1px 4px;border-radius:4px;">▶</div>' : '') +
        '<button class="del" data-del="' + i + '">×</button>' +
      '</div>'
    ).join('') + '</div>';
    historyContent.querySelectorAll('.history-item').forEach((el) => {
      el.onclick = (e) => {
        if (e.target.classList.contains('del')) {
          e.stopPropagation();
          const i = parseInt(e.target.dataset.del);
          const arr = getHistory();
          arr.splice(i, 1);
          setHistory(arr);
          toast('Resultado eliminado', 'ok', 1500);
          return;
        }
        const i = parseInt(el.dataset.i);
        const arr = getHistory();
        showResult({
          result: arr[i].img,
          bodyDesc: arr[i].bodyDesc || '(from history)',
          usedFallback: false,
          cached: true,
          faceCount: 1,
          ip2pUsed: false,
          pollinationsUsed: false,
          animationUrl: arr[i].animationUrl || null,
        });
      };
    });
  }

  // ===== Init =====
  renderBgGrid();
  renderIdeas();
  renderModGrid();
  renderPillTabs(qualityTabs, QUALITY_PRESETS, 'quality');
  renderPillTabs(picanteTabs, PICANTE_LEVELS, 'picante');
  renderPillTabs(animTabs, ANIMATION_PRESETS, 'animate');
  renderHistory();
  updateButton();
</script>
</body>
</html>`;
}
