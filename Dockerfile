# MagicVisual — Dockerfile universal
# Funciona en: Railway, Render, Fly.io, DigitalOcean App Platform, AWS App Runner, etc.
# Imagen base Bun (multi-arch, liviana)
FROM oven/bun:1.3 AS base

WORKDIR /app

# Instalar dependencias del sistema para sharp
# (libvips ya viene en la imagen oven/bun)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libvips-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar package.json primero (cache layer)
COPY package.json bun.lock* ./

# Instalar dependencias
RUN bun install --frozen-lockfile 2>/dev/null || bun install

# Copiar código
COPY index.ts ./

# Exponer puerto (Railway/Render inyectan PORT automáticamente)
EXPOSE 3000

# Healthcheck (opcional)
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:${PORT:-3000}/health || exit 1

# Arrancar
CMD ["bun", "run", "index.ts"]
