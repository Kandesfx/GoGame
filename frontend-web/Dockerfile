# Build stage
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Build arguments for Vite (cần lúc build time)
# Try both ARG and ENV to ensure it's available
ARG VITE_API_URL
ENV VITE_API_URL=${VITE_API_URL:-https://gogame-backend.fly.dev}

# Debug: Print build args
RUN echo "=== Build Args ===" && \
    echo "VITE_API_URL=$VITE_API_URL" && \
    echo "=== Environment ===" && \
    env | grep VITE || echo "No VITE env vars"

# Build application
RUN npm run build

# Debug: Verify API URL in built files
RUN echo "=== Checking built files for API URL ===" && \
    grep -r "localhost:8000" /app/dist 2>/dev/null && echo "⚠️ Found localhost:8000 in build!" || echo "✅ No localhost:8000 found" && \
    grep -r "gogame-backend.fly.dev" /app/dist 2>/dev/null && echo "✅ Found gogame-backend.fly.dev in build!" || echo "⚠️ gogame-backend.fly.dev not found"

# Verify build output
RUN echo "=== Build Verification ===" && \
    ls -la /app/dist && \
    echo "=== Assets ===" && \
    ls -la /app/dist/assets 2>/dev/null || echo "No assets folder" && \
    echo "=== index.html ===" && \
    test -f /app/dist/index.html && echo "✅ index.html exists" || (echo "❌ index.html missing" && exit 1) && \
    cat /app/dist/index.html | head -5

# Production stage
FROM nginx:alpine

# Copy built files
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Create a simple health check file
RUN echo '<!DOCTYPE html><html><head><title>OK</title></head><body>OK</body></html>' > /usr/share/nginx/html/health.html

# Expose port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]

