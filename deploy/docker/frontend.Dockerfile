# Frontend Dockerfile
FROM node:18-alpine as builder
WORKDIR /app
# Copy package files
COPY frontend/package*.json ./
# Install dependencies
RUN npm ci --only=production
# Copy source code
COPY frontend/ .
# Build application
RUN npm run build
# Runtime stage
FROM nginx:alpine
# Copy built application
COPY --from=builder /app/build /usr/share/nginx/html
# Copy nginx configuration
COPY deploy/docker/nginx.conf /etc/nginx/nginx.conf
# Expose port
EXPOSE 80
# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
CMD wget --no-verbose --tries=1 --spider http://localhost:80 || exit 1
# Start nginx
CMD ["nginx", "-g", "daemon off;"]
