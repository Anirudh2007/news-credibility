# The web app: Node builds the static files, Caddy serves them and forwards the API routes the app uses.
# Built by docker-compose.yml as the `frontend` service; the API runs from the image in the folder above.
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM caddy:2.10
COPY Caddyfile /etc/caddy/Caddyfile
COPY --from=build /app/dist /srv
EXPOSE 8501
