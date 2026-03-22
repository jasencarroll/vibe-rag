---
name: docker-ops
description: Manage Docker containers, images, and compose stacks
license: MIT
user-invocable: true
allowed-tools:
  - read_file
  - grep
  - glob
  - run_command
  - ask_user_question
---

# Docker Ops

Manage Docker containers, images, and compose stacks.

## Common Operations

### Build
- `docker build -t <tag> .`
- `docker compose build [service]`

### Run & Manage
- `docker compose up -d` / `docker compose down`
- `docker compose ps` — check running services
- `docker compose logs -f [service]` — tail logs
- `docker exec -it <container> sh` — shell into container

### Debug
- `docker compose logs --tail=50 [service]` — recent logs
- `docker inspect <container>` — full container details
- `docker stats` — resource usage
- `docker system df` — disk usage

### Cleanup
- `docker compose down -v` — stop and remove volumes
- `docker system prune -f` — remove unused resources
- `docker image prune -f` — remove dangling images

## Guidelines

- Always check `docker compose ps` before making changes
- Read `docker-compose.yml` / `Dockerfile` before modifying
- Confirm with user before `down -v` (destroys volumes/data)
- Confirm before `system prune` (removes unused resources)
- Use `--tail` on logs to avoid flooding output
