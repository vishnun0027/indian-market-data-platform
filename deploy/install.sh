#!/usr/bin/env bash
# =============================================================================
# install.sh — Deploy market-data sync as a systemd timer on Ubuntu/Debian VM
#
# Usage (run on the VM as the user who will own the service):
#   git clone <your-repo> ~/market-data-platform
#   cd ~/market-data-platform
#   bash deploy/install.sh
# =============================================================================
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_NAME="market-data-sync"
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"

# ── Detect uv ─────────────────────────────────────────────────────────────────
if command -v uv &>/dev/null; then
    UV_BIN="$(command -v uv)"
else
    echo "⚙  uv not found — installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    UV_BIN="$HOME/.local/bin/uv"
fi
echo "✓  uv: $UV_BIN"

# ── Install Python deps ───────────────────────────────────────────────────────
echo "⚙  Installing project dependencies..."
cd "$PROJECT_DIR"
"$UV_BIN" sync
echo "✓  Dependencies installed"

# ── Check .env ────────────────────────────────────────────────────────────────
if [[ ! -f "$PROJECT_DIR/.env" ]]; then
    echo ""
    echo "⚠  No .env file found! Copying from .env.example..."
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
    echo "   👉  Edit $PROJECT_DIR/.env and set HF_TOKEN and HF_REPO_ID, then re-run this script."
    exit 1
fi

# Warn if HF_TOKEN is missing or still placeholder
if ! grep -qE "^HF_TOKEN=.+" "$PROJECT_DIR/.env" 2>/dev/null; then
    echo ""
    echo "❌  HF_TOKEN is not set in .env — the push step will fail."
    echo "   Get your token from https://huggingface.co/settings/tokens"
    echo "   Then add to .env:  HF_TOKEN=hf_xxxxxxxxxxxx"
    exit 1
fi

echo "✓  .env looks good"

# ── Create systemd user directory ─────────────────────────────────────────────
mkdir -p "$SYSTEMD_USER_DIR"

# ── Install service and timer ─────────────────────────────────────────────────
echo "⚙  Installing systemd units..."

for UNIT in service timer; do
    SRC="$PROJECT_DIR/deploy/${SERVICE_NAME}.${UNIT}"
    DST="$SYSTEMD_USER_DIR/${SERVICE_NAME}.${UNIT}"

    # Substitute placeholders
    sed \
        -e "s|REPLACE_USER|$(whoami)|g" \
        -e "s|REPLACE_PROJECT_DIR|$PROJECT_DIR|g" \
        -e "s|REPLACE_UV_BIN|$UV_BIN|g" \
        "$SRC" > "$DST"

    echo "   → Installed $DST"
done

# ── Enable lingering so the timer survives logout ─────────────────────────────
# (requires sudo; skip if not available)
if sudo -n loginctl enable-linger "$(whoami)" 2>/dev/null; then
    echo "✓  Lingering enabled (timer survives logout)"
else
    echo "⚠  Could not enable linger (no sudo). Timer will stop when you log out."
    echo "   Ask your sysadmin to run:  sudo loginctl enable-linger $(whoami)"
fi

# ── Reload and enable ─────────────────────────────────────────────────────────
systemctl --user daemon-reload
systemctl --user enable --now "${SERVICE_NAME}.timer"
echo "✓  Timer enabled and started"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  ✅  market-data-sync is installed!              ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║                                                  ║"
echo "║  Next scheduled run:                             ║"
systemctl --user list-timers "${SERVICE_NAME}.timer" --no-pager 2>/dev/null | \
    grep "${SERVICE_NAME}" | awk '{print "║    " $1 " " $2 " " $3 "                       ║"}' || true
echo "║                                                  ║"
echo "║  Useful commands:                                ║"
echo "║                                                  ║"
echo "║  Run NOW (don't wait for timer):                 ║"
echo "║    systemctl --user start market-data-sync       ║"
echo "║                                                  ║"
echo "║  Watch live logs:                                ║"
echo "║    journalctl --user -u market-data-sync -f      ║"
echo "║                                                  ║"
echo "║  Check timer status:                             ║"
echo "║    systemctl --user list-timers                  ║"
echo "║                                                  ║"
echo "║  Disable the timer:                              ║"
echo "║    systemctl --user disable market-data-sync     ║"
echo "╚══════════════════════════════════════════════════╝"
