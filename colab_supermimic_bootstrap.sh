#!/usr/bin/env bash
# One-shot SuperMimic bootstrap on a fresh Colab GPU VM. Expects /content/setup.sh and /content/motion.sh
# (colab_setup_supermimic.sh / colab_supermimic_motion.sh) to be uploaded already. Idempotent.
set -euxo pipefail
bash /content/setup.sh
cd /content/supermimic_ws/supermimic
uv pip install --python .venv/bin/python 'setuptools<81'   # tensorboard needs pkg_resources
bash /content/motion.sh
[ -x /content/cloudflared ] || (curl -sL -o /content/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x /content/cloudflared)
echo BOOTSTRAP_OK
