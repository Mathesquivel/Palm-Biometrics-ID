#!/usr/bin/env bash
# Launcher — Palm Biometrics ID

cd "$(dirname "$(readlink -f "$0")")"

export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/home/palmtech/.Xauthority}"

# Ativa virtual env se disponível
if [ -f "palm_env/bin/activate" ]; then
    source palm_env/bin/activate
fi

exec python3 app.py "$@"
