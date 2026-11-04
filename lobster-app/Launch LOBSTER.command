#!/bin/zsh
# Launch the local research application using its own Python environment.
app_directory="${0:A:h}/lobster_app"
app_python="${app_directory}/.venv/bin/python"
if [[ ! -x "$app_python" ]]; then
    print "The LOBSTER Python environment is missing. See lobster_app/README.md."
    read "reply?Press Return to close."
    exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
export DASH_DISABLE_VERSION_CHECK=true
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
exec "$app_python" -B "${app_directory}/app.py" --open-browser
