# LOBSTER Research Studio

A local Dash app for exploring limit order book data and running research calculations.

## Folder structure

```text
lobster-app/
    data/
        .gitkeep
        AAPL_2013-09-29_2013-09-30_10/   (local dataset)
        AAPL_2013-09-29_2013-09-30_10.7z (local archive)
    lobster_app/
        app.py
        research.py
        assets/style.css
        requirements.txt
        test_research.py
        README.md
        .venv/                          (local Python environment)
    lobster_exploration.py
    Launch LOBSTER.command
    .gitignore
    README.md
```

## Data

Put extracted matching message and orderbook CSV pairs inside `data`. Each dataset can have its own subfolder. Restart the app after adding datasets. The app searches only this data folder, including its subfolders. Archives are not extracted automatically.

The `.gitignore` excludes the contents of `data`, except the empty `.gitkeep` placeholder. It also excludes Python environments and caches. Sharing through Git therefore preserves an empty data folder without including local datasets. Recipients must add their own CSV pairs before starting the app. Files already tracked by Git are not affected by ignore rules.

## Setup and launch

Use Python 3.11. Run these commands from the `lobster-app` folder on a new installation. They install the dependencies into a local environment.

### macOS and Linux

```sh
python3.11 -m venv lobster_app/.venv
lobster_app/.venv/bin/python -m pip install -r lobster_app/requirements.txt
lobster_app/.venv/bin/python -B lobster_app/app.py --open-browser
```

On this computer the existing environment has been moved with the app. On macOS, double click `Launch LOBSTER.command` to launch after setup.

### Windows PowerShell

```powershell
py -3.11 -m venv lobster_app/.venv
.\lobster_app\.venv\Scripts\python.exe -m pip install -r lobster_app/requirements.txt
.\lobster_app\.venv\Scripts\python.exe -B lobster_app/app.py --open-browser
```

Open <http://127.0.0.1:8050>. Keep the terminal open while using the app. Press Ctrl+C to stop it. Append `--port 8051` to the launch command if port 8050 is occupied.

Python and installed dependencies are required on each computer. Internet access is needed for installation. The Windows and Linux setup instructions have not been tested on those systems.

## Research documentation and tests

See [the detailed documentation](lobster_app/README.md) for usage and mathematical definitions.

Run tests from this folder on macOS or Linux.

```sh
lobster_app/.venv/bin/python -B -m unittest lobster_app.test_research
```

On Windows, use `lobster_app\.venv\Scripts\python.exe` instead.
