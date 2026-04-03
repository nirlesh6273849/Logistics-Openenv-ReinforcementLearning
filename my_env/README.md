# Warehouse Load Distribution Environment

An [OpenEnv](https://github.com/speckai/openenv)-compatible reinforcement learning environment for warehouse load optimization.

## Structure

```
my_env/
├── .dockerignore
├── __init__.py
├── models.py               # WarehouseAction, WarehouseObservation, WarehouseState
├── client.py               # WarehouseEnv async HTTP client
├── openenv.yaml
├── pyproject.toml
├── README.md
├── outputs/
│   ├── logs/
│   └── evals/
└── server/
    ├── your_environment.py  # WarehouseEnvironment with task definitions
    ├── app.py               # FastAPI: /start, /reset, /step, /state, /health
    ├── requirements.txt
    └── Dockerfile
```

## Task Modes

### Easy — 2D Grid, Simple Placement

- **Grid:** 5x5 with 3 blocked cells
- **Products:** 5 uniform boxes (Box A through E)
- **Constraints:** None — just avoid blocked cells
- **Goal:** Maximize space utilization, avoid fragmentation
- **Action:** `[row, col]`

```
Grid (0 = empty, -1 = blocked):
[ 0,  0,  0,  0,  0]
[ 0, -1,  0,  0,  0]
[ 0,  0,  0, -1,  0]
[ 0,  0,  0,  0,  0]
[ 0,  0, -1,  0,  0]
```

### Medium — 2D Grid, Adjacency Constraints

- **Grid:** 6x6, starts empty
- **Products:** 10 categorized items (tools, finishing, adhesives)
- **Constraints:** 6 adjacency rules (e.g., Hammer near Nails)
- **Goal:** Place related products close together
- **Action:** `[row, col]`

```
Adjacency constraints:
  Hammer <-> Nails
  Screwdriver <-> Screws
  Paint <-> Brush
  Paint <-> Sandpaper
  Wood Glue <-> Tape
  Hammer <-> Measuring Tape
```

### Hard — 3D Grid, Safety + Zone Constraints

- **Grid:** 4x4x3 (4 rows, 4 cols, 3 rack levels)
- **Products:** 12 items with size (big/medium/small), fragile, flammable properties
- **Constraints:**
  - Fragile items -> top rack (level 2)
  - Big items -> bottom rack (level 0)
  - Flammable items -> away from heat and electrical zones
  - Separate flammable from non-flammable
- **Danger zones:** Heat (rows 0, cols 0-1) and Electrical (rows 3, cols 2-3)
- **Action:** `[row, col, level]`

## Quick Start

### Run the server

```bash
cd my_env/server
uvicorn app:app --reload --port 8000
```

### Task execution flow

Each mode runs as a **completely independent episode**:

```python
import requests

BASE = "http://localhost:8000"

for mode in ["easy", "medium", "hard"]:
    # 1. Configure mode
    requests.post(f"{BASE}/start", json={"mode": mode})

    # 2. Fresh reset
    r = requests.post(f"{BASE}/reset")
    obs = r.json()["observation"]

    # 3. Step loop
    done = False
    while not done:
        position = agent.pick_position(obs)  # your agent logic
        r = requests.post(f"{BASE}/step", json={"position": position})
        data = r.json()
        obs = data["observation"]
        done = data["done"]

    # 4. Independent scoring per mode
    print(f"{mode}: done")
```

## API Endpoints

| Method | Path      | Body                    | Description                           |
|--------|-----------|-------------------------|---------------------------------------|
| GET    | `/health` | —                       | Health check                          |
| POST   | `/start`  | `{"mode": "easy"}`      | Configure task mode                   |
| POST   | `/reset`  | —                       | Reset episode, get initial observation|
| POST   | `/step`   | `{"position": [r, c]}`  | Place product at position             |
| GET    | `/state`  | —                       | Get current episode state             |

## Rewards

**Not implemented yet.** All reward values currently return `0.0`.

## License

MIT
