"""
Warehouse Load Distribution Environment

Easy:   2D grid (5x5), simple box placement, avoid fragmentation
Medium: 2D grid (6x6), products with category + adjacency constraints
Hard:   3D grid (4x4x3), products with size/fragile/flammable + zone constraints

Each task mode runs as a completely separate episode.
Rewards are NOT implemented yet — all reward values return 0.0.
"""

import copy
import uuid
from typing import Any


# ═══════════════════════════════════════════════════════════════════════
#  TASK DEFINITIONS
#  Grid values:  0 = empty | -1 = blocked/obstacle | >0 = product ID
# ═══════════════════════════════════════════════════════════════════════

EASY_TASK = {
    "grid_shape": [5, 5],
    "initial_grid": [
        [ 0,  0,  0,  0,  0],
        [ 0, -1,  0,  0,  0],
        [ 0,  0,  0, -1,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  0, -1,  0,  0],
    ],
    "products": [
        {"id": 1, "name": "Box A"},
        {"id": 2, "name": "Box B"},
        {"id": 3, "name": "Box C"},
        {"id": 4, "name": "Box D"},
        {"id": 5, "name": "Box E"},
    ],
    "max_steps": 20,
    "description": "Place boxes on a 2D grid. Avoid fragmentation.",
}


MEDIUM_TASK = {
    "grid_shape": [6, 6],
    "initial_grid": [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ],
    "products": [
        {"id": 1,  "name": "Hammer",         "category": "tools"},
        {"id": 2,  "name": "Nails",           "category": "tools"},
        {"id": 3,  "name": "Screwdriver",     "category": "tools"},
        {"id": 4,  "name": "Screws",          "category": "tools"},
        {"id": 5,  "name": "Paint",           "category": "finishing"},
        {"id": 6,  "name": "Brush",           "category": "finishing"},
        {"id": 7,  "name": "Sandpaper",       "category": "finishing"},
        {"id": 8,  "name": "Wood Glue",       "category": "adhesives"},
        {"id": 9,  "name": "Tape",            "category": "adhesives"},
        {"id": 10, "name": "Measuring Tape",  "category": "tools"},
    ],
    "adjacency_constraints": [
        [1, 2],    # Hammer ↔ Nails
        [3, 4],    # Screwdriver ↔ Screws
        [5, 6],    # Paint ↔ Brush
        [5, 7],    # Paint ↔ Sandpaper
        [8, 9],    # Wood Glue ↔ Tape
        [1, 10],   # Hammer ↔ Measuring Tape
    ],
    "max_steps": 40,
    "description": "Place products while respecting adjacency constraints.",
}


HARD_TASK = {
    # 3 levels (0 = bottom rack, 2 = top rack), 4 rows x 4 cols each
    "grid_shape": [4, 4, 3],
    "initial_grid": [
        # Level 0 — bottom rack
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        # Level 1 — middle rack
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        # Level 2 — top rack
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
    ],
    "products": [
        {"id": 1,  "name": "Glass Vase",       "size": "small",  "fragile": True,  "flammable": False},
        {"id": 2,  "name": "Ceramic Plate Set", "size": "medium", "fragile": True,  "flammable": False},
        {"id": 3,  "name": "Fuel Can",          "size": "medium", "fragile": False, "flammable": True},
        {"id": 4,  "name": "Propane Tank",      "size": "big",    "fragile": False, "flammable": True},
        {"id": 5,  "name": "Steel Shelf Unit",  "size": "big",    "fragile": False, "flammable": False},
        {"id": 6,  "name": "Wooden Crate",      "size": "big",    "fragile": False, "flammable": True},
        {"id": 7,  "name": "LED Bulbs",         "size": "small",  "fragile": True,  "flammable": False},
        {"id": 8,  "name": "Paint Thinner",     "size": "small",  "fragile": False, "flammable": True},
        {"id": 9,  "name": "Rubber Tires",      "size": "big",    "fragile": False, "flammable": True},
        {"id": 10, "name": "Cotton Fabric",     "size": "medium", "fragile": False, "flammable": True},
        {"id": 11, "name": "Mirror",            "size": "medium", "fragile": True,  "flammable": False},
        {"id": 12, "name": "Concrete Bags",     "size": "big",    "fragile": False, "flammable": False},
    ],
    # Cells near heat sources — flammable items must NOT go here
    "heat_zones": [
        [0, 0, 0], [0, 1, 0],   # row 0, cols 0–1, level 0
        [0, 0, 1], [0, 1, 1],   # row 0, cols 0–1, level 1
    ],
    # Cells near electrical panels — flammable items must NOT go here
    "electrical_zones": [
        [3, 3, 0], [3, 2, 0],   # row 3, cols 2–3, level 0
        [3, 3, 1], [3, 2, 1],   # row 3, cols 2–3, level 1
    ],
    "safety_rules": [
        "Fragile items -> top rack (level 2)",
        "Flammable items -> away from heat zones and electrical zones",
        "Big items -> bottom rack (level 0)",
        "Flammable and non-flammable items must be separated",
    ],
    "max_steps": 60,
    "description": "3D warehouse with safety, size, and zone constraints.",
}


TASKS = {
    "easy":   EASY_TASK,
    "medium": MEDIUM_TASK,
    "hard":   HARD_TASK,
}


# ═══════════════════════════════════════════════════════════════════════
#  ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════

class WarehouseEnvironment:
    """
    Warehouse Load Distribution Environment.

    Lifecycle per episode:
      start(mode)  → configure task (easy / medium / hard)
      reset()      → fresh episode, build grid + product queue
      step(action) → agent places one product per step
      state()      → inspect current episode metadata
    """

    VALID_MODES = list(TASKS.keys())

    def __init__(self):
        self._mode: str = "easy"
        self._episode_id: str = ""
        self._step_count: int = 0
        self._max_steps: int = 20
        self._done: bool = False

        self._grid: list = []
        self._grid_shape: list[int] = []
        self._products_queue: list[dict] = []
        self._placed_products: list[dict] = []
        self._task_config: dict = {}

    # ── LIFECYCLE ─────────────────────────────────────────────────────

    def start(self, mode: str = "easy") -> dict:
        """Configure for a task mode. Must call reset() after."""
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Choose from {self.VALID_MODES}")

        self._mode = mode
        self._task_config = TASKS[mode]
        self._grid_shape = self._task_config["grid_shape"]
        self._max_steps = self._task_config["max_steps"]

        # Clear episode state
        self._episode_id = ""
        self._step_count = 0
        self._done = False
        self._grid = []
        self._products_queue = []
        self._placed_products = []

        return {
            "mode": mode,
            "grid_shape": self._grid_shape,
            "total_products": len(self._task_config["products"]),
            "max_steps": self._max_steps,
            "description": self._task_config["description"],
            "status": "configured — call /reset to begin",
        }

    def reset(self) -> dict:
        """Start a fresh episode for the current mode."""
        self._episode_id = str(uuid.uuid4())
        self._step_count = 0
        self._done = False
        self._grid = copy.deepcopy(self._task_config.get("initial_grid", []))
        self._products_queue = list(self._task_config.get("products", []))
        self._placed_products = []

        return {"observation": self._build_observation("Episode started.")}

    def step(self, action: dict) -> dict:
        """Place the next product at the given position."""
        if self._done:
            return {
                "observation": self._build_observation("Episode finished. Call reset()."),
                "reward": 0.0,
                "done": True,
                "info": {"error": "episode_done"},
            }

        if not self._episode_id:
            return {
                "observation": self._build_observation("No active episode. Call reset()."),
                "reward": 0.0,
                "done": False,
                "info": {"error": "no_episode"},
            }

        if not self._products_queue:
            self._done = True
            return {
                "observation": self._build_observation("All products placed."),
                "reward": 0.0,
                "done": True,
                "info": {"result": "all_placed"},
            }

        self._step_count += 1
        position = action.get("position", [])

        # ── Validate position ────────────────────────────────────────
        expected_dims = 3 if self._mode == "hard" else 2
        if len(position) != expected_dims:
            return {
                "observation": self._build_observation(
                    f"Invalid position dimensions. Expected {expected_dims}D, got {len(position)}D."
                ),
                "reward": 0.0,
                "done": False,
                "info": {"error": "bad_dimensions"},
            }

        if not self._is_in_bounds(position):
            return {
                "observation": self._build_observation(
                    f"Position {position} is out of grid bounds {self._grid_shape}."
                ),
                "reward": 0.0,
                "done": False,
                "info": {"error": "out_of_bounds"},
            }

        cell_value = self._get_cell(position)
        if cell_value != 0:
            label = "blocked" if cell_value == -1 else f"occupied by product {cell_value}"
            return {
                "observation": self._build_observation(
                    f"Position {position} is {label}. Choose an empty cell."
                ),
                "reward": 0.0,
                "done": False,
                "info": {"error": "cell_unavailable"},
            }

        # ── Place the product ────────────────────────────────────────
        product = self._products_queue.pop(0)
        self._set_cell(position, product["id"])
        self._placed_products.append({
            "product": product,
            "position": position,
        })

        # Check if episode is done
        done = len(self._products_queue) == 0 or self._step_count >= self._max_steps
        self._done = done

        # ── Reward placeholder (NOT implemented yet) ─────────────────
        reward = 0.0

        msg = f"Placed '{product['name']}' (id={product['id']}) at {position}."
        if done and len(self._products_queue) == 0:
            msg += " All products placed."
        elif done:
            msg += " Max steps reached."

        return {
            "observation": self._build_observation(msg),
            "reward": reward,
            "done": done,
            "info": {
                "episode_id": self._episode_id,
                "mode": self._mode,
                "step_count": self._step_count,
                "product_placed": product,
                "position": position,
            },
        }

    def state(self) -> dict:
        """Return current episode metadata."""
        return {
            "state": {
                "episode_id": self._episode_id,
                "mode": self._mode,
                "step_count": self._step_count,
                "max_steps": self._max_steps,
                "grid_shape": self._grid_shape,
                "grid": self._grid,
                "products_placed": len(self._placed_products),
                "products_remaining": len(self._products_queue),
                "placed_products": self._placed_products,
                "done": self._done,
            }
        }

    # ── GRID HELPERS ─────────────────────────────────────────────────

    def _get_cell(self, position: list[int]) -> int:
        if self._mode == "hard":
            row, col, level = position
            return self._grid[level][row][col]
        else:
            row, col = position
            return self._grid[row][col]

    def _set_cell(self, position: list[int], value: int):
        if self._mode == "hard":
            row, col, level = position
            self._grid[level][row][col] = value
        else:
            row, col = position
            self._grid[row][col] = value

    def _is_in_bounds(self, position: list[int]) -> bool:
        if self._mode == "hard":
            row, col, level = position
            rows, cols, levels = self._grid_shape
            return 0 <= row < rows and 0 <= col < cols and 0 <= level < levels
        else:
            row, col = position
            rows, cols = self._grid_shape
            return 0 <= row < rows and 0 <= col < cols

    # ── OBSERVATION BUILDER ──────────────────────────────────────────

    def _build_observation(self, message: str) -> dict:
        """Build a mode-appropriate observation dict."""
        current_product = self._products_queue[0] if self._products_queue else None

        obs = {
            "message": message,
            "mode": self._mode,
            "grid": self._grid,
            "grid_shape": self._grid_shape,
            "current_product": current_product,
            "products_remaining": len(self._products_queue),
            "done": self._done,
            "reward": 0.0,
        }

        # ── Easy: no extra constraints ───────────────────────────────
        if self._mode == "easy":
            obs["blocked_cells"] = self._find_blocked_cells()

        # ── Medium: add adjacency constraints + related products ─────
        elif self._mode == "medium":
            constraints = self._task_config.get("adjacency_constraints", [])
            obs["adjacency_constraints"] = constraints
            if current_product:
                obs["related_products"] = self._get_related_products(current_product["id"])
            obs["placed_products"] = self._placed_products

        # ── Hard: add zones, safety rules, product properties ────────
        elif self._mode == "hard":
            obs["heat_zones"] = self._task_config.get("heat_zones", [])
            obs["electrical_zones"] = self._task_config.get("electrical_zones", [])
            obs["safety_rules"] = self._task_config.get("safety_rules", [])
            obs["placed_products"] = self._placed_products

        return obs

    def _find_blocked_cells(self) -> list[list[int]]:
        """Find all blocked (-1) cells in a 2D grid."""
        blocked = []
        for r, row in enumerate(self._grid):
            for c, val in enumerate(row):
                if val == -1:
                    blocked.append([r, c])
        return blocked

    def _get_related_products(self, product_id: int) -> list[dict]:
        """For medium mode: find products that should be adjacent."""
        constraints = self._task_config.get("adjacency_constraints", [])
        all_products = self._task_config.get("products", [])
        product_map = {p["id"]: p for p in all_products}

        related_ids = set()
        for a, b in constraints:
            if a == product_id:
                related_ids.add(b)
            elif b == product_id:
                related_ids.add(a)

        return [product_map[rid] for rid in related_ids if rid in product_map]
