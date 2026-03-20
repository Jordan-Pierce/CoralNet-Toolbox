# Credit: https://www.a1k0n.net/code/tron.html
# Algorithm: Google AI Challenge 2010 winner post-mortem
#   https://www.a1k0n.net/2010/03/04/google-ai-challenge-postmortem.html

import random
import warnings
from collections import deque

warnings.filterwarnings("ignore")

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont
from PyQt5.QtWidgets import (QMainWindow, QMessageBox, QDialog, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QButtonGroup)

from coralnet_toolbox.Icons import get_icon


# ----------------------------------------------------------------------------------------------------------------------
# Constants / Configurations
# ----------------------------------------------------------------------------------------------------------------------

BOARD_WIDTH  = 80
BOARD_HEIGHT = 60
CELL_SIZE    = 12

BACKGROUND_COLOR = QColor(0, 0, 0)
GRID_COLOR       = QColor(0, 100, 100)
PLAYER_COLOR     = QColor(255, 215, 0)
OPPONENT_COLOR   = QColor(255, 0, 0)

UP    = 0
DOWN  = 1
LEFT  = 2
RIGHT = 3

OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
DELTA    = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}
NBRS     = ((-1, 0), (1, 0), (0, -1), (0, 1))


# ----------------------------------------------------------------------------------------------------------------------
# LightCycle
# ----------------------------------------------------------------------------------------------------------------------

class LightCycle:
    """Represents a light cycle in the game."""

    def __init__(self, start_x, start_y, direction, color):
        self.x         = start_x
        self.y         = start_y
        self.direction = direction
        self.color     = color
        self.trail     = [(start_x, start_y)]
        self.alive     = True

    def move(self):
        if not self.alive:
            return
        dx, dy  = DELTA[self.direction]
        self.x += dx
        self.y += dy
        self.trail.append((self.x, self.y))

    def turn(self, new_direction):
        """Change direction unless it would be a 180-degree reversal."""
        if OPPOSITE[self.direction] != new_direction:
            self.direction = new_direction

    def check_collision(self, board_width, board_height, opponent_trail, obstacles):
        """Return True (and mark dead) if the cycle has crashed."""
        if self.x < 0 or self.x >= board_width or self.y < 0 or self.y >= board_height:
            self.alive = False; return True
        if (self.x, self.y) in self.trail[:-1]:
            self.alive = False; return True
        if (self.x, self.y) in opponent_trail:
            self.alive = False; return True
        if (self.x, self.y) in obstacles:
            self.alive = False; return True
        return False


# ----------------------------------------------------------------------------------------------------------------------
# LightCycleAI  --  faithful 1v1 implementation of the a1k0n algorithm
# ----------------------------------------------------------------------------------------------------------------------

class LightCycleAI:
    """
    1v1 Light Cycle AI based on the Google AI Challenge 2010 winning algorithm.

    Components (in descending order of impact):
      1. Chamber tree / battlefront heuristic  -- the contest-winning idea
      2. Data-mined Voronoi coefficients        -- K1 = 0.055, K2 = 0.194
      3. Iterative-deepening endgame search     -- greedy-fill leaves
      4. Minimax with alpha-beta pruning        -- Voronoi evaluation at leaves
      5. Checkerboard parity territory bound    -- tighter endgame count
      6. Wall-hugging edge tiebreaker           -- automatic boundary-following
    """

    def __init__(self, difficulty="Medium"):
        cfg = {
            "Easy":   dict(search_depth=2, endgame_depth=3, mistake_chance=0.12),
            "Medium": dict(search_depth=3, endgame_depth=4, mistake_chance=0.06),
            "Hard":   dict(search_depth=4, endgame_depth=5, mistake_chance=0.02),
            "Insane": dict(search_depth=5, endgame_depth=6, mistake_chance=0.00),
        }.get(difficulty, dict(search_depth=3, endgame_depth=4, mistake_chance=0.06))

        self.search_depth   = cfg["search_depth"]
        self.endgame_depth  = cfg["endgame_depth"]
        self.mistake_chance = cfg["mistake_chance"]

        # Coefficients from data-mining 11,691 contest games (post-mortem "data mining")
        self.K1 = 0.055   # node coefficient
        self.K2 = 0.194   # edge coefficient (double-counted open neighbours)

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sim(pos, d):
        dx, dy = DELTA[d]
        return (pos[0] + dx, pos[1] + dy)

    @staticmethod
    def _valid(bw, bh, occ, pos):
        x, y = pos
        return 0 <= x < bw and 0 <= y < bh and pos not in occ

    def _valid_dirs(self, pos, cur_dir, bw, bh, occ):
        return [d for d in (UP, DOWN, LEFT, RIGHT)
                if d != OPPOSITE[cur_dir] and self._valid(bw, bh, occ, self._sim(pos, d))]

    @staticmethod
    def _open_nbrs(bw, bh, occ, x, y):
        """Count free (non-wall, non-occupied) neighbours of (x, y)."""
        return sum(
            1 for dx, dy in NBRS
            if 0 <= x + dx < bw and 0 <= y + dy < bh and (x + dx, y + dy) not in occ
        )

    # ------------------------------------------------------------------
    # Separation check  (endgame trigger)
    # ------------------------------------------------------------------

    @staticmethod
    def _separated(bw, bh, occ, p_pos, o_pos):
        """True iff p_pos and o_pos are in different connected components of free space."""
        visited = {p_pos}
        q = deque([p_pos])
        while q:
            x, y = q.popleft()
            if (x, y) == o_pos:
                return False
            for dx, dy in NBRS:
                nb = (x + dx, y + dy)
                nx, ny = nb
                if (0 <= nx < bw and 0 <= ny < bh
                        and nb not in occ and nb not in visited):
                    visited.add(nb)
                    q.append(nb)
        return True

    # ------------------------------------------------------------------
    # Voronoi territories  (simultaneous BFS from both players)
    # ------------------------------------------------------------------

    def _voronoi(self, bw, bh, occ, p_pos, o_pos):
        """
        Compute Voronoi diagram: each free cell is 'player', 'opponent', or 'neutral'.

        Returns (territories_dict, p_nodes, p_edges, o_nodes, o_edges).
        Edges are open-neighbour counts summed over owned cells (double-counted,
        matching the original contest's metric).

        occ should already have the head positions removed so BFS starts
        from accessible cells.
        """
        dist  = {}
        owner = {}
        q     = deque()

        for pos, who in ((p_pos, 'player'), (o_pos, 'opponent')):
            dist[pos]  = 0
            owner[pos] = who
            q.append((pos[0], pos[1], 0, who))

        while q:
            x, y, d, who = q.popleft()
            if dist.get((x, y), 9999) < d:
                continue
            for dx, dy in NBRS:
                nx, ny = x + dx, y + dy
                nb     = (nx, ny)
                if not (0 <= nx < bw and 0 <= ny < bh) or nb in occ:
                    continue
                nd = d + 1
                if nb not in dist:
                    dist[nb]  = nd
                    owner[nb] = who
                    q.append((nx, ny, nd, who))
                elif dist[nb] == nd and owner.get(nb) != who:
                    owner[nb] = 'neutral'

        p_n = p_e = o_n = o_e = 0
        for (x, y), w in owner.items():
            if (x, y) in (p_pos, o_pos):
                continue                             # heads are already occupied
            e = self._open_nbrs(bw, bh, occ, x, y)
            if w == 'player':
                p_n += 1; p_e += e
            elif w == 'opponent':
                o_n += 1; o_e += e

        return owner, p_n, p_e, o_n, o_e

    # ------------------------------------------------------------------
    # Checkerboard territory  (tighter endgame upper bound)
    # ------------------------------------------------------------------

    def _checkerboard(self, bw, bh, occ, p_pos, o_pos):
        """
        Rerun Voronoi and count red/black squares per player.

        Upper bound on fillable squares = 2*min(red,black) + min(1,|red-black|).
        Players alternate colours on every step, so a colour surplus is wasted.
        Returns (p_effective, o_effective).
        """
        dist  = {}
        owner = {}
        q     = deque()
        for pos, who in ((p_pos, 'player'), (o_pos, 'opponent')):
            dist[pos]  = 0
            owner[pos] = who
            q.append((pos[0], pos[1], 0, who))

        while q:
            x, y, d, who = q.popleft()
            if dist.get((x, y), 9999) < d:
                continue
            for dx, dy in NBRS:
                nx, ny = x + dx, y + dy
                nb     = (nx, ny)
                if not (0 <= nx < bw and 0 <= ny < bh) or nb in occ:
                    continue
                nd = d + 1
                if nb not in dist:
                    dist[nb]  = nd
                    owner[nb] = who
                    q.append((nx, ny, nd, who))
                elif dist[nb] == nd and owner.get(nb) != who:
                    owner[nb] = 'neutral'

        p_r = p_b = o_r = o_b = 0
        for (x, y), w in owner.items():
            red = (x + y) % 2 == 0
            if w == 'player':
                if red: p_r += 1
                else:   p_b += 1
            elif w == 'opponent':
                if red: o_r += 1
                else:   o_b += 1

        def eff(r, b):
            return 2 * min(r, b) + min(1, abs(r - b))

        return eff(p_r, p_b), eff(o_r, o_b)

    # ------------------------------------------------------------------
    # Articulation points  (iterative Tarjan's, safe for large boards)
    # ------------------------------------------------------------------

    def _articulation_points(self, bw, bh, occ):
        """
        Find all articulation points (cut vertices) in the free-space graph.
        Removing an AP disconnects the graph into separate chambers.
        Uses iterative DFS to stay within Python's stack limits on 80x60 boards.
        """
        free   = []
        to_idx = {}
        for x in range(bw):
            for y in range(bh):
                p = (x, y)
                if p not in occ:
                    to_idx[p] = len(free)
                    free.append(p)

        n = len(free)
        if n <= 2:
            return set()

        adj = [[] for _ in range(n)]
        for i, (x, y) in enumerate(free):
            for dx, dy in NBRS:
                nb = (x + dx, y + dy)
                if nb in to_idx:
                    adj[i].append(to_idx[nb])

        disc      = [-1] * n
        low       = [0]  * n
        parent    = [-1] * n
        ap        = [False] * n
        child_cnt = [0]  * n
        timer     = [0]

        for start in range(n):
            if disc[start] != -1:
                continue
            disc[start] = low[start] = timer[0]; timer[0] += 1
            stack = [[start, 0]]          # [node, next_neighbour_index]

            while stack:
                u, i = stack[-1]
                if i < len(adj[u]):
                    stack[-1][1] += 1
                    v = adj[u][i]
                    if disc[v] == -1:
                        child_cnt[u] += 1
                        parent[v]     = u
                        disc[v] = low[v] = timer[0]; timer[0] += 1
                        stack.append([v, 0])
                    elif v != parent[u]:
                        low[u] = min(low[u], disc[v])
                else:
                    stack.pop()
                    if stack:
                        p_node = stack[-1][0]
                        low[p_node] = min(low[p_node], low[u])
                        if parent[p_node] == -1:
                            if child_cnt[p_node] > 1:
                                ap[p_node] = True
                        elif low[u] >= disc[p_node]:
                            ap[p_node] = True

        return {free[i] for i in range(n) if ap[i]}

    # ------------------------------------------------------------------
    # Chamber tree helpers
    # ------------------------------------------------------------------

    def _flood_chamber(self, bw, bh, occ, territories, start, aps, forbidden):
        """
        Flood-fill the current 'chamber' starting from `start`.

        Stops at articulation points (records them as borders for later recursion)
        but always enters `start` even when it is itself an AP.

        Returns (node_count, edge_count, border_ap_set, visited_cells).
        """
        visited    = set()
        border_aps = set()
        nodes = edges = 0
        stack = [start]

        while stack:
            pos = stack.pop()
            if pos in visited or pos in forbidden:
                continue
            t = territories.get(pos)
            if t not in ('player', 'neutral'):
                continue
            # Treat APs as chamber boundaries -- but always enter the start cell
            if pos in aps and pos != start:
                border_aps.add(pos)
                continue

            visited.add(pos)
            nodes += 1
            x, y = pos
            for dx, dy in NBRS:
                nb = (x + dx, y + dy)
                nx, ny = nb
                if not (0 <= nx < bw and 0 <= ny < bh) or nb in occ:
                    continue
                if territories.get(nb) in ('player', 'neutral'):
                    edges += 1
                    if nb not in visited and nb not in forbidden:
                        stack.append(nb)

        return nodes, edges, border_aps, visited

    def _is_battlefront(self, bw, bh, occ, territories, start, aps, forbidden):
        """
        True if the chamber accessible from `start` (without crossing APs or
        `forbidden`) contains a neutral square or is adjacent to opponent territory.
        That makes it a 'battlefront' chamber -- it contests the Voronoi boundary.
        """
        stack   = [start]
        visited = set()

        while stack:
            pos = stack.pop()
            if pos in visited or pos in forbidden:
                continue
            if pos in aps and pos != start:
                continue                             # don't cross APs
            t = territories.get(pos)
            if t in ('neutral', 'opponent'):
                return True
            if t != 'player':
                continue
            visited.add(pos)
            x, y = pos
            for dx, dy in NBRS:
                nb = (x + dx, y + dy)
                nx, ny = nb
                if not (0 <= nx < bw and 0 <= ny < bh) or nb in occ:
                    continue
                if territories.get(nb) in ('neutral', 'opponent'):
                    return True
                if nb not in visited and nb not in forbidden:
                    stack.append(nb)

        return False

    # ------------------------------------------------------------------
    # Chamber tree  (the contest-winning heuristic)
    # ------------------------------------------------------------------

    def _max_articulated_space(self, bw, bh, occ, territories, start_pos, aps):
        """
        Chamber-tree evaluation -- a1k0n's key insight from the post-mortem.

        When a player is not in the chamber that borders the battlefront they face
        a mutual-exclusion choice: fill safe chambers OR move to contest the
        battlefront.  Naive Voronoi counts both, wildly overestimating territory.

        Algorithm (recursive):
          1. Flood-fill the current chamber, noting bordering APs.
          2. For each adjacent chamber (via AP):
             - BATTLEFRONT chamber -> value = 1 (just the entry cost).
               Ignore the current chamber's size; you cannot claim both.
             - SAFE chamber        -> value = current + recurse(child).
          3. Return the maximum achievable (nodes, edges).
        """
        def recurse(pos, forbidden):
            nodes, edges, border_aps, my_cells = self._flood_chamber(
                bw, bh, occ, territories, pos, aps, forbidden)

            if not border_aps:
                return nodes, edges

            new_forbidden  = forbidden | my_cells
            best_n, best_e = nodes, edges          # default: stay in current chamber

            for ap in border_aps:
                ax, ay = ap
                for dx, dy in NBRS:
                    child = (ax + dx, ay + dy)
                    cx, cy = child
                    if not (0 <= cx < bw and 0 <= cy < bh) or child in occ:
                        continue
                    if child in new_forbidden:
                        continue
                    if territories.get(child) not in ('player', 'neutral'):
                        continue

                    bf_forbidden = new_forbidden | {ap}
                    is_bf = self._is_battlefront(
                        bw, bh, occ, territories, child, aps, bf_forbidden)

                    if is_bf:
                        # Battlefront: ignore current chamber, count only entry cost
                        pot_n, pot_e = 1, 0
                    else:
                        # Safe child chamber: accumulate sizes
                        child_n, child_e = recurse(child, new_forbidden | {ap})
                        pot_n = nodes + child_n
                        pot_e = edges + child_e

                    if pot_n > best_n:
                        best_n, best_e = pot_n, pot_e

            return best_n, best_e

        # Exclude start_pos from forbidden so the flood fill can enter it
        return recurse(start_pos, occ - {start_pos})

    @staticmethod
    def _bounded_reach(bw, bh, occ, pos, limit=250):
        """
        BFS from pos, stopping after `limit` cells found.
        Returns (cells_found, open_edges) -- fast proxy for Voronoi territory.
        """
        visited = {pos}
        q       = deque([pos])
        edges   = 0
        while q and len(visited) < limit:
            x, y = q.popleft()
            for dx, dy in NBRS:
                nx, ny = x + dx, y + dy
                nb     = (nx, ny)
                if not (0 <= nx < bw and 0 <= ny < bh) or nb in occ:
                    continue
                edges += 1
                if nb not in visited:
                    visited.add(nb)
                    q.append(nb)
        return len(visited), edges

    # ------------------------------------------------------------------
    # Position evaluation  (called at minimax leaves)
    # ------------------------------------------------------------------

    def _evaluate(self, bw, bh, occ, p_pos, o_pos):
        """
        Score from the AI (opponent) perspective -- higher = better for AI.

        Uses a bounded BFS (up to 80 cells each) to approximate the Voronoi
        territory difference.  Fast enough for minimax leaves while still
        driving the AI to aggressively cut off the player's reachable space.
        """
        bfs_occ   = occ - {p_pos, o_pos}
        o_n, o_e  = self._bounded_reach(bw, bh, bfs_occ, o_pos)
        p_n, p_e  = self._bounded_reach(bw, bh, bfs_occ, p_pos)
        return int(((o_n - p_n) * self.K1 + (o_e - p_e) * self.K2) * 1000)

    # ------------------------------------------------------------------
    # Minimax with alpha-beta pruning  (clean 1v1, no ad-hoc bonuses)
    # ------------------------------------------------------------------

    def _minimax(self, bw, bh, occ, p_pos, p_dir, o_pos, o_dir,
                 depth, maximizing, alpha=float('-inf'), beta=float('inf')):
        if depth == 0:
            return self._evaluate(bw, bh, occ, p_pos, o_pos)

        if maximizing:          # AI's turn
            best = float('-inf')
            for d in (UP, DOWN, LEFT, RIGHT):
                if d == OPPOSITE[o_dir]:
                    continue
                np_ = self._sim(o_pos, d)
                s   = (self._minimax(bw, bh, occ | {np_},
                                     p_pos, p_dir, np_, d,
                                     depth - 1, False, alpha, beta)
                       if self._valid(bw, bh, occ, np_) else -10000)
                best  = max(best, s)
                alpha = max(alpha, s)
                if beta <= alpha:
                    break
            return best

        else:                   # Player's turn
            best = float('inf')
            for d in (UP, DOWN, LEFT, RIGHT):
                if d == OPPOSITE[p_dir]:
                    continue
                np_ = self._sim(p_pos, d)
                s   = (self._minimax(bw, bh, occ | {np_},
                                     np_, d, o_pos, o_dir,
                                     depth - 1, True, alpha, beta)
                       if self._valid(bw, bh, occ, np_) else 10000)
                best = min(best, s)
                beta = min(beta, s)
                if beta <= alpha:
                    break
            return best

    # ------------------------------------------------------------------
    # Endgame: iterative-deepening greedy search
    # ------------------------------------------------------------------

    def _greedy_fill_count(self, bw, bh, occ_in, pos, direction):
        """
        Run the greedy wall-hugging heuristic to completion from pos/direction.

        Always picks the next cell with the most surrounding walls (fewest open
        neighbours), which is the space-filling rule from the post-mortem Phase 2.
        Returns the number of additional squares filled.
        """
        occ   = set(occ_in)
        start = len(occ)
        x, y  = pos
        d     = direction

        while True:
            best_d     = None
            most_walls = -1
            for nd in (UP, DOWN, LEFT, RIGHT):
                if nd == OPPOSITE[d]:
                    continue
                np_ = self._sim((x, y), nd)
                if self._valid(bw, bh, occ, np_):
                    walls = 4 - self._open_nbrs(bw, bh, occ, np_[0], np_[1])
                    # Current direction wins tiebreaks (avoids unnecessary turns)
                    if walls > most_walls or (walls == most_walls and nd == d):
                        most_walls = walls
                        best_d     = nd
            if best_d is None:
                break
            np_ = self._sim((x, y), best_d)
            occ.add(np_)
            x, y = np_
            d    = best_d

        return len(occ) - start

    def _endgame_search(self, bw, bh, occ, my_pos, my_dir):
        """
        Iterative-deepening search for the endgame (players separated).

        Exhaustively expands the move tree to self.endgame_depth, then runs
        _greedy_fill_count at each leaf to 'prime' the evaluator.  Whichever
        opening move leads to the best-primed greedy completion wins.

        Returns the direction maximising reachable squares.
        """
        def dfs(pos, d, cur_occ, depth):
            if depth == 0:
                return self._greedy_fill_count(bw, bh, cur_occ, pos, d)
            best = -1
            for nd in (UP, DOWN, LEFT, RIGHT):
                if nd == OPPOSITE[d]:
                    continue
                np_ = self._sim(pos, nd)
                if not self._valid(bw, bh, cur_occ, np_):
                    continue
                s = dfs(np_, nd, cur_occ | {np_}, depth - 1)
                if s > best:
                    best = s
            return max(best, 0)

        best_dir   = my_dir
        best_score = -1
        for d in (UP, DOWN, LEFT, RIGHT):
            if d == OPPOSITE[my_dir]:
                continue
            np_ = self._sim(my_pos, d)
            if not self._valid(bw, bh, occ, np_):
                continue
            s = dfs(np_, d, occ | {np_}, self.endgame_depth - 1)
            if s > best_score:
                best_score = s
                best_dir   = d
        return best_dir

    # ------------------------------------------------------------------
    # Main move selection
    # ------------------------------------------------------------------

    def choose_direction(self, opponent, player, bw, bh, occ):
        """
        Select the AI's next direction.

        Pipeline:
          1. Occasional deliberate mistake (difficulty scaling).
          2. Endgame (separated): iterative-deepening greedy search.
          3. Mid-game: compute Voronoi + APs once at root, then score each
             candidate move with the chamber-tree heuristic, combined with
             a minimax look-ahead using simple Voronoi at leaves.
        """
        p_pos = (player.x,   player.y)
        o_pos = (opponent.x, opponent.y)

        # Occasional deliberate mistake
        if random.random() < self.mistake_chance:
            vd = self._valid_dirs(o_pos, opponent.direction, bw, bh, occ)
            return random.choice(vd) if vd else opponent.direction

        # Treat head positions as accessible for BFS analysis
        bfs_occ = occ - {p_pos, o_pos}

        # Endgame: players are in separate components
        if self._separated(bw, bh, bfs_occ, p_pos, o_pos):
            return self._endgame_search(bw, bh, occ, o_pos, opponent.direction)

        # Mid-game: compute shared context once at root
        territories, _, _, _, _ = self._voronoi(bw, bh, bfs_occ, p_pos, o_pos)
        aps = self._articulation_points(bw, bh, bfs_occ)

        # Flip label perspective for evaluating the player's chamber tree
        flip = {
            pos: ('player'   if t == 'opponent' else
                  'opponent' if t == 'player'   else t)
            for pos, t in territories.items()
        }

        best_dir   = opponent.direction
        best_score = float('-inf')

        for d in (UP, DOWN, LEFT, RIGHT):
            if d == OPPOSITE[opponent.direction]:
                continue
            np_ = self._sim(o_pos, d)
            if not self._valid(bw, bh, occ, np_):
                continue

            new_occ = occ | {np_}
            new_bfs = new_occ - {np_, p_pos}

            # Chamber-tree score at this candidate move (computed once, at root)
            o_n, o_e = self._max_articulated_space(bw, bh, new_bfs, territories, np_,    aps)
            p_n, p_e = self._max_articulated_space(bw, bh, new_bfs, flip,        p_pos,  aps)
            chamber_score = int(((o_n - p_n) * self.K1 + (o_e - p_e) * self.K2) * 1000)

            # Minimax look-ahead with Voronoi at leaves
            mm_score = self._minimax(
                bw, bh, new_occ,
                p_pos, player.direction,
                np_,   d,
                self.search_depth - 1, False)

            total = chamber_score * 2 + mm_score
            if total > best_score:
                best_score = total
                best_dir   = d

        return best_dir


# ----------------------------------------------------------------------------------------------------------------------
# LightCycleGame  --  1v1 Qt window
# ----------------------------------------------------------------------------------------------------------------------

class LightCycleGame(QMainWindow):
    """Main game window for 1v1 Light Cycle."""

    def __init__(self, main_window, parent=None):
        super().__init__()
        self.main_window = main_window
        self.setWindowFlags(Qt.Window | Qt.CustomizeWindowHint |
                            Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.setWindowIcon(get_icon("lightcycle.svg"))
        self.title = "Light Cycle Game"

        self.board_width  = BOARD_WIDTH
        self.board_height = BOARD_HEIGHT
        self.cell_size    = CELL_SIZE
        self.game_speed   = 150

        self.timer      = QTimer(self)
        self.difficulty = "Medium"

        self.player       = None
        self.opponent     = None
        self.ai           = None
        self.game_started = False
        self.title_timer  = 0
        self.obstacles    = set()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def start_game(self):
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
        try:
            self.timer.timeout.disconnect()
        except TypeError:
            pass

        self.difficulty = "Hard"
        self.game_speed = {"Easy": 120, "Medium": 80, "Hard": 55, "Insane": 80}[self.difficulty]

        msg = (
            f"Welcome to Light Cycle -- {self.difficulty} Mode! (1v1)\n\n"
            "Rules:\n"
            "  - WASD to steer your cycle (gold)\n"
            "  - Avoid walls, your trail, and the opponent's trail (red)\n"
            "  - Last cycle standing wins!\n\n"
            "Click OK to start."
        )
        if QMessageBox.information(self, "Light Cycle", msg, QMessageBox.Ok) == QMessageBox.Ok:
            self.init_game()
            self.init_ui()
            self.timer.timeout.connect(self.update_game)
            self.timer.start(self.game_speed)
        else:
            self.close()

    def init_ui(self):
        w = self.board_width  * self.cell_size
        h = self.board_height * self.cell_size
        self.resize(w, h)
        self.setFixedSize(w, h)
        self.setWindowTitle(self.title)
        self.show()

    def init_game(self):
        self.game_started = False

        # Both start at mid-height, ~50 cells apart, heading at each other
        mid_y = self.board_height // 2
        self.player   = LightCycle(10,                    mid_y, RIGHT, PLAYER_COLOR)
        self.opponent = LightCycle(self.board_width - 11, mid_y, LEFT,  OPPONENT_COLOR)
        self.ai       = LightCycleAI(self.difficulty)

        # Clean 1v1 -- no random obstacles; pure strategy
        self.obstacles = set()

        self.game_started = True
        self.title_timer  = 10

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def keyPressEvent(self, event):
        if not self.game_started or not self.player.alive:
            return
        key_map = {Qt.Key_W: UP, Qt.Key_S: DOWN, Qt.Key_A: LEFT, Qt.Key_D: RIGHT}
        if event.key() in key_map:
            self.player.turn(key_map[event.key()])

    def keyReleaseEvent(self, event):
        pass

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)

        painter.fillRect(self.rect(), BACKGROUND_COLOR)

        # Grid
        painter.setPen(QPen(GRID_COLOR, 2))
        for x in range(0, self.board_width * self.cell_size, self.cell_size):
            painter.drawLine(x, 0, x, self.board_height * self.cell_size)
        for y in range(0, self.board_height * self.cell_size, self.cell_size):
            painter.drawLine(0, y, self.board_width * self.cell_size, y)

        painter.setPen(Qt.NoPen)

        def draw_cycle(cycle, glow_rgba):
            painter.setBrush(QBrush(QColor(*glow_rgba)))
            for x, y in cycle.trail:
                painter.drawRect(x * self.cell_size - 1, y * self.cell_size - 1,
                                 self.cell_size + 2, self.cell_size + 2)
            painter.setBrush(QBrush(cycle.color))
            for x, y in cycle.trail:
                painter.drawRect(x * self.cell_size, y * self.cell_size,
                                 self.cell_size, self.cell_size)

        if self.player:
            draw_cycle(self.player,   (255, 0, 255, 100))
        if self.opponent:
            draw_cycle(self.opponent, (0, 255, 255, 100))

        # Obstacles
        painter.setBrush(QBrush(QColor(128, 128, 128)))
        for x, y in self.obstacles:
            painter.drawRect(x * self.cell_size, y * self.cell_size,
                             self.cell_size, self.cell_size)

        painter.end()

        # Title animation
        if self.title_timer > 0:
            painter.begin(self)
            painter.setFont(QFont('Arial', 36, QFont.Bold))
            colors = [QColor(255, 0, 255), QColor(0, 255, 255),
                      QColor(0, 255, 0),   QColor(255, 255, 0)]
            ci    = (self.title_timer // 10) % len(colors)
            title = "LIGHT CYCLES GAME"
            tr    = painter.fontMetrics().boundingRect(title)
            tx    = (self.width()  - tr.width())  // 2
            ty    = self.height() // 2
            glow  = [QColor(255, 0, 255, 150), QColor(0, 255, 255, 150)]
            gi    = (self.title_timer // 5) % len(glow)
            painter.setPen(glow[gi])
            for ox, oy in ((-2, -2), (2, -2), (-2, 2), (2, 2)):
                painter.drawText(tx + ox, ty + oy, title)
            painter.setPen(colors[ci])
            painter.drawText(tx, ty, title)
            painter.end()

    # ------------------------------------------------------------------
    # Game loop
    # ------------------------------------------------------------------

    def update_game(self):
        if not getattr(self, 'game_started', False):
            return

        if self.title_timer > 0:
            self.title_timer -= 1
            self.update()
            return

        # Move player
        self.player.move()

        # Occupied set for AI: all trail cells + obstacles
        occ = set(self.player.trail) | set(self.opponent.trail) | self.obstacles

        # AI picks a direction and moves
        best_dir = self.ai.choose_direction(
            self.opponent, self.player,
            self.board_width, self.board_height, occ)
        self.opponent.turn(best_dir)
        self.opponent.move()

        # Collision checks
        p_crashed = self.player.check_collision(
            self.board_width, self.board_height,
            set(self.opponent.trail), self.obstacles)
        o_crashed = self.opponent.check_collision(
            self.board_width, self.board_height,
            set(self.player.trail), self.obstacles)

        if p_crashed and o_crashed:
            self.game_over("Both crashed -- it's a draw!")
        elif p_crashed:
            self.game_over("You crashed!  Opponent wins.")
        elif o_crashed:
            self.game_over("Opponent crashed!  You win!")
        else:
            self.update()

    # ------------------------------------------------------------------
    # Game over / cleanup
    # ------------------------------------------------------------------

    def game_over(self, message):
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
        self.game_started = False
        reply = QMessageBox.question(self, "Game Over",
                                     f"{message}\n\nPlay again?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.start_game()
        else:
            self.cleanup_and_close()

    def cleanup_and_close(self):
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
        try:
            self.timer.timeout.disconnect()
        except TypeError:
            pass
        self.player       = None
        self.opponent     = None
        self.ai           = None
        self.game_started = False
        self.close()
        if hasattr(self, 'main_window') and self.main_window:
            if hasattr(self.main_window, 'current_game'):
                self.main_window.current_game = None

    def closeEvent(self, event):
        self.cleanup_and_close()
        event.accept()