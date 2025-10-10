# Credit: https://www.a1k0n.net/code/tron.html

import warnings

import random

warnings.filterwarnings("ignore")

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont
from PyQt5.QtWidgets import (QMainWindow, QMessageBox, QDialog, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QButtonGroup)

from coralnet_toolbox.Icons import get_icon

# ----------------------------------------------------------------------------------------------------------------------
# Constants / Configurations
# ----------------------------------------------------------------------------------------------------------------------

# Game settings
BOARD_WIDTH = 80
BOARD_HEIGHT = 60
CELL_SIZE = 12
GRID_LINE_WIDTH = 2

# Colors
BACKGROUND_COLOR = QColor(0, 0, 0)  # Black
GRID_COLOR = QColor(0, 100, 100)  # Darker cyan
PLAYER_COLOR = QColor(255, 215, 0)  # Gold
OPPONENT_COLOR = QColor(255, 0, 0)  # Red

# Directions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# Game states
PLAYER_ALIVE = 0
PLAYER_CRASHED = 1
OPPONENT_ALIVE = 0
OPPONENT_CRASHED = 1

# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class DifficultyDialog(QDialog):
    """
    Dialog for selecting game difficulty level.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_difficulty = "Medium"  # Default
        self.init_ui()

    def init_ui(self):
        """Initialize the difficulty selection UI."""
        self.setWindowTitle("Select Difficulty")
        self.setModal(True)
        self.setFixedSize(400, 300)

        layout = QVBoxLayout()

        # Title
        title = QLabel("Choose Difficulty Level")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(title.font())
        title.font().setPointSize(16)
        title.font().setBold(True)
        layout.addWidget(title)

        # Difficulty descriptions
        difficulties = {
            "Easy": "🟢 Slower opponent, larger board\nPerfect for beginners!",
            "Medium": "🟡 Balanced gameplay\nThe classic experience",
            "Hard": "🟠 Faster opponent, smaller board\nFor experienced players",
            "Insane": "🔴 Lightning fast opponent\nOnly for the brave!"
        }

        self.button_group = QButtonGroup()

        for i, (difficulty, description) in enumerate(difficulties.items()):
            btn = QPushButton(f"{difficulty}\n{description}")
            btn.setFixedHeight(50)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, d=difficulty: self.set_difficulty(d))
            self.button_group.addButton(btn, i)
            layout.addWidget(btn)

            if difficulty == "Medium":  # Default selection
                btn.setChecked(True)

        # OK/Cancel buttons
        button_layout = QHBoxLayout()
        ok_btn = QPushButton("Start Game")
        cancel_btn = QPushButton("Cancel")

        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)

        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(ok_btn)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def set_difficulty(self, difficulty):
        """Set the selected difficulty."""
        self.selected_difficulty = difficulty

    def get_difficulty(self):
        """Return the selected difficulty."""
        return self.selected_difficulty


class LightCycle:
    """
    Represents a light cycle in the game.
    """

    def __init__(self, start_x, start_y, direction, color):
        self.x = start_x
        self.y = start_y
        self.direction = direction
        self.color = color
        self.trail = [(start_x, start_y)]  # List of (x, y) positions
        self.alive = True

    def move(self):
        """Move the cycle in its current direction."""
        if not self.alive:
            return

        if self.direction == UP:
            self.y -= 1
        elif self.direction == DOWN:
            self.y += 1
        elif self.direction == LEFT:
            self.x -= 1
        elif self.direction == RIGHT:
            self.x += 1

        # Add current position to trail
        self.trail.append((self.x, self.y))

    def turn(self, new_direction):
        """Change direction if it's not reversing."""
        # Prevent immediate reversal
        if (self.direction == UP and new_direction == DOWN) or \
           (self.direction == DOWN and new_direction == UP) or \
           (self.direction == LEFT and new_direction == RIGHT) or \
           (self.direction == RIGHT and new_direction == LEFT):
            return
        self.direction = new_direction

    def check_collision(self, board_width, board_height, opponent_trail):
        """Check if the cycle has crashed."""
        # Wall collision
        if self.x < 0 or self.x >= board_width or self.y < 0 or self.y >= board_height:
            self.alive = False
            return True

        # Self collision (hit own trail)
        if (self.x, self.y) in self.trail[:-1]:  # Exclude current position
            self.alive = False
            return True

        # Opponent collision
        if (self.x, self.y) in opponent_trail:
            self.alive = False
            return True

        return False


class LightCycleAI:
    """
    Advanced AI for Light Cycle based on Google AI Challenge winning strategies.
    Implements Voronoi heuristic, articulation points, space-filling, and minimax.
    """
    
    def __init__(self, difficulty="Medium"):
        self.difficulty = difficulty
        self.think_ahead = 4
        self.mistake_chance = 0.04
        
        # Difficulty settings
        difficulty_settings = {
            "Easy": {"think_ahead": 2, "mistake_chance": 0.15, "search_depth": 3},
            "Medium": {"think_ahead": 3, "mistake_chance": 0.08, "search_depth": 4},
            "Hard": {"think_ahead": 4, "mistake_chance": 0.04, "search_depth": 5},
            "Insane": {"think_ahead": 5, "mistake_chance": 0.02, "search_depth": 6}
        }
        
        if difficulty in difficulty_settings:
            settings = difficulty_settings[difficulty]
            self.think_ahead = settings["think_ahead"]
            self.mistake_chance = settings["mistake_chance"]
            self.search_depth = settings["search_depth"]
        else:
            self.search_depth = 5
            
        # Voronoi heuristic coefficients (from data mining in original contest)
        self.node_coefficient = 0.055
        self.edge_coefficient = 0.194

    def compute_voronoi_territories(self, board_width, board_height, occupied_positions, player_pos, opponent_pos):
        """
        Compute Voronoi diagram - for each cell, determine which player can reach it first.
        Returns (player_territory, opponent_territory, player_edges, opponent_edges)
        """
        # Use BFS to find shortest distance from each player to all reachable cells
        distances = {}
        territories = {}
        
        # Initialize BFS for both players
        from collections import deque
        queue = deque()
        
        # Add starting positions
        if player_pos not in occupied_positions:
            queue.append((player_pos[0], player_pos[1], 0, 'player'))
            distances[player_pos] = 0
            territories[player_pos] = 'player'
            
        if opponent_pos not in occupied_positions:
            queue.append((opponent_pos[0], opponent_pos[1], 0, 'opponent'))
            distances[opponent_pos] = 0
            territories[opponent_pos] = 'opponent'
        
        # BFS to find territories
        while queue:
            x, y, dist, owner = queue.popleft()
            current_pos = (x, y)
            
            # Skip if we've found a shorter path already
            if current_pos in distances and distances[current_pos] < dist:
                continue
                
            # Explore neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                new_pos = (nx, ny)
                
                # Check bounds and walls
                if (nx < 0 or nx >= board_width or ny < 0 or ny >= board_height or
                    new_pos in occupied_positions):
                    continue
                    
                new_dist = dist + 1
                
                # If unvisited or we found a shorter path
                if new_pos not in distances or distances[new_pos] > new_dist:
                    distances[new_pos] = new_dist
                    territories[new_pos] = owner
                    queue.append((nx, ny, new_dist, owner))
                elif distances[new_pos] == new_dist and territories[new_pos] != owner:
                    # Tie - mark as neutral
                    territories[new_pos] = 'neutral'
        
        # Count territories and edges
        player_nodes = sum(1 for pos, owner in territories.items() if owner == 'player')
        opponent_nodes = sum(1 for pos, owner in territories.items() if owner == 'opponent')
        
        # Count edges (neighbors of territory cells)
        player_edges = 0
        opponent_edges = 0
        
        for (x, y), owner in territories.items():
            if owner == 'player':
                # Count open neighbors
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < board_width and 0 <= ny < board_height and
                        (nx, ny) not in occupied_positions):
                        player_edges += 1
            elif owner == 'opponent':
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < board_width and 0 <= ny < board_height and
                        (nx, ny) not in occupied_positions):
                        opponent_edges += 1
        
        return player_nodes, opponent_nodes, player_edges, opponent_edges

    def find_articulation_points(self, board_width, board_height, occupied_positions):
        """
        Find articulation points (cut vertices) in the graph of free spaces.
        These are points whose removal would disconnect the graph.
        """
        # Build adjacency list for free spaces
        free_spaces = []
        space_to_index = {}
        
        for x in range(board_width):
            for y in range(board_height):
                if (x, y) not in occupied_positions:
                    space_to_index[(x, y)] = len(free_spaces)
                    free_spaces.append((x, y))
        
        if len(free_spaces) <= 2:
            return set()
            
        n = len(free_spaces)
        adj = [[] for _ in range(n)]
        
        # Build adjacency list
        for i, (x, y) in enumerate(free_spaces):
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) in space_to_index:
                    j = space_to_index[(nx, ny)]
                    adj[i].append(j)
        
        # Find articulation points using Tarjan's algorithm
        visited = [False] * n
        disc = [0] * n
        low = [0] * n
        parent = [-1] * n
        ap = [False] * n
        time = [0]
        
        def bridge_util(u):
            children = 0
            visited[u] = True
            disc[u] = low[u] = time[0]
            time[0] += 1
            
            for v in adj[u]:
                if not visited[v]:
                    children += 1
                    parent[v] = u
                    bridge_util(v)
                    
                    low[u] = min(low[u], low[v])
                    
                    # Root of DFS tree is articulation point if it has more than one child
                    if parent[u] == -1 and children > 1:
                        ap[u] = True
                        
                    # Non-root is articulation point if removing it disconnects the tree
                    if parent[u] != -1 and low[v] >= disc[u]:
                        ap[u] = True
                        
                elif v != parent[u]:
                    low[u] = min(low[u], disc[v])
        
        # Find articulation points for all components
        for i in range(n):
            if not visited[i]:
                bridge_util(i)
        
        # Convert back to coordinates
        articulation_points = set()
        for i in range(n):
            if ap[i]:
                articulation_points.add(free_spaces[i])
                
        return articulation_points

    def compute_checkerboard_territories(self, board_width, board_height, occupied_positions, player_pos, opponent_pos):
        """
        Compute territories considering checkerboard parity constraints.
        In Tron, players alternate between "red" and "black" squares, so surplus squares
        of one color will be wasted.
        """
        player_nodes, opponent_nodes, player_edges, opponent_edges = self.compute_voronoi_territories(
            board_width, board_height, occupied_positions, player_pos, opponent_pos)
        
        # Count red/black squares for each player's territory
        player_red = 0
        player_black = 0
        opponent_red = 0
        opponent_black = 0
        
        # Re-traverse territories to count colors
        distances = {}
        territories = {}
        
        # BFS again to get territories
        from collections import deque
        queue = deque()
        
        if player_pos not in occupied_positions:
            queue.append((player_pos[0], player_pos[1], 0, 'player'))
            distances[player_pos] = 0
            territories[player_pos] = 'player'
            
        if opponent_pos not in occupied_positions:
            queue.append((opponent_pos[0], opponent_pos[1], 0, 'opponent'))
            distances[opponent_pos] = 0
            territories[opponent_pos] = 'opponent'
        
        while queue:
            x, y, dist, owner = queue.popleft()
            current_pos = (x, y)
            
            if current_pos in distances and distances[current_pos] < dist:
                continue
                
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                new_pos = (nx, ny)
                
                if (nx < 0 or nx >= board_width or ny < 0 or ny >= board_height or
                    new_pos in occupied_positions):
                    continue
                    
                new_dist = dist + 1
                
                if new_pos not in distances or distances[new_pos] > new_dist:
                    distances[new_pos] = new_dist
                    territories[new_pos] = owner
                    queue.append((nx, ny, new_dist, owner))
                elif distances[new_pos] == new_dist and territories[new_pos] != owner:
                    territories[new_pos] = 'neutral'
        
        # Count red/black squares
        for (x, y), owner in territories.items():
            is_red = (x + y) % 2 == 0  # Checkerboard pattern
            if owner == 'player':
                if is_red:
                    player_red += 1
                else:
                    player_black += 1
            elif owner == 'opponent':
                if is_red:
                    opponent_red += 1
                else:
                    opponent_black += 1
        
        # Calculate effective territory considering parity constraints
        player_effective = min(player_red, player_black) * 2 + abs(player_red - player_black)
        opponent_effective = min(opponent_red, opponent_black) * 2 + abs(opponent_red - opponent_black)
        
        return player_effective, opponent_effective

    def count_edge_removal(self, board_width, board_height, occupied_positions, x, y):
        """
        Count how many edges would be removed by moving to position (x, y).
        Lower values indicate better space-filling moves.
        """
        if (x, y) in occupied_positions:
            return float('inf')
            
        edge_count = 0
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < board_width and 0 <= ny < board_height and
                (nx, ny) not in occupied_positions):
                edge_count += 1
                
        return 4 - edge_count  # Higher removal = worse move

    def evaluate_position(self, board_width, board_height, occupied_positions, player_pos, opponent_pos):
        """
        Advanced position evaluation using Voronoi heuristic and space-filling principles.
        Returns a score from the opponent's perspective (higher = better for opponent).
        """
        # Check for immediate game over
        if player_pos in occupied_positions:
            return 10000  # Opponent wins
        if opponent_pos in occupied_positions:
            return -10000  # Player wins
        
        # Compute Voronoi territories
        player_nodes, opponent_nodes, player_edges, opponent_edges = self.compute_voronoi_territories(
            board_width, board_height, occupied_positions, player_pos, opponent_pos)
        
        # If players are separated, use territory count with endgame bonus
        if self.are_players_separated(board_width, board_height, occupied_positions, player_pos, opponent_pos):
            # Use checkerboard-aware territory calculation for endgame
            player_effective, opponent_effective = self.compute_checkerboard_territories(
                board_width, board_height, occupied_positions, player_pos, opponent_pos)
            territory_diff = opponent_effective - player_effective
            return territory_diff * 1000  # Large multiplier for certain endgame
        
        # Use learned coefficients from Google AI Challenge data mining
        node_score = (opponent_nodes - player_nodes) * self.node_coefficient
        edge_score = (opponent_edges - player_edges) * self.edge_coefficient
        
        base_score = (node_score + edge_score) * 1000  # Scale up for integer math
        
        # Add space-filling bonus - prefer moves that don't waste edges
        space_filling_bonus = 0
        
        # Find articulation points to avoid creating bottlenecks
        articulation_points = self.find_articulation_points(board_width, board_height, occupied_positions)
        
        # Penalty for being near articulation points (might get cut off)
        articulation_penalty = 0
        for ax, ay in articulation_points:
            opponent_dist = abs(opponent_pos[0] - ax) + abs(opponent_pos[1] - ay)
            player_dist = abs(player_pos[0] - ax) + abs(player_pos[1] - ay)
            if opponent_dist <= 2:
                articulation_penalty -= 50
            if player_dist <= 2:
                articulation_penalty += 50
        
        return int(base_score + space_filling_bonus + articulation_penalty)

    def are_players_separated(self, board_width, board_height, occupied_positions, player_pos, opponent_pos):
        """
        Check if players are in separate connected components (endgame condition).
        """
        # BFS from player position
        visited = set()
        queue = [player_pos]
        visited.add(player_pos)
        
        while queue:
            x, y = queue.pop(0)
            
            if (x, y) == opponent_pos:
                return False  # Players are connected
                
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < board_width and 0 <= ny < board_height and
                    (nx, ny) not in occupied_positions and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        
        return True  # Opponent not reachable from player

    def minimax(self, board_width, board_height, occupied_positions, player_pos, player_dir, 
                opponent_pos, opponent_dir, depth, maximizing_player, alpha=float('-inf'), beta=float('inf')):
        """
        Minimax search with alpha-beta pruning.
        """
        # Base case - evaluate position
        if depth == 0:
            return self.evaluate_position(board_width, board_height, occupied_positions, player_pos, opponent_pos)
        
        if maximizing_player:  # Opponent's turn (AI)
            max_eval = float('-inf')
            for direction in [UP, DOWN, LEFT, RIGHT]:
                # Skip invalid moves (reversing direction)
                if ((opponent_dir == UP and direction == DOWN) or
                    (opponent_dir == DOWN and direction == UP) or
                    (opponent_dir == LEFT and direction == RIGHT) or
                    (opponent_dir == RIGHT and direction == LEFT)):
                    continue
                
                # Simulate move
                new_opponent_pos = self.simulate_move(opponent_pos, direction)
                if self.is_valid_move(board_width, board_height, occupied_positions, new_opponent_pos):
                    new_occupied = occupied_positions.union({new_opponent_pos})
                    eval_score = self.minimax(board_width, board_height, new_occupied, player_pos, player_dir,
                                            new_opponent_pos, direction, depth - 1, False, alpha, beta)
                    max_eval = max(max_eval, eval_score)
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break  # Alpha-beta pruning
                else:
                    # Invalid move = loss
                    max_eval = max(max_eval, -10000)
            return max_eval
        else:  # Player's turn
            min_eval = float('inf')
            for direction in [UP, DOWN, LEFT, RIGHT]:
                # Skip invalid moves
                if ((player_dir == UP and direction == DOWN) or
                    (player_dir == DOWN and direction == UP) or
                    (player_dir == LEFT and direction == RIGHT) or
                    (player_dir == RIGHT and direction == LEFT)):
                    continue
                
                # Simulate move
                new_player_pos = self.simulate_move(player_pos, direction)
                if self.is_valid_move(board_width, board_height, occupied_positions, new_player_pos):
                    new_occupied = occupied_positions.union({new_player_pos})
                    eval_score = self.minimax(board_width, board_height, new_occupied, new_player_pos, direction,
                                            opponent_pos, opponent_dir, depth - 1, True, alpha, beta)
                    min_eval = min(min_eval, eval_score)
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break  # Alpha-beta pruning
                else:
                    # Invalid move = opponent wins
                    min_eval = min(min_eval, 10000)
            return min_eval

    def simulate_move(self, pos, direction):
        """Simulate a move in the given direction."""
        x, y = pos
        if direction == UP:
            return (x, y - 1)
        elif direction == DOWN:
            return (x, y + 1)
        elif direction == LEFT:
            return (x - 1, y)
        elif direction == RIGHT:
            return (x + 1, y)
        return pos

    def is_valid_move(self, board_width, board_height, occupied_positions, pos):
        """Check if a move to the given position is valid."""
        x, y = pos
        return (0 <= x < board_width and 0 <= y < board_height and pos not in occupied_positions)

    def choose_direction(self, opponent, player, board_width, board_height):
        """
        Choose the best direction using advanced AI techniques.
        """
        # Occasionally make mistakes based on difficulty
        if random.random() < self.mistake_chance:
            valid_directions = []
            for direction in [UP, DOWN, LEFT, RIGHT]:
                # Don't reverse direction
                if ((opponent.direction == UP and direction == DOWN) or
                    (opponent.direction == DOWN and direction == UP) or
                    (opponent.direction == LEFT and direction == RIGHT) or
                    (opponent.direction == RIGHT and direction == LEFT)):
                    continue
                    
                # Check if move is valid
                new_pos = self.simulate_move((opponent.x, opponent.y), direction)
                occupied_positions = set(opponent.trail + player.trail)
                if self.is_valid_move(board_width, board_height, occupied_positions, new_pos):
                    valid_directions.append(direction)
            
            if valid_directions:
                return random.choice(valid_directions)
        
        # Use minimax search for best move
        occupied_positions = set(opponent.trail + player.trail)
        best_direction = opponent.direction
        best_score = float('-inf')
        
        for direction in [UP, DOWN, LEFT, RIGHT]:
            # Don't reverse direction
            if ((opponent.direction == UP and direction == DOWN) or
                (opponent.direction == DOWN and direction == UP) or
                (opponent.direction == LEFT and direction == RIGHT) or
                (opponent.direction == RIGHT and direction == LEFT)):
                continue
            
            # Check if move is valid
            new_opponent_pos = self.simulate_move((opponent.x, opponent.y), direction)
            if not self.is_valid_move(board_width, board_height, occupied_positions, new_opponent_pos):
                continue
            
            # Use minimax to evaluate this move
            new_occupied = occupied_positions.union({new_opponent_pos})
            score = self.minimax(board_width, board_height, new_occupied,
                               (player.x, player.y), player.direction,
                               new_opponent_pos, direction, 
                               min(self.search_depth, 4), False)  # Start with player's turn
            
            if score > best_score:
                best_score = score
                best_direction = direction
        
        # Fallback to space-filling heuristic if no good moves found
        if best_score == float('-inf'):
            best_edge_removal = float('inf')
            for direction in [UP, DOWN, LEFT, RIGHT]:
                if ((opponent.direction == UP and direction == DOWN) or
                    (opponent.direction == DOWN and direction == UP) or
                    (opponent.direction == LEFT and direction == RIGHT) or
                    (opponent.direction == RIGHT and direction == LEFT)):
                    continue
                
                new_pos = self.simulate_move((opponent.x, opponent.y), direction)
                if self.is_valid_move(board_width, board_height, occupied_positions, new_pos):
                    edge_removal = self.count_edge_removal(board_width, board_height, occupied_positions,
                                                         new_pos[0], new_pos[1])
                    if edge_removal < best_edge_removal:
                        best_edge_removal = edge_removal
                        best_direction = direction
        
        return best_direction


class LightCycleGame(QMainWindow):
    """
    Main game window for the Light Cycle game.
    """

    def __init__(self, main_window, parent=None):
        super().__init__()
        self.main_window = main_window

        # Remove minimize and maximize buttons
        self.setWindowFlags(Qt.Window | Qt.CustomizeWindowHint |
                            Qt.WindowTitleHint | Qt.WindowCloseButtonHint)

        # Set the window icon
        self.setWindowIcon(get_icon("lightcycle.png"))
        # Set the window title
        self.title = "Light Cycle Game"

        # Game settings
        self.board_width = BOARD_WIDTH
        self.board_height = BOARD_HEIGHT
        self.cell_size = CELL_SIZE
        self.game_speed = 150  # milliseconds

        # Initialize timer
        self.timer = QTimer(self)
        self.difficulty = "Medium"

        # Initialize game
        self.player = None
        self.opponent = None
        self.ai = None  # AI opponent
        self.game_started = False
        self.title_timer = 0  # For title animation
        self.pressed_keys = set()  # Track pressed movement keys for speed boost
        self.player_speed = 1  # Speed multiplier for player

    def start_game(self):
        """Start the game by initializing."""
        # Hardcode difficulty to Hard for AI opponent
        self.difficulty = "Hard"
        
        # Always reset game speed to baseline for this difficulty
        speed_settings = {
            "Easy": 200,
            "Medium": 150,
            "Hard": 100,
            "Insane": 75
        }
        self.game_speed = speed_settings[self.difficulty]
        
        # Stop any existing timer to prevent multiple timers running
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()

        # Show welcome dialog
        welcome_msg = (
            f"Welcome to Light Cycle - {self.difficulty} Mode!\n\n"
            "Rules:\n"
            " - Use WASD keys to turn your cycle (gold)\n"
            " - Avoid walls, your trail, and the opponent's trail (red)\n"
            " - The opponent moves automatically\n"
            " - Last cycle standing wins!\n\n"
            "Click 'OK' to start playing."
        )
        if QMessageBox.information(self, "Welcome to Light Cycle", welcome_msg, QMessageBox.Ok) == QMessageBox.Ok:
            self.init_game()
            self.init_ui()
            self.timer.timeout.connect(self.update_game)
            self.timer.start(self.game_speed)
        else:
            self.close()

    def init_ui(self):
        """Set up the user interface."""
        window_width = self.board_width * self.cell_size
        window_height = self.board_height * self.cell_size

        self.resize(window_width, window_height)
        self.setFixedSize(window_width, window_height)
        self.setWindowTitle(self.title)
        self.show()

    def init_game(self):
        """Initialize the game state."""
        # Reset game state completely
        self.game_started = False
        self.pressed_keys = set()
        self.player_speed = 1
        
        # Player starts at bottom left, moving right
        self.player = LightCycle(5, self.board_height - 5, RIGHT, PLAYER_COLOR)

        # Opponent starts at top right, moving left
        self.opponent = LightCycle(self.board_width - 5, 5, LEFT, OPPONENT_COLOR)

        # Initialize AI with current difficulty
        self.ai = LightCycleAI(self.difficulty)

        self.game_started = True
        self.title_timer = 10  # Show title for 0.5 seconds (30 frames at ~60fps)

    def paintEvent(self, event):
        """Render the game board."""
        painter = QPainter()
        painter.begin(self)

        # Draw background
        painter.fillRect(self.rect(), BACKGROUND_COLOR)

        # Draw grid with synth-wave glow
        grid_pen = QPen(GRID_COLOR, 2)  # Thicker lines
        painter.setPen(grid_pen)
        for x in range(0, self.board_width * self.cell_size, self.cell_size):
            painter.drawLine(x, 0, x, self.board_height * self.cell_size)
        for y in range(0, self.board_height * self.cell_size, self.cell_size):
            painter.drawLine(0, y, self.board_width * self.cell_size, y)

        # Draw trails with glow effect
        if self.player:
            # Draw glow/shadow first
            painter.setPen(Qt.NoPen)
            glow_color = QColor(255, 0, 255, 100)  # Semi-transparent magenta glow
            painter.setBrush(QBrush(glow_color))
            for x, y in self.player.trail:
                painter.drawRect((x * self.cell_size) - 1, (y * self.cell_size) - 1,
                                 self.cell_size + 2, self.cell_size + 2)

            # Draw main trail
            painter.setBrush(QBrush(self.player.color))
            for x, y in self.player.trail:
                painter.drawRect(x * self.cell_size, y * self.cell_size,
                                 self.cell_size, self.cell_size)

        if self.opponent:
            # Draw glow/shadow first
            painter.setPen(Qt.NoPen)
            glow_color = QColor(0, 255, 255, 100)  # Semi-transparent cyan glow
            painter.setBrush(QBrush(glow_color))
            for x, y in self.opponent.trail:
                painter.drawRect((x * self.cell_size) - 1, (y * self.cell_size) - 1,
                                 self.cell_size + 2, self.cell_size + 2)

            # Draw main trail
            painter.setBrush(QBrush(self.opponent.color))
            for x, y in self.opponent.trail:
                painter.drawRect(x * self.cell_size, y * self.cell_size,
                                 self.cell_size, self.cell_size)

        painter.end()

        # Draw synth-wave title animation
        if self.title_timer > 0:
            painter.begin(self)
            title_font = QFont('Arial', 36, QFont.Bold)
            painter.setFont(title_font)

            # Create animated colors
            colors = [QColor(255, 0, 255), QColor(0, 255, 255), QColor(0, 255, 0), QColor(255, 255, 0)]
            color_index = (self.title_timer // 10) % len(colors)
            painter.setPen(colors[color_index])

            title_text = "LIGHT CYCLES"
            text_rect = painter.fontMetrics().boundingRect(title_text)
            x = (self.width() - text_rect.width()) // 2
            y = self.height() // 2

            # Add glow effect to title
            glow_colors = [QColor(255, 0, 255, 150), QColor(0, 255, 255, 150)]
            glow_index = (self.title_timer // 5) % len(glow_colors)
            painter.setPen(glow_colors[glow_index])
            painter.drawText(x - 2, y - 2, title_text)
            painter.drawText(x + 2, y - 2, title_text)
            painter.drawText(x - 2, y + 2, title_text)
            painter.drawText(x + 2, y + 2, title_text)

            painter.setPen(colors[color_index])
            painter.drawText(x, y, title_text)
            painter.end()

    def update_game(self):
        """Handle game updates."""
        # Safety check - don't update if game is not properly initialized
        if not hasattr(self, 'game_started') or not self.game_started:
            return
            
        if self.title_timer > 0:
            self.title_timer -= 1
            self.update()
            return

        if self.game_started:
            # Move player multiple times based on speed
            for _ in range(self.player_speed):
                self.player.move()
            
            # AI decides opponent's next move
            if self.ai and self.opponent.alive:
                best_direction = self.ai.choose_direction(
                    self.opponent, self.player, self.board_width, self.board_height)
                self.opponent.turn(best_direction)
            
            # Move opponent
            self.opponent.move()

            # Check collisions
            player_crashed = self.player.check_collision(self.board_width, self.board_height, self.opponent.trail)
            opponent_crashed = self.opponent.check_collision(self.board_width, self.board_height, self.player.trail)

            # Check win conditions
            if player_crashed and opponent_crashed:
                self.game_over("It's a tie!")
            elif player_crashed:
                self.game_over("You crashed! Opponent wins.")
            elif opponent_crashed:
                self.game_over("Opponent crashed! You win!")

            self.update()

    def keyPressEvent(self, event):
        """Handle key presses for player control."""
        if not self.game_started or not self.player.alive:
            return

        key = event.key()
        movement_keys = {Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D}
        
        if key in movement_keys:
            self.pressed_keys.add(key)
            self.adjust_speed()
            
            # Handle movement
            if key == Qt.Key_W:
                self.player.turn(UP)
            elif key == Qt.Key_S:
                self.player.turn(DOWN)
            elif key == Qt.Key_A:
                self.player.turn(LEFT)
            elif key == Qt.Key_D:
                self.player.turn(RIGHT)

    def keyReleaseEvent(self, event):
        """Handle key releases for speed control."""
        key = event.key()
        movement_keys = {Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D}
        
        if key in movement_keys and key in self.pressed_keys:
            self.pressed_keys.remove(key)
            self.adjust_speed()

    def adjust_speed(self):
        """Adjust player speed based on pressed keys."""
        self.player_speed = 2 if self.pressed_keys else 1

    def game_over(self, message):
        """Handle game over."""
        # Stop the timer immediately to prevent multiple calls
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
        
        # Reset game state to prevent further updates
        self.game_started = False
        
        # Show game over dialog
        reply = QMessageBox.question(self, "Game Over", f"{message}\n\nPlay again?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.start_game()
        else:
            # Close the window when user doesn't want to play again
            self.close()

    def closeEvent(self, event):
        """Handle window close."""
        # Ensure timer is stopped and game state is reset
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
        self.game_started = False
        event.accept()
