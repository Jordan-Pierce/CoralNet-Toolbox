# Credit: https://www.a1k0n.net/code/tron.html

import random
import warnings

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

    def check_collision(self, board_width, board_height, opponent_trail, obstacles):
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

        # Obstacle collision
        if (self.x, self.y) in obstacles:
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
            "Hard": {"think_ahead": 4, "mistake_chance": 0.04, "search_depth": 6},
            "Insane": {"think_ahead": 5, "mistake_chance": 0.01, "search_depth": 8}
        }
        
        if difficulty in difficulty_settings:
            settings = difficulty_settings[difficulty]
            self.think_ahead = settings["think_ahead"]
            self.mistake_chance = settings["mistake_chance"]
            self.search_depth = settings["search_depth"]
        else:
            self.search_depth = 5
            
        # Voronoi heuristic coefficients (from data mining in original contest)
        self.node_coefficient = 0.080
        self.edge_coefficient = 0.150

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
                if (nx < 0 or 
                    nx >= board_width or 
                    ny < 0 or ny >= board_height or 
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
                    if (0 <= nx < board_width and 
                        0 <= ny < board_height and
                        (nx, ny) not in occupied_positions):
                        player_edges += 1
            elif owner == 'opponent':
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < board_width and 
                        0 <= ny < board_height and
                        (nx, ny) not in occupied_positions):
                        opponent_edges += 1
        
        # Find neutral (battlefront) squares
        neutral_squares = {pos for pos, owner in territories.items() if owner == 'neutral'}
        
        return player_nodes, opponent_nodes, player_edges, opponent_edges, neutral_squares

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
        player_nodes, opponent_nodes, player_edges, opponent_edges, _ = self.compute_voronoi_territories(
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
        player_nodes, opponent_nodes, player_edges, opponent_edges, neutral_squares = self.compute_voronoi_territories(
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
        
        # --- START OF AGGRESSIVENESS CODE ---
        # 1. Proximity Bonus: Reward AI for being close to the player.
        # The original formula (e.g., 5000 / distance) creates a weak gradient at long
        # range, leading to passive play. A linear bonus provides a stronger, constant
        # incentive to engage the opponent.
        distance = abs(player_pos[0] - opponent_pos[0]) + abs(player_pos[1] - opponent_pos[1])
        max_dist = board_width + board_height
        # A high weight (e.g., 500) makes the score change from moving one step closer
        # comparable to the score from gaining territory, encouraging aggression.
        proximity_bonus = (max_dist - distance) * 1000

        # 2. Battlefront Bonus: Reward AI for controlling the contested border.
        # Being on the "battlefront" (neutral squares) is a key strategic goal.
        # This bonus should be large to heavily incentivize moving to contested areas.
        battlefront_bonus = 0
        if opponent_pos in neutral_squares:
            battlefront_bonus = 10000
        # --- END OF AGGRESSIVENESS CODE ---
        
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
        
        return int(base_score + space_filling_bonus + articulation_penalty + proximity_bonus + battlefront_bonus)

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
                    
                    # --- START OF PURSUIT REVERSAL BONUS ---
                    # This tactical bonus encourages the AI to make sharp turns
                    # to cut off a player it is trailing ("pursuing").
                    reversal_bonus = 0
                    distance = abs(player_pos[0] - opponent_pos[0]) + abs(player_pos[1] - opponent_pos[1])
                    if 2 < distance < 15:  # Only applies at medium-close range
                        is_behind = False
                        # Check if AI is generally "behind" the player, relative to its direction of travel.
                        if opponent_dir == RIGHT and player_pos[0] > opponent_pos[0]: 
                            is_behind = True
                        elif opponent_dir == LEFT and player_pos[0] < opponent_pos[0]: 
                            is_behind = True
                        elif opponent_dir == DOWN and player_pos[1] > opponent_pos[1]: 
                            is_behind = True
                        elif opponent_dir == UP and player_pos[1] < opponent_pos[1]: 
                            is_behind = True

                        is_turn = (opponent_dir != direction)

                        if is_behind and is_turn:
                            # A large bonus makes this a high-priority tactical move.
                            reversal_bonus = 8000
                    # --- END OF PURSUIT REVERSAL BONUS ---
                    
                    new_occupied = occupied_positions.union({new_opponent_pos})
                    eval_score = self.minimax(board_width, 
                                              board_height, 
                                              new_occupied, 
                                              player_pos, 
                                              player_dir,
                                              new_opponent_pos, 
                                              direction, 
                                              depth - 1, 
                                              False,
                                              alpha, 
                                              beta)
                    
                    eval_score += reversal_bonus  # Add bonus to the evaluated score for this move
                    
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
                    eval_score = self.minimax(board_width, 
                                              board_height, 
                                              new_occupied, 
                                              new_player_pos, 
                                              direction,
                                              opponent_pos, 
                                              opponent_dir, 
                                              depth - 1, 
                                              True, 
                                              alpha, 
                                              beta)
                    
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

    def choose_direction(self, opponent, player, board_width, board_height, occupied_positions):
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
                if self.is_valid_move(board_width, board_height, occupied_positions, new_pos):
                    valid_directions.append(direction)
            
            if valid_directions:
                return random.choice(valid_directions)
        
        # Use minimax search for best move
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
            score = self.minimax(board_width, 
                                 board_height, 
                                 new_occupied,
                                 (player.x, player.y), 
                                 player.direction,
                                 new_opponent_pos, 
                                 direction, 
                                 min(self.search_depth, 4), 
                                 False)  # Start with player's turn
            
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
                    edge_removal = self.count_edge_removal(board_width, 
                                                           board_height, 
                                                           occupied_positions,
                                                           new_pos[0], 
                                                           new_pos[1])
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
        self.setWindowFlags(Qt.Window | 
                            Qt.CustomizeWindowHint |
                            Qt.WindowTitleHint | 
                            Qt.WindowCloseButtonHint)

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
        self.opponents = []  # List of AI opponents
        self.ais = []  # List of AI instances
        self.game_started = False
        self.title_timer = 0  # For title animation
        self.obstacles = set()  # Set of (x, y) positions for obstacles

    def start_game(self):
        """Start the game by initializing."""
        # Stop and disconnect any existing timer before starting a new game
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
        
        # Disconnect all previous connections to prevent multiple connections
        try:
            self.timer.timeout.disconnect()
        except TypeError:
            # No connections to disconnect
            pass
        
        self.difficulty = "Hard"
        
        # Always reset game speed to baseline for this difficulty
        speed_settings = {
            "Easy": 200,
            "Medium": 150,
            "Hard": 100,
            "Insane": 75
        }
        self.game_speed = speed_settings[self.difficulty]

        # Show welcome dialog
        welcome_msg = (
            f"Welcome to Light Cycle - {self.difficulty} Mode!\n\n"
            "Rules:\n"
            " - Use WASD keys to turn your cycle (gold)\n"
            " - Avoid walls, your trail, and the opponents' trails (red)\n"
            " - The opponents move automatically\n"
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
        
        # Player starts at bottom left, moving right
        self.player = LightCycle(5, self.board_height - 5, RIGHT, PLAYER_COLOR)

        # Create 5 opponents at different starting positions
        opponent_starts = [
            (self.board_width // 2, self.board_height // 2, DOWN),  # center
            (10, 10, DOWN),  # top-left
            (self.board_width - 10, 10, LEFT),  # top-right
            (10, self.board_height - 10, RIGHT),  # bottom-left
            (self.board_width - 10, self.board_height - 10, UP)  # bottom-right
        ]
        self.opponents = []
        for x, y, direction in opponent_starts:
            opponent = LightCycle(x, y, direction, OPPONENT_COLOR)
            self.opponents.append(opponent)

        # Create AI instances for each opponent
        self.ais = [LightCycleAI(self.difficulty) for _ in range(5)]

        # Generate 100 random obstacles
        self.obstacles = set()
        opponent_positions = [(x, y) for x, y, _ in opponent_starts]
        for _ in range(100):
            while True:
                x = random.randint(0, self.board_width - 1)
                y = random.randint(0, self.board_height - 1)
                pos = (x, y)
                player_start = (5, self.board_height - 5)
                if pos not in [player_start] + opponent_positions and pos not in self.obstacles:
                    self.obstacles.add(pos)
                    break

        self.game_started = True
        self.title_timer = 10  # Show title for 0.5 seconds (30 frames at ~60fps)
        
    def keyPressEvent(self, event):
        """Handle key presses for player control."""
        if not self.game_started or not self.player.alive:
            return

        key = event.key()
        
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
        """Handle key releases (no longer needed for speed control)."""
        pass

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

        for opponent in self.opponents:
            # Draw glow/shadow first
            painter.setPen(Qt.NoPen)
            glow_color = QColor(0, 255, 255, 100)  # Semi-transparent cyan glow
            painter.setBrush(QBrush(glow_color))
            for x, y in opponent.trail:
                painter.drawRect((x * self.cell_size) - 1, (y * self.cell_size) - 1,
                                 self.cell_size + 2, self.cell_size + 2)

            # Draw main trail
            painter.setBrush(QBrush(opponent.color))
            for x, y in opponent.trail:
                painter.drawRect(x * self.cell_size, y * self.cell_size,
                                 self.cell_size, self.cell_size)

        # Draw obstacles
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(128, 128, 128)))  # Gray
        for x, y in self.obstacles:
            painter.drawRect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)

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

            title_text = "LIGHT CYCLES GAME"
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
            # Move player once per frame at constant speed
            self.player.move()
            
            # Calculate current occupied positions
            all_trails = self.player.trail + [pos for opp in self.opponents for pos in opp.trail]
            occupied_positions = set(all_trails) | self.obstacles
            
            # AI decides each opponent's next move
            intended_moves = {}
            for i, opponent in enumerate(self.opponents):
                if opponent.alive:
                    best_direction = self.ais[i].choose_direction(
                        opponent, self.player, self.board_width, self.board_height, occupied_positions)
                    new_pos = self.ais[i].simulate_move((opponent.x, opponent.y), best_direction)
                    intended_moves[i] = (best_direction, new_pos)
            
            # Check for conflicts in intended positions
            new_positions = [pos for _, pos in intended_moves.values()]
            position_counts = {}
            for pos in new_positions:
                position_counts[pos] = position_counts.get(pos, 0) + 1
            
            # Resolve conflicts by re-choosing for conflicting opponents
            for i in intended_moves:
                direction, new_pos = intended_moves[i]
                if position_counts[new_pos] > 1:
                    # Conflict, choose a random valid direction avoiding other intended moves
                    opponent = self.opponents[i]
                    valid_directions = []
                    for d in [UP, DOWN, LEFT, RIGHT]:
                        if ((opponent.direction == UP and d == DOWN) or
                            (opponent.direction == DOWN and d == UP) or
                            (opponent.direction == LEFT and d == RIGHT) or
                            (opponent.direction == RIGHT and d == LEFT)):
                            continue
                        np = self.ais[i].simulate_move((opponent.x, opponent.y), d)
                        # Occupied includes current + other intended (excluding this one's current intended)
                        other_intended = set([p for j, p in intended_moves.values() if j != i])
                        occ = occupied_positions | other_intended
                        if self.ais[i].is_valid_move(self.board_width, self.board_height, occ, np):
                            valid_directions.append(d)
                    if valid_directions:
                        new_dir = random.choice(valid_directions)
                        new_pos = self.ais[i].simulate_move((opponent.x, opponent.y), new_dir)
                        intended_moves[i] = (new_dir, new_pos)
            
            # Apply the moves
            for i, opponent in enumerate(self.opponents):
                if opponent.alive:
                    direction, _ = intended_moves[i]
                    opponent.turn(direction)
                    opponent.move()

            # Check player collision
            all_opponent_trails = [pos for opp in self.opponents for pos in opp.trail]
            player_crashed = self.player.check_collision(self.board_width, 
                                                         self.board_height, 
                                                         all_opponent_trails, 
                                                         self.obstacles)
            
            # Check each opponent collision
            for opponent in self.opponents:
                if opponent.alive:
                    other_trails = (
                        self.player.trail +
                        [pos for opp in self.opponents 
                         if opp != opponent for pos in opp.trail]
                    )
                    opponent_crashed = opponent.check_collision(self.board_width, 
                                                                self.board_height,
                                                                other_trails, 
                                                                self.obstacles)

            # Check win conditions
            alive_opponents = sum(1 for opp in self.opponents if opp.alive)
            if player_crashed:
                self.game_over("You crashed! Opponents win.")
            elif alive_opponents == 0:
                self.game_over("All opponents crashed! You win!")

            self.update()

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
            # Clean up completely and close
            self.cleanup_and_close()

    def cleanup_and_close(self):
        """Completely clean up and close the game window."""
        # Stop the timer and disconnect all connections
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
        
        # Disconnect all timer connections to prevent memory leaks
        try:
            self.timer.timeout.disconnect()
        except TypeError:
            # No connections to disconnect
            pass
        
        # Clear references
        self.player = None
        self.opponents = []
        self.ais = []
        
        # Reset game state
        self.game_started = False
        
        # Close the window
        self.close()
        
        # Clear any reference from main window
        if hasattr(self, 'main_window') and self.main_window:
            # If the main window has a reference to this game, clear it
            if hasattr(self.main_window, 'current_game'):
                self.main_window.current_game = None

    def closeEvent(self, event):
        """Handle window close."""
        # Ensure proper cleanup when window is closed
        self.cleanup_and_close()
        event.accept()
