# Credit: JustSong, @songquanpeng

import warnings

warnings.filterwarnings("ignore")

import queue
import random

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QBrush, QColor, QFont
from PyQt5.QtWidgets import (QMainWindow, QMessageBox, QDialog, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QButtonGroup)

from coralnet_toolbox.Icons import get_icon


# ----------------------------------------------------------------------------------------------------------------------
# Constants / Configurations
# ----------------------------------------------------------------------------------------------------------------------


# Base game settings (will be modified by difficulty)
BASE_FOOD_NUM = 30
BASE_TIME_INTERVAL = 5
BASE_BOARD_ROW = 20
BASE_BOARD_COLUMN = 30
AUTO_PLAY = False

# Base initial delay speed (in milliseconds)
BASE_SPEED = 500
SPECIAL_FOOD = 4

# Direction constants
UP = 1
DOWN = -1
LEFT = 2
RIGHT = -2

# Board cell types (prohibit changing numbers)
BLANK = 0
HEAD = 1
BODY = 2
FOOD = 3


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Snake:
    """
    Class representing the snake in the game.
    """

    def __init__(self):
        """
        Initialize the snake with default values.
        """
        # y coordinate
        self.row = BOARD_ROW
        # x coordinate
        self.column = BOARD_COLUMN
        # Initialize the head position (x, y)
        self.head = [random.randint(3, self.column - 3), random.randint(3, self.row - 3)]
        # Set the tail position to be the same as the head initially
        self.tail = self.head
        # Initialize the board as a 2D list with BLANK cells
        self.board = [[BLANK for _ in range(self.row)] for _ in range(self.column)]
        # Set the head position on the board
        self.board[self.head[0]][self.head[1]] = HEAD
        # Initialize the direction of the snake
        self.direction = UP
        # Initialize food parameters (normal and special) and extra growth counter
        self.food = [-1, -1]
        self.food_created_time = 0
        self.special_food = [-1, -1]
        self.special_food_created_time = 0
        self.extra_growth = 0
        # The snake is alive and starts with length 1
        self.live = True
        self.length = 1
        # Initialize the snake queue for tracking the snake's position (max size is board size)
        self.snake_queue = queue.Queue(self.row * self.column)
        self.snake_queue.put(self.head)
        # Generate the initial food
        self.new_food(0)

    def new_food(self, current_time):
        """
        Create a new food position on the board.
        """
        # Clear previous food marker if exists
        if self.food != [-1, -1]:
            self.board[self.food[0]][self.food[1]] = BLANK
        # Place new food on the board at a random BLANK position
        while True:
            temp_food = [random.randint(0, self.column - 1), random.randint(0, self.row - 1)]
            if self.board[temp_food[0]][temp_food[1]] == BLANK:
                self.food = temp_food
                self.board[self.food[0]][self.food[1]] = FOOD
                self.food_created_time = current_time
                break

    def new_special_food(self, current_time):
        """
        Create a new special food position on the board.
        """
        if self.special_food != [-1, -1]:
            self.board[self.special_food[0]][self.special_food[1]] = BLANK
        while True:
            temp_food = [random.randint(0, self.column - 1), random.randint(0, self.row - 1)]
            if self.board[temp_food[0]][temp_food[1]] == BLANK:
                self.special_food = temp_food
                self.board[self.special_food[0]][self.special_food[1]] = SPECIAL_FOOD
                self.special_food_created_time = current_time
                break

    def check_collision(self, direction):
        """
        Check if moving in the given direction will result in a collision.

        Returns:
            collision (bool): True if a collision will occur.
            new_head (list): New head position [x, y] after moving.
        """
        collision = False
        new_head = self.head[:]
        if direction == UP:
            # Check if head is at the top border
            if self.head[1] == 0:
                collision = True
            # Move up by decreasing y coordinate
            new_head[1] -= 1
        elif direction == DOWN:
            # Check if head is at the bottom border
            if self.head[1] == self.row - 1:
                collision = True
            # Move down by increasing y coordinate
            new_head[1] += 1
        elif direction == LEFT:
            # Check if head is at the left border
            if self.head[0] == 0:
                collision = True
            # Move left by decreasing x coordinate
            new_head[0] -= 1
        elif direction == RIGHT:
            # Check if head is at the right border
            if self.head[0] == self.column - 1:
                collision = True
            # Move right by increasing x coordinate
            new_head[0] += 1
        return collision, new_head

    def go(self, direction):
        """
        Move the snake in the given direction.

        If encountering food, the snake grows.
        If colliding with the wall or its own body, the snake dies.
        """
        # Do not reverse direction directly
        if direction + self.direction == 0:
            pass
        else:
            self.direction = direction

        collision, new_head = self.check_collision(self.direction)
        if collision:
            self.live = False
        else:
            if new_head == self.food:
                # The snake has encountered food and should grow
                self.length += 1
                # Set new head position on the board
                self.board[new_head[0]][new_head[1]] = HEAD
                # Update previous head position to BODY
                self.board[self.head[0]][self.head[1]] = BODY
                self.head = new_head[:]
                # Add new head to the snake queue
                self.snake_queue.put(self.head)
                # Reset food to default before generating new one
                self.food = [-1, -1]
            elif new_head == self.special_food:
                self.length += 3
                self.board[new_head[0]][new_head[1]] = HEAD
                self.board[self.head[0]][self.head[1]] = BODY
                self.head = new_head[:]
                self.snake_queue.put(self.head)
                self.special_food = [-1, -1]
                self.extra_growth = 2  # Skip tail removal for 2 subsequent moves
            elif self.board[new_head[0]][new_head[1]] == BODY:
                # The snake has collided with its own body
                self.live = False
            elif self.board[new_head[0]][new_head[1]] == BLANK:
                # The snake is moving forward without growing
                # Update previous head position to BODY
                self.board[self.head[0]][self.head[1]] = BODY
                # Set the new head position
                self.head = new_head[:]
                # Mark the new head position on the board
                self.board[self.head[0]][self.head[1]] = HEAD
                # Add the new head to the queue
                self.snake_queue.put(new_head)
                if self.extra_growth > 0:
                    self.extra_growth -= 1
                else:
                    # Remove tail from the queue and update its board cell to BLANK
                    self.tail = self.snake_queue.get()
                    self.board[self.tail[0]][self.tail[1]] = BLANK


class SnakeGame(QMainWindow):
    """
    Class representing the main game window and logic.
    """

    def __init__(self, main_window, parent=None):
        """
        Initialize the game window, the snake, and the game timer.
        """
        super().__init__()
        self.main_window = main_window

        # Remove minimize and maximize buttons.
        self.setWindowFlags(Qt.Window |
                            Qt.CustomizeWindowHint |
                            Qt.WindowTitleHint |
                            Qt.WindowCloseButtonHint)

        # Set the window icon
        self.setWindowIcon(get_icon("snake.png"))
        # Set the window title
        self.title = "Coral Snake Game"

        # Set the size used for each board cell
        self.size = 30
        # Set the window opacity
        self.opacity = 1
        # Initialize the timer for game updates
        self.update_timer = QTimer(self)
        # Set up the status bar
        self.status_bar = self.statusBar()
        # Create brushes for drawing the board cells
        self.brush = [
            QBrush(QColor(144, 238, 144), Qt.SolidPattern),  # Light green (grass-like) for BLANK
            QBrush(Qt.black, Qt.SolidPattern),               # HEAD starts as black
            QBrush(Qt.green, Qt.SolidPattern),               # BODY (will be overridden dynamically)
            QBrush(Qt.yellow, Qt.SolidPattern),              # FOOD
            QBrush(QColor(128, 0, 128), Qt.SolidPattern)     # Purple for SPECIAL_FOOD
        ]
        # Initialize time counter
        self.time_count = 0
        # Store the current speed (delay in ms)
        self.speed = BASE_SPEED
        # Store difficulty level
        self.difficulty = "Medium"
        # For title animation
        self.title_timer = 0
        # Track timer connection state
        self.timer_connected = False

    def setup_difficulty_parameters(self):
        """Set game parameters based on difficulty level."""
        difficulty_settings = {
            "Easy": {
                "board_rows": 15,
                "board_columns": 20,
                "base_speed": 700,
                "special_food_duration": 8,
                "speed_increase_factor": 50
            },
            "Medium": {
                "board_rows": 20,
                "board_columns": 30,
                "base_speed": 500,
                "special_food_duration": 5,
                "speed_increase_factor": 70
            },
            "Hard": {
                "board_rows": 25,
                "board_columns": 35,
                "base_speed": 350,
                "special_food_duration": 3,
                "speed_increase_factor": 90
            },
            "Insane": {
                "board_rows": 30,
                "board_columns": 40,
                "base_speed": 200,
                "special_food_duration": 2,
                "speed_increase_factor": 120
            }
        }
        
        settings = difficulty_settings[self.difficulty]
        
        # Update global constants based on difficulty
        global BOARD_ROW, BOARD_COLUMN, BASE_SPEED, SPECIAL_FOOD
        BOARD_ROW = settings["board_rows"]
        BOARD_COLUMN = settings["board_columns"]
        BASE_SPEED = settings["base_speed"]
        self.speed = BASE_SPEED
        self.special_food_duration = settings["special_food_duration"]
        self.speed_increase_factor = settings["speed_increase_factor"]

    def start_game(self):
        """
        Start the game by initializing the game window and UI.
        """
        # Stop and disconnect any existing timer before starting a new game
        if self.update_timer.isActive():
            self.update_timer.stop()
        
        # Safely disconnect all previous connections to prevent multiple connections
        if self.timer_connected:
            try:
                self.update_timer.timeout.disconnect()
                self.timer_connected = False
            except TypeError:
                # No connections to disconnect
                pass
        
        self.difficulty = "Hard"
        self.setup_difficulty_parameters()
        
        # Show welcome dialog with instructions before starting the game.
        welcome_msg = (
            f"Welcome to Snake - {self.difficulty} Mode!\n\n"
            "Rules:\n"
            " - Use 'W' to move up\n"
            " - Use 'A' to move left\n"
            " - Use 'S' to move down\n"
            " - Use 'D' to move right\n"
            " - Avoid colliding with walls or your tail.\n"
            " - Eat yellow food to grow!\n"
            " - Eat purple special food for extra growth!\n\n"
            "Click 'OK' to start playing."
        )
        if QMessageBox.information(self, "Welcome to Coral Snake", welcome_msg, QMessageBox.Ok) == QMessageBox.Ok:
            self.init_game()                     # Create snake and set board dimensions.
            self.init_ui()                       # Use self.row and self.column for UI sizing.
            self.update_timer.start(self.speed)  # Start the timer.
        else:
            self.close()
            
    def init_ui(self):
        """
        Set up the user interface dimensions and appearance.
        """
        # Calculate game board dimensions
        board_width = self.column * self.size
        board_height = self.row * self.size
        
        # Add padding around the board for centering
        padding = 40  # 20 pixels on each side
        window_width = board_width + padding
        window_height = board_height + padding + 40  # Extra space for status bar
        
        # Store offset for centering the board
        self.board_offset_x = padding // 2
        self.board_offset_y = padding // 2
        
        self.resize(window_width, window_height)
        self.setFixedSize(window_width, window_height)  # Prevent resizing/minimization.
        self.setWindowTitle(self.title)
        self.setWindowOpacity(self.opacity)
        self.show()

    def init_game(self):
        """
        Create the snake instance and update board dimensions.
        """
        self.snake = Snake()
        self.row = self.snake.row
        self.column = self.snake.column
        self.title_timer = 10  # Show title for 0.5 seconds
        self.update_timer.timeout.connect(self.update_game)
        self.timer_connected = True
        
    def keyPressEvent(self, event):
        """
        Process keyboard events for controlling the snake.
        """
        if self.snake is not None and self.snake.live:
            if event.key() == Qt.Key_W:
                self.snake.go(UP)
            elif event.key() == Qt.Key_S:
                self.snake.go(DOWN)
            elif event.key() == Qt.Key_A:
                self.snake.go(LEFT)
            elif event.key() == Qt.Key_D:
                self.snake.go(RIGHT)
            elif event.key() == Qt.Key_P:
                pass
            # Redraw the game board after processing the key press
            self.update()

    def paintEvent(self, event):
        """
        Render the game board and the snake.
        """
        painter = QPainter()
        painter.begin(self)
        
        # Check if snake exists before trying to render the board
        if self.snake is not None:
            # Draw each cell based on the board data with centering offset
            for x in range(self.column):
                for y in range(self.row):
                    cell_type = self.snake.board[x][y]
                    if cell_type == BODY:
                        # Dynamic coloring for snake body based on length
                        brush = self.get_snake_body_brush(x, y)
                    else:
                        brush = self.brush[cell_type]
                    painter.setBrush(brush)
                    # Draw the corresponding rectangle for the cell with offset for centering
                    painter.drawRect(
                        x * self.size + self.board_offset_x, 
                        y * self.size + self.board_offset_y, 
                        self.size, 
                        self.size
                    )
        painter.end()

        # Draw coral snake title animation
        if self.title_timer > 0:
            painter.begin(self)
            title_font = QFont('Arial', 36, QFont.Bold)
            painter.setFont(title_font)

            # Create animated colors matching coral snake theme
            colors = [QColor(255, 0, 0), QColor(255, 255, 0), QColor(0, 0, 0), QColor(255, 0, 0)]
            color_index = (self.title_timer // 10) % len(colors)
            painter.setPen(colors[color_index])

            title_text = "CORAL SNAKE GAME"
            text_rect = painter.fontMetrics().boundingRect(title_text)
            x = (self.width() - text_rect.width()) // 2
            y = self.height() // 2

            # Add glow effect to title
            glow_colors = [QColor(255, 0, 0, 150), QColor(255, 255, 0, 150)]
            glow_index = (self.title_timer // 5) % len(glow_colors)
            painter.setPen(glow_colors[glow_index])
            painter.drawText(x - 2, y - 2, title_text)
            painter.drawText(x + 2, y - 2, title_text)
            painter.drawText(x - 2, y + 2, title_text)
            painter.drawText(x + 2, y + 2, title_text)

            painter.setPen(colors[color_index])
            painter.drawText(x, y, title_text)
            painter.end()

    def get_snake_body_brush(self, x, y):
        """
        Get the appropriate brush for a snake body segment based on its position and snake length.
        """
        # Find the position of this segment in the snake
        position = 0
        snake_positions = list(self.snake.snake_queue.queue)
        for i, pos in enumerate(snake_positions):
            if pos[0] == x and pos[1] == y:
                position = i
                break
        
        # Coral snake pattern: red touches yellow, with black bands
        # Pattern: black (head), red, black, yellow, black, red, black, yellow...
        # Head (position 0) is always black
        if position == 0:
            return QBrush(Qt.black, Qt.SolidPattern)
        else:
            pattern_index = (position - 1) % 4
            if pattern_index == 0:
                return QBrush(Qt.red, Qt.SolidPattern)     # Red
            elif pattern_index == 1:
                return QBrush(Qt.black, Qt.SolidPattern)   # Black
            elif pattern_index == 2:
                return QBrush(Qt.yellow, Qt.SolidPattern)  # Yellow
            else:
                return QBrush(Qt.black, Qt.SolidPattern)   # Black

    def update_game(self):
        """
        Handle game updates on each timer tick.
        """
        # Check if snake is None (game ended) - prevent NoneType errors
        if self.snake is None:
            return
            
        if self.title_timer > 0:
            self.title_timer -= 1
            self.update()
            return

        # Adjust the speed based on the snake's growth (each food eaten speeds up the snake)
        # Use difficulty-based speed increase factor
        new_speed = max(BASE_SPEED - (self.snake.length - 1) * self.speed_increase_factor, 50)
        if new_speed != self.speed:
            self.speed = new_speed
            self.update_timer.setInterval(self.speed)

        # Increase the time counter
        self.time_count += 1
        # If the snake is alive, update its position using auto_play logic
        if self.snake.live:
            self.snake.go(self.auto_play())
            # Check for win condition: snake fills the board.
            if self.snake.length == (self.column * self.row):
                self.win_game()
                return
            # Refresh yellow food only if eaten (do not auto-refresh after 10 seconds)
            if self.snake.food == [-1, -1]:
                self.snake.new_food(self.time_count)
            # Spawn special food every 15 seconds if not present
            if self.time_count % 15 == 0 and self.snake.special_food == [-1, -1]:
                self.snake.new_special_food(self.time_count)
            # Remove special food based on difficulty duration
            if self.snake.special_food != [-1, -1]: 
                if (self.time_count - self.snake.special_food_created_time) >= self.special_food_duration:
                    self.snake.board[self.snake.special_food[0]][self.snake.special_food[1]] = BLANK
                    self.snake.special_food = [-1, -1]
        else:
            # End the game if the snake is dead
            self.game_over()
        # Redraw the game board after handling the timer event
        self.update()

    def win_game(self):
        """
        Handle winning the game and offer to play again.
        """
        # Stop the timer and congratulate the user.
        self.update_timer.stop()
        win_message = (f"Congratulations! You've filled the entire board and won on {self.difficulty} mode!\n\n"
                       "Do you want to play again?")
        reply = QMessageBox.question(self, "You Win!", win_message, QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.start_game()
        else:
            self.cleanup_and_close()

    def auto_play(self):
        """
        Determine the snake's next move automatically if AUTO_PLAY is enabled.

        Returns:
            final_direction (int): The determined direction for the snake.
        """
        if not AUTO_PLAY:
            return self.snake.direction
        else:
            # List of possible directions
            direction_list = [UP, DOWN, RIGHT, LEFT]
            final_direction = self.snake.direction
            for d in direction_list:
                # Skip if the new direction is directly opposite to the current direction
                if d + self.snake.direction != 0:
                    collision, new_head = self.snake.check_collision(d)
                    # Skip if collision will occur or the new position is part of the snake's body
                    if collision or self.snake.board[new_head[0]][new_head[1]] == BODY:
                        continue
                    else:
                        # Prioritize food if encountered
                        if new_head == self.snake.food:
                            final_direction = d
                            return final_direction
                        elif self.snake.board[new_head[0]][new_head[1]] == BLANK:
                            final_direction = d
                            continue
            return final_direction

    def game_over(self):
        """
        Display a message box when the game is over and handle restarting or closing.
        """
        # Stop the timer immediately when game is over
        self.update_timer.stop()
        
        # Show game-over dialog offering the choice to play again or get back to work.
        message = (f"Difficulty: {self.difficulty}\nTime: {self.time_count}\n"
                   f"Length: {self.snake.length}\n\nDo you want to play again?")
        reply = QMessageBox.question(
            self,
            "Game Over",
            message,
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.start_game()
        else:
            self.cleanup_and_close()
            
    def end_game(self):
        """
        End the game by stopping the timer and closing the window.
        """
        self.cleanup_and_close()

    def cleanup_and_close(self):
        """Completely clean up and close the game window."""
        # Stop the timer and disconnect all connections
        if self.update_timer.isActive():
            self.update_timer.stop()
        
        # Disconnect all timer connections to prevent memory leaks
        if self.timer_connected:
            try:
                self.update_timer.timeout.disconnect()
                self.timer_connected = False
            except TypeError:
                # No connections to disconnect
                pass
        
        # Clear references
        self.snake = None
        
        # Close the window
        self.close()
        
        # Clear any reference from main window
        if hasattr(self, 'main_window') and self.main_window:
            # If the main window has a reference to this game, clear it
            if hasattr(self.main_window, 'current_game'):
                self.main_window.current_game = None

    def closeEvent(self, event):
        """
        Handle the window close event.
        """
        # Ensure proper cleanup when window is closed
        self.cleanup_and_close()
        event.accept()
