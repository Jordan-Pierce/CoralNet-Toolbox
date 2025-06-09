# Credit: JustSong, @songquanpeng

import warnings

warnings.filterwarnings("ignore")

import queue
import random

from PyQt5.QtCore import Qt, QBasicTimer
from PyQt5.QtGui import QPainter, QBrush
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


class DifficultyDialog(QDialog):
    """
    Dialog for selecting game difficulty level.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_difficulty = "Easy"  # Default
        self.init_ui()
        
    def init_ui(self):
        """Initialize the difficulty selection UI."""
        self.setWindowTitle("Select Difficulty")
        self.setModal(True)
        self.setFixedSize(400, 300)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Choose Your Difficulty Level")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(title.font())
        title.font().setPointSize(16)
        title.font().setBold(True)
        layout.addWidget(title)
        
        # Difficulty descriptions
        difficulties = {
            "Easy": "🟢 Slower snake, smaller board, longer special food time\nPerfect for beginners!",
            "Medium": "🟡 Balanced gameplay, standard board size\nThe classic experience",
            "Hard": "🟠 Faster snake, larger board, shorter special food time\nFor experienced players",
            "Insane": "🔴 Lightning fast, massive board, very short special food\nOnly for the brave!"
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
        self.title = "Snake Game"

        # Set the size used for each board cell
        self.size = 30
        # Set the window opacity
        self.opacity = 1
        # Initialize the timer for game updates
        self.update_timer = QBasicTimer()
        # Set up the status bar
        self.status_bar = self.statusBar()
        # Create brushes for drawing the board cells
        self.brush = [
            QBrush(Qt.white, Qt.SolidPattern),
            QBrush(Qt.darkBlue, Qt.SolidPattern),
            QBrush(Qt.green, Qt.SolidPattern),
            QBrush(Qt.yellow, Qt.SolidPattern),
            QBrush(Qt.red, Qt.SolidPattern)
        ]
        # Initialize time counter
        self.time_count = 0
        # Store the current speed (delay in ms)
        self.speed = BASE_SPEED
        # Store difficulty level
        self.difficulty = "Medium"

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
        Start the game by showing difficulty selection and initializing the game window and UI.
        """
        # Show difficulty selection dialog
        difficulty_dialog = DifficultyDialog(self)
        if difficulty_dialog.exec_() == DifficultyDialog.Accepted:
            self.difficulty = difficulty_dialog.get_difficulty()
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
                " - Eat red special food for extra growth!\n\n"
                "Click 'OK' to start playing."
            )
            if QMessageBox.information(self, "Welcome to Snake", welcome_msg, QMessageBox.Ok) == QMessageBox.Ok:
                self.init_game()                   # Create snake and set board dimensions.
                self.init_ui()                     # Use self.row and self.column for UI sizing.
                self.update_timer.start(self.speed, self)  # Start the timer.
        else:
            self.close()

    def end_game(self):
        """
        End the game by stopping the timer and closing the window.
        """
        self.snake = None
        self.update_timer.stop()
        self.close()

    def closeEvent(self, event):
        """
        Handle the window close event.
        """
        self.end_game()
        event.accept()

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

    def paintEvent(self, event):
        """
        Render the game board and the snake.
        """
        painter = QPainter()
        painter.begin(self)
        # Draw each cell based on the board data with centering offset
        for x in range(self.column):
            for y in range(self.row):
                # Set the brush based on the current block state
                painter.setBrush(self.brush[self.snake.board[x][y]])
                # Draw the corresponding rectangle for the cell with offset for centering
                painter.drawRect(
                    x * self.size + self.board_offset_x, 
                    y * self.size + self.board_offset_y, 
                    self.size, 
                    self.size
                )
        painter.end()

    def timerEvent(self, event):
        """
        Handle game updates on each timer tick.
        """
        # Adjust the speed based on the snake's growth (each food eaten speeds up the snake)
        # Use difficulty-based speed increase factor
        new_speed = max(BASE_SPEED - (self.snake.length - 1) * self.speed_increase_factor, 50)
        if new_speed != self.speed:
            self.speed = new_speed
            # Restart the timer with new speed
            self.update_timer.stop()
            self.update_timer.start(self.speed, self)

        # Check if the event corresponds to our update timer
        if event.timerId() == self.update_timer.timerId():
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
        # Stop the timer and congratulate the user.
        self.update_timer.stop()
        win_message = (f"Congratulations! You've filled the entire board and won on {self.difficulty} mode!\n\n"
                       "Do you want to play again?")
        reply = QMessageBox.question(self, "You Win!", win_message, QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.start_game()
        else:
            self.end_game()

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
        # Show game-over dialog offering the choice to play again or get back to work.
        message = f"Difficulty: {self.difficulty}\nTime: {self.time_count}\nLength: {self.snake.length}\n\nDo you want to play again?"
        reply = QMessageBox.question(
            self,
            "Game Over",
            message,
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.start_game()
        else:
            self.end_game()

    def keyPressEvent(self, event):
        """
        Process keyboard events for controlling the snake.
        """
        if self.snake.live:
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
