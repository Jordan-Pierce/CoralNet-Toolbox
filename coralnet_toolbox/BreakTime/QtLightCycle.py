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
        self.game_started = False
        self.title_timer = 0  # For title animation
        self.pressed_keys = set()  # Track pressed movement keys for speed boost
        self.player_speed = 1  # Speed multiplier for player

    def start_game(self):
        """Start the game by initializing."""
        # Hardcode difficulty to Hard
        self.difficulty = "Hard"
        
        # Set game speed based on difficulty
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
        # Player starts at bottom left, moving right
        self.player = LightCycle(5, self.board_height - 5, RIGHT, PLAYER_COLOR)

        # Opponent starts at top right, moving left
        self.opponent = LightCycle(self.board_width - 5, 5, LEFT, OPPONENT_COLOR)

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
        if self.title_timer > 0:
            self.title_timer -= 1
            self.update()
            return

        if self.game_started:
            # Move player multiple times based on speed
            for _ in range(self.player_speed):
                self.player.move()
            
            # Move opponent
            self.opponent.move()

            # Simple AI for opponent: random turns occasionally
            if random.random() < 0.1:  # 10% chance to turn
                directions = [UP, DOWN, LEFT, RIGHT]
                new_dir = random.choice(directions)
                self.opponent.turn(new_dir)

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
        self.timer.stop()
        reply = QMessageBox.question(self, "Game Over", f"{message}\n\nPlay again?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.start_game()
        else:
            self.close()

    def closeEvent(self, event):
        """Handle window close."""
        self.timer.stop()
        event.accept()
