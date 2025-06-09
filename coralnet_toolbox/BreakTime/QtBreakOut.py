import warnings

warnings.filterwarnings("ignore")

import sys

from PyQt5.QtGui import QPainter, QColor, QFont, QPen
from PyQt5.QtCore import Qt, QBasicTimer, pyqtSignal, QRect
from PyQt5.QtWidgets import (QMainWindow, QApplication, QDesktopWidget, QWidget, 
                             QMessageBox, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QButtonGroup)

from coralnet_toolbox.Icons import get_icon


# ----------------------------------------------------------------------------------------------------------------------
# Constants / Configurations
# ----------------------------------------------------------------------------------------------------------------------

# Define base game constants
BASE_WIDTH = 600
BASE_HEIGHT = 600
BASE_PADDLE_WIDTH = 75
PADDLE_HEIGHT = 15
PADDLE_SPEED = 15
BASE_BALL_DIAMETER = 15
BASE_GAME_SPEED = 10  # Lower is faster


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
        title = QLabel("Choose Your Difficulty Level")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(title.font())
        title.font().setPointSize(16)
        title.font().setBold(True)
        layout.addWidget(title)
        
        # Difficulty descriptions
        difficulties = {
            "Easy": "🟢 Slow ball, large paddle, fewer bricks\nPerfect for beginners!",
            "Medium": "🟡 Balanced gameplay\nThe classic experience",
            "Hard": "🟠 Faster ball, smaller paddle, more bricks\nFor experienced players",
            "Insane": "🔴 Lightning fast, tiny paddle, maximum bricks\nOnly for the brave!"
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


class Board(QWidget):
    """
    The main game board where all the action happens.
    It handles drawing, game logic, and user input.
    """
    # Define base game constants
    WIDTH = BASE_WIDTH
    HEIGHT = BASE_HEIGHT
    PADDLE_WIDTH = BASE_PADDLE_WIDTH
    PADDLE_HEIGHT = PADDLE_HEIGHT
    PADDLE_SPEED = PADDLE_SPEED
    BALL_DIAMETER = BASE_BALL_DIAMETER
    GAME_SPEED = BASE_GAME_SPEED

    def __init__(self, parent, difficulty):
        super().__init__(parent)
        self.parent_window = parent
        self.difficulty = difficulty
        self.setup_difficulty_parameters()
        self.initGame()

    def setup_difficulty_parameters(self):
        """Set game parameters based on difficulty level."""
        difficulty_settings = {
            "Easy": {
                "paddle_width_multiplier": 1.5,
                "ball_speed_multiplier": 0.7,
                "game_speed": 16,  # Slower
                "brick_rows": 4,
                "ball_diameter_multiplier": 1.2
            },
            "Medium": {
                "paddle_width_multiplier": 1.0,
                "ball_speed_multiplier": 1.0,
                "game_speed": 12,  # Normal
                "brick_rows": 6,
                "ball_diameter_multiplier": 1.0
            },
            "Hard": {
                "paddle_width_multiplier": 0.7,
                "ball_speed_multiplier": 1.4,
                "game_speed": 8,  # Faster
                "brick_rows": 8,
                "ball_diameter_multiplier": 0.8
            },
            "Insane": {
                "paddle_width_multiplier": 0.5,
                "ball_speed_multiplier": 2.0,
                "game_speed": 5,  # Very fast
                "brick_rows": 10,
                "ball_diameter_multiplier": 0.6
            }
        }
        
        settings = difficulty_settings[self.difficulty]
        
        # Apply difficulty settings
        self.PADDLE_WIDTH = int(BASE_PADDLE_WIDTH * settings["paddle_width_multiplier"])
        self.BALL_DIAMETER = int(BASE_BALL_DIAMETER * settings["ball_diameter_multiplier"])
        self.GAME_SPEED = settings["game_speed"]
        self.BRICK_ROWS = settings["brick_rows"]
        self.ball_speed_multiplier = settings["ball_speed_multiplier"]
        
        # Adjust paddle speed based on difficulty (smaller paddles move faster for compensation)
        self.PADDLE_SPEED = int(PADDLE_SPEED * (2 - settings["paddle_width_multiplier"]))

    def initGame(self):
        """Initializes all game variables and objects."""
        # Game state flags
        self.isStarted = False
        self.isPaused = False

        # Paddle setup
        self.paddle = QRect((self.WIDTH - self.PADDLE_WIDTH) // 2, 
                            self.HEIGHT - 50, 
                            self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

        # Ball setup
        self.resetBall()

        # Bricks setup
        self.initBricks()

        # Timer for game loop
        self.timer = QBasicTimer()
        
        # Enable keyboard focus
        self.setFocusPolicy(Qt.StrongFocus)

    def initBricks(self):
        """Creates the grid of bricks based on difficulty."""
        self.bricks = []
        BRICK_WIDTH = 50
        BRICK_HEIGHT = 20
        BRICK_COLS = 10
        PADDING = 5
        TOP_OFFSET = 50

        colors = [QColor(0, 255, 0), QColor(0, 0, 255), QColor(255, 255, 0),
                  QColor(255, 165, 0), QColor(255, 0, 0), QColor(128, 0, 128),
                  QColor(255, 192, 203), QColor(0, 255, 255), QColor(255, 20, 147), QColor(50, 205, 50)]

        for row in range(self.BRICK_ROWS):
            for col in range(BRICK_COLS):
                x = col * (BRICK_WIDTH + PADDING) + PADDING + 20
                y = row * (BRICK_HEIGHT + PADDING) + TOP_OFFSET
                color = colors[row % len(colors)]
                self.bricks.append({'rect': QRect(x, y, BRICK_WIDTH, BRICK_HEIGHT), 'color': color})

    def resetBall(self):
        """Resets the ball to its starting position and state."""
        self.ball = QRect(self.paddle.center().x() - self.BALL_DIAMETER // 2,
                          self.paddle.top() - self.BALL_DIAMETER,
                          self.BALL_DIAMETER, self.BALL_DIAMETER)
        # Set initial direction with difficulty-based speed
        base_speed = 1
        self.xDir = int(base_speed * self.ball_speed_multiplier)
        self.yDir = int(-base_speed * self.ball_speed_multiplier)

    def paintEvent(self, event):
        """Handles all the drawing."""
        painter = QPainter(self)
        
        # Draw white background
        painter.fillRect(self.rect(), QColor(255, 255, 255))
        self.drawObjects(painter)

    def drawObjects(self, painter):
        """Draws all game objects."""
        # Draw Paddle
        painter.setBrush(QColor(0, 180, 255))
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.paddle)

        # Draw Ball
        painter.setBrush(QColor(50, 50, 50))
        painter.drawEllipse(self.ball)

        # Draw Bricks
        for brick in self.bricks:
            painter.setBrush(brick['color'])
            painter.setPen(QPen(QColor(30, 30, 30), 2))
            painter.drawRect(brick['rect'])
            
        # Draw difficulty indicator
        painter.setPen(QColor(50, 50, 50))
        painter.setFont(QFont('Arial', 12))
        painter.drawText(10, 25, f"Difficulty: {self.difficulty}")
        
        # Draw start/pause message
        if not self.isStarted:
            self.drawInfoText(painter, "Press SPACE to start")
        elif self.isPaused:
            self.drawInfoText(painter, "PAUSED - Press P to continue")

    def drawInfoText(self, painter, text):
        """Draws informational text on the screen."""
        painter.setPen(QColor(50, 50, 50))
        painter.setFont(QFont('Arial', 15))
        text_rect = painter.fontMetrics().boundingRect(text)
        x = (self.width() - text_rect.width()) // 2
        y = self.height() - 200
        painter.drawText(x, y, text)

    def keyPressEvent(self, event):
        """Handles key presses."""
        key = event.key()

        if key == Qt.Key_A:
            if self.paddle.left() > 0:
                self.paddle.translate(-self.PADDLE_SPEED, 0)
        elif key == Qt.Key_D:
            if self.paddle.right() < self.WIDTH:
                self.paddle.translate(self.PADDLE_SPEED, 0)
        elif key == Qt.Key_Space:
            if not self.isStarted and not self.isPaused:
                self.startGame()
        elif key == Qt.Key_P:
            self.pauseGame()
        
        self.update()  # Redraw after key press

    def startGame(self):
        """Starts the game loop."""
        if not self.isStarted:
            self.resetBall()
            self.isStarted = True
            self.timer.start(self.GAME_SPEED, self)

    def pauseGame(self):
        """Pauses or unpauses the game."""
        if self.isStarted:
            self.isPaused = not self.isPaused
            if self.isPaused:
                self.timer.stop()
            else:
                self.timer.start(self.GAME_SPEED, self)

    def timerEvent(self, event):
        """The main game loop, called by the timer."""
        if event.timerId() == self.timer.timerId():
            self.moveBall()
            self.checkCollision()
            self.update()

    def moveBall(self):
        """Moves the ball according to its direction vectors."""
        if self.isStarted and not self.isPaused:
            self.ball.translate(self.xDir, self.yDir)

    def checkCollision(self):
        """Checks for and handles collisions."""
        # Ball hits bottom wall -> Game Over
        if self.ball.bottom() > self.HEIGHT:
            self.timer.stop()
            self.parent_window.game_over(won=False)
            return

        # Ball hits top, left, or right walls
        if self.ball.left() <= 0 or self.ball.right() >= self.WIDTH:
            self.xDir = -self.xDir
        if self.ball.top() <= 0:
            self.yDir = -self.yDir

        # Ball hits paddle
        if self.ball.intersects(self.paddle) and self.yDir > 0:
            # Move ball above paddle to prevent sticking
            self.ball.moveBottom(self.paddle.top() - 1)
            
            # Complex paddle collision for better control
            paddle_center = self.paddle.center().x()
            ball_center = self.ball.center().x()
            
            # Change horizontal direction based on where it hits the paddle
            hit_factor = (ball_center - paddle_center) / (self.PADDLE_WIDTH / 2)
            max_speed = int(2 * self.ball_speed_multiplier)
            self.xDir = max(-max_speed, min(max_speed, int(hit_factor * max_speed)))
            if self.xDir == 0:
                self.xDir = int(self.ball_speed_multiplier) if hit_factor >= 0 else int(-self.ball_speed_multiplier)

            self.yDir = int(-1 * self.ball_speed_multiplier)  # Always bounce up

        # Ball hits a brick
        brick_to_remove = -1
        for i, brick in enumerate(self.bricks):
            if self.ball.intersects(brick['rect']):
                # Get the overlap amount to determine bounce direction more accurately
                ball_rect = self.ball
                brick_rect = brick['rect']
                
                # Calculate overlap on each side
                overlap_left = ball_rect.right() - brick_rect.left()
                overlap_right = brick_rect.right() - ball_rect.left()
                overlap_top = ball_rect.bottom() - brick_rect.top()
                overlap_bottom = brick_rect.bottom() - ball_rect.top()
                
                # Find the minimum overlap to determine collision side
                min_overlap = min(overlap_left, overlap_right, overlap_top, overlap_bottom)
                
                # Bounce based on the side with minimum overlap
                if min_overlap == overlap_left and self.xDir > 0:
                    self.xDir = -self.xDir
                    self.ball.moveRight(brick_rect.left() - self.ball.width() - 1)
                elif min_overlap == overlap_right and self.xDir < 0:
                    self.xDir = -self.xDir
                    self.ball.moveLeft(brick_rect.right() + 1)
                elif min_overlap == overlap_top and self.yDir > 0:
                    self.yDir = -self.yDir
                    self.ball.moveBottom(brick_rect.top() - 1)
                elif min_overlap == overlap_bottom and self.yDir < 0:
                    self.yDir = -self.yDir
                    self.ball.moveTop(brick_rect.bottom() + 1)
                    
                brick_to_remove = i
                break

        # Remove the brick after bouncing
        if brick_to_remove != -1:
            del self.bricks[brick_to_remove]

        # Check for win condition
        if not self.bricks:
            self.timer.stop()
            self.parent_window.game_over(won=True)


class BreakoutGame(QMainWindow):
    """
    Main application window for the Breakout game.
    """
    def __init__(self, main_window, parent=None):
        super().__init__()
        self.main_window = main_window
        
        # Remove minimize and maximize buttons
        self.setWindowFlags(Qt.Window |
                            Qt.CustomizeWindowHint |
                            Qt.WindowTitleHint |
                            Qt.WindowCloseButtonHint)
        
        # Set the window icon and title
        self.setWindowIcon(get_icon("breakout.png"))
        self.title = "Breakout Game"
        
        # Set the window opacity
        self.opacity = 1
        
        # Initialize the game board
        self.board = None
        self.difficulty = "Medium"  # Default difficulty

    def start_game(self):
        """Start the game by showing difficulty selection and initializing the game window and UI."""
        # Show difficulty selection dialog
        difficulty_dialog = DifficultyDialog(self)
        if difficulty_dialog.exec_() == QDialog.Accepted:
            self.difficulty = difficulty_dialog.get_difficulty()
            
            # Show welcome dialog with instructions
            welcome_msg = (
                f"Welcome to Breakout - {self.difficulty} Mode!\n\n"
                "Rules:\n"
                " - Use 'A' to move paddle left\n"
                " - Use 'D' to move paddle right\n"
                " - Press SPACE to start the game\n"
                " - Press P to pause/unpause\n"
                " - Break all the bricks to win!\n"
                " - Don't let the ball fall off the bottom!\n\n"
                "Click 'OK' to start playing."
            )
            if QMessageBox.information(self, "Welcome to Breakout", welcome_msg, QMessageBox.Ok) == QMessageBox.Ok:
                self.init_game()                   # Create board and set dimensions.
                self.init_ui()                     # Set up UI dimensions.
        else:
            self.close()

    def end_game(self):
        """End the game by stopping the timer and closing the window."""
        if self.board:
            self.board.timer.stop()
        self.board = None
        self.close()

    def closeEvent(self, event):
        """Handle the window close event."""
        self.end_game()
        event.accept()

    def init_ui(self):
        """Set up the user interface dimensions and appearance."""
        self.setFixedSize(self.board.WIDTH, self.board.HEIGHT)
        self.setWindowTitle(self.title)
        self.setWindowOpacity(self.opacity)
        self.show()

    def init_game(self):
        """Initialize the game board with selected difficulty."""
        self.board = Board(self, self.difficulty)
        self.setCentralWidget(self.board)

    def game_over(self, won=False):
        """Display game over message and handle restart/close."""
        if won:
            title = "Congratulations!"
            message = f"You destroyed all the bricks and won on {self.difficulty} mode!\n\nDo you want to play again?"
        else:
            title = "Game Over"
            message = f"The ball fell off the screen on {self.difficulty} mode!\n\nDo you want to play again?"
            
        reply = QMessageBox.question(
            self,
            title,
            message,
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.start_game()
        else:
            self.end_game()
