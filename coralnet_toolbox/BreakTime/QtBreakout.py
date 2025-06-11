import warnings

warnings.filterwarnings("ignore")

import sys
import random
import math
import json
import os

from PyQt5.QtGui import QPainter, QColor, QFont, QPen, QBrush
from PyQt5.QtCore import Qt, QBasicTimer, pyqtSignal, QRect, QTimer, QPoint
from PyQt5.QtWidgets import (QMainWindow, QApplication, QDesktopWidget, QWidget, 
                             QMessageBox, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QButtonGroup)

from coralnet_toolbox.Icons import get_icon


# ----------------------------------------------------------------------------------------------------------------------
# Constants / Configurations
# ----------------------------------------------------------------------------------------------------------------------

# Define base game constants
BASE_WIDTH = 700
BASE_HEIGHT = 700
BASE_PADDLE_WIDTH = 75
PADDLE_HEIGHT = 15
PADDLE_SPEED = 15
BASE_BALL_DIAMETER = 15
BASE_GAME_SPEED = 5  # Lower is faster


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------

class BallTrail:
    """Represents a trailing effect behind the ball."""
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size
        self.life = 30  # Increased from 15 to 30 for a longer tail
        self.max_life = 30
        
    def update(self):
        """Update trail position and properties."""
        self.life -= 1
        self.size = max(1, self.size - 0.15)  # Slower shrink for longer visible trail
        return self.life > 0
        
    def draw(self, painter):
        """Draw the trail segment."""
        if self.life > 0:
            # Fade out over time
            alpha = int(200 * (self.life / self.max_life))
            color = QColor(100, 100, 100)  # Gray trail
            color.setAlpha(alpha)
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(int(self.x), int(self.y), int(self.size), int(self.size))


class Particle:
    """Represents a visual particle for brick destruction effects."""
    def __init__(self, x, y, color, velocity_x=None, velocity_y=None):
        self.x = x
        self.y = y
        self.color = color
        self.velocity_x = velocity_x if velocity_x is not None else random.uniform(-3, 3)
        self.velocity_y = velocity_y if velocity_y is not None else random.uniform(-5, -1)
        self.life = 30  # Frames to live
        self.size = random.randint(2, 5)
        self.gravity = 0.2
        
    def update(self):
        """Update particle position and properties."""
        self.x += self.velocity_x
        self.y += self.velocity_y
        self.velocity_y += self.gravity
        self.life -= 1
        self.size = max(1, self.size - 0.1)
        return self.life > 0
        
    def draw(self, painter):
        """Draw the particle."""
        if self.life > 0:
            # Fade out over time
            alpha = int(255 * (self.life / 30))
            color = QColor(self.color)
            color.setAlpha(alpha)
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(int(self.x), int(self.y), int(self.size), int(self.size))


class PowerUp:
    """Represents a power-up that falls from destroyed bricks."""
    def __init__(self, x, y, power_type):
        self.rect = QRect(x, y, 20, 20)
        self.power_type = power_type
        self.velocity_y = 2
        self.colors = {
            'multiball': QColor(255, 0, 255),       # Magenta
            'big_paddle': QColor(0, 255, 0),        # Green
            'small_paddle': QColor(255, 165, 0),    # Orange
            'laser': QColor(255, 0, 0),             # Red
            'slow_ball': QColor(0, 0, 255),         # Blue
            'fast_ball': QColor(255, 255, 255),     # White
            'reverse_paddle': QColor(128, 0, 128),  # Purple
            'extra_life': QColor(255, 20, 147)      # Deep Pink
        }
        
    def update(self):
        """Update power-up position."""
        self.rect.translate(0, self.velocity_y)
        
    def draw(self, painter):
        """Draw the power-up."""
        painter.setBrush(QBrush(self.colors[self.power_type]))
        painter.setPen(QPen(QColor(0, 0, 0), 2))
        painter.drawEllipse(self.rect)
        
        # Draw power-up symbol
        painter.setPen(QPen(QColor(0, 0, 0), 2))
        painter.setFont(QFont('Arial', 8, QFont.Bold))
        symbols = {
            'multiball': 'M',
            'big_paddle': '+',
            'small_paddle': '-',
            'laser': 'L',
            'slow_ball': 'SL',
            'fast_ball': 'F',
            'reverse_paddle': 'R',
            'extra_life': '♥'
        }
        symbol = symbols.get(self.power_type, '?')
        painter.drawText(self.rect, Qt.AlignCenter, symbol)


class Laser:
    """Represents a laser shot from the paddle."""
    def __init__(self, x, y):
        self.rect = QRect(x, y, 3, 10)
        self.velocity_y = -8
        
    def update(self):
        """Update laser position."""
        self.rect.translate(0, self.velocity_y)
        
    def draw(self, painter):
        """Draw the laser."""
        painter.setBrush(QBrush(QColor(255, 0, 0)))
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.rect)


class HighScoreManager:
    """Manages high scores storage and retrieval."""
    def __init__(self):
        self.high_scores = {"Easy": [], "Medium": [], "Hard": [], "Insane": []}
        
    def add_score(self, difficulty, score):
        """Add a new score and return if it's a high score."""
        if difficulty not in self.high_scores:
            self.high_scores[difficulty] = []
            
        self.high_scores[difficulty].append(score)
        self.high_scores[difficulty].sort(reverse=True)
        self.high_scores[difficulty] = self.high_scores[difficulty][:10]  # Keep top 10
        
        return score in self.high_scores[difficulty][:3]  # Top 3 is considered "high score"
        
    def get_high_score(self, difficulty):
        """Get the highest score for a difficulty."""
        if difficulty in self.high_scores and self.high_scores[difficulty]:
            return self.high_scores[difficulty][0]
        return 0


class DifficultyDialog(QDialog):
    """
    Dialog for selecting game difficulty level.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_difficulty = "Easy"  # Default
        self.high_score_manager = HighScoreManager()
        self.init_ui()
        
    def init_ui(self):
        """Initialize the difficulty selection UI."""
        self.setWindowTitle("Select Difficulty")
        self.setModal(True)
        self.setFixedSize(450, 350)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Choose Your Difficulty Level")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(title.font())
        title.font().setPointSize(16)
        title.font().setBold(True)
        layout.addWidget(title)
        
        # Difficulty descriptions with high scores
        difficulties = {
            "Easy": "🟢 Slow ball, large paddle, fewer bricks\nPerfect for beginners!",
            "Medium": "🟡 Balanced gameplay\nThe classic experience",
            "Hard": "🟠 Faster ball, smaller paddle, more bricks\nFor experienced players",
            "Insane": "🔴 Lightning fast, tiny paddle, maximum bricks\nOnly for the brave!"
        }
        
        self.button_group = QButtonGroup()
        
        for i, (difficulty, description) in enumerate(difficulties.items()):
            high_score = self.high_score_manager.get_high_score(difficulty)
            score_text = f"\nHigh Score: {high_score}" if high_score > 0 else "\nHigh Score: ---"
            
            btn = QPushButton(f"{difficulty}\n{description}{score_text}")
            btn.setFixedHeight(60)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, d=difficulty: self.set_difficulty(d))
            self.button_group.addButton(btn, i)
            layout.addWidget(btn)
            
            if difficulty == "Hard":  # Default selection
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
        self.high_score_manager = HighScoreManager()
        self.setup_difficulty_parameters()
        self.initGame()

    def setup_difficulty_parameters(self):
        """Set game parameters based on difficulty level."""
        difficulty_settings = {
            "Easy": {
                "paddle_width_multiplier": 1.5,
                "ball_speed_multiplier": 0.7,
                "game_speed": 10,  # Slower
                "brick_rows": 6,
                "ball_diameter_multiplier": 1.2,
                "score_multiplier": 1.0,
                "width_multiplier": 0.8,  # Smaller play area
                "height_multiplier": 0.8
            },
            "Medium": {
                "paddle_width_multiplier": 1.0,
                "ball_speed_multiplier": 1.0,
                "game_speed": 8,  # Normal
                "brick_rows": 8,
                "ball_diameter_multiplier": 1.0,
                "score_multiplier": 1.5,
                "width_multiplier": 1.0,  # Normal play area
                "height_multiplier": 1.0
            },
            "Hard": {
                "paddle_width_multiplier": 0.7,
                "ball_speed_multiplier": 1.4,
                "game_speed": 6,  # Faster
                "brick_rows": 10,
                "ball_diameter_multiplier": 0.8,
                "score_multiplier": 2.0,
                "width_multiplier": 1.2,  # Larger play area
                "height_multiplier": 1.2
            },
            "Insane": {
                "paddle_width_multiplier": 0.5,
                "ball_speed_multiplier": 2.0,
                "game_speed": 4,  # Very fast
                "brick_rows": 12,
                "ball_diameter_multiplier": 0.6,
                "score_multiplier": 3.0,
                "width_multiplier": 1.4,  # Much larger play area
                "height_multiplier": 1.4
            }
        }
        
        settings = difficulty_settings[self.difficulty]
        
        # Apply difficulty settings including dynamic width/height
        self.WIDTH = int(BASE_WIDTH * settings["width_multiplier"])
        self.HEIGHT = int(BASE_HEIGHT * settings["height_multiplier"])
        self.PADDLE_WIDTH = int(BASE_PADDLE_WIDTH * settings["paddle_width_multiplier"])
        self.BALL_DIAMETER = int(BASE_BALL_DIAMETER * settings["ball_diameter_multiplier"])
        self.GAME_SPEED = settings["game_speed"]
        self.BRICK_ROWS = settings["brick_rows"]
        self.ball_speed_multiplier = settings["ball_speed_multiplier"]
        self.score_multiplier = settings["score_multiplier"]
        
        # Adjust paddle speed based on difficulty (smaller paddles move faster for compensation)
        self.PADDLE_SPEED = int(PADDLE_SPEED * (2 - settings["paddle_width_multiplier"]))

    def initGame(self):
        """Initializes all game variables and objects."""
        # Game state flags
        self.isStarted = False
        self.isPaused = False
        
        # Game progression
        self.current_level = 1
        self.score = 0
        self.lives = 3
        self.combo_count = 0
        
        # Ball speed progression
        self.speed_increase_factor = 1.05  # 5% speed increase per collision
        self.max_speed_multiplier = 5   # Maximum speed multiplier
        
        # Paddle setup
        self.original_paddle_width = self.PADDLE_WIDTH
        self.paddle = QRect((self.WIDTH - self.PADDLE_WIDTH) // 2, 
                            self.HEIGHT - 50, 
                            self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

        # Ball setup
        self.balls = []
        self.resetBall()
        
        # Sticky paddle tracking
        self.sticky_balls = []  # List of balls that are stuck to paddle
        
        # Power-up system
        self.power_ups = []
        self.active_power_ups = {}
        self.power_up_timers = {}
        self.lasers = []
        
        # Visual effects
        self.particles = []
        self.trails = []  # List of active trails

        # Bricks setup
        self.initBricks()

        # Timer for game loop
        self.timer = QBasicTimer()
        
        # Power-up effect timer
        self.effect_timer = QTimer()
        self.effect_timer.timeout.connect(self.update_power_up_effects)
        self.effect_timer.start(100)  # Update every 100ms
        
        # Enable keyboard focus
        self.setFocusPolicy(Qt.StrongFocus)

    def initBricks(self):
        """Creates the grid of bricks based on difficulty and level."""
        self.bricks = []
        
        # Dynamic brick sizing based on screen dimensions
        BRICK_WIDTH = max(40, min(60, self.WIDTH // 15))  # Scale brick width
        BRICK_HEIGHT = max(15, min(25, self.HEIGHT // 35))  # Scale brick height
        BRICK_COLS = max(8, min(14, (self.WIDTH - 40) // (BRICK_WIDTH + 5)))  # Dynamic columns
        PADDING = max(3, self.WIDTH // 160)  # Scale padding
        TOP_OFFSET = max(60, self.HEIGHT // 10)  # Scale top offset

        # Base colors
        base_colors = [QColor(0, 255, 0), QColor(0, 0, 255), QColor(255, 255, 0),
                       QColor(255, 165, 0), QColor(255, 0, 0), QColor(128, 0, 128),
                       QColor(255, 192, 203), QColor(0, 255, 255), QColor(255, 20, 147), QColor(50, 205, 50)]

        # Adjust brick layout based on level and available space
        base_rows = self.BRICK_ROWS + (self.current_level - 1) // 2  # More rows each 2 levels
        max_rows = min(base_rows, (self.HEIGHT - TOP_OFFSET - 150) // (BRICK_HEIGHT + PADDING))  # Ensure bricks fit
        rows = max(3, max_rows)  # Minimum 3 rows
        
        # Center the brick layout horizontally
        total_brick_width = BRICK_COLS * BRICK_WIDTH + (BRICK_COLS - 1) * PADDING
        start_x = (self.WIDTH - total_brick_width) // 2
        
        for row in range(rows):
            for col in range(BRICK_COLS):
                x = start_x + col * (BRICK_WIDTH + PADDING)
                y = row * (BRICK_HEIGHT + PADDING) + TOP_OFFSET
                color = base_colors[row % len(base_colors)]
                
                # Special bricks for higher levels
                brick_type = 'normal'
                brick_hits = 1
                
                if self.current_level >= 3 and random.random() < 0.1:  # 10% chance of strong bricks
                    brick_type = 'strong'
                    brick_hits = 2
                    color = color.darker(150)
                
                self.bricks.append({
                    'rect': QRect(x, y, BRICK_WIDTH, BRICK_HEIGHT), 
                    'color': color,
                    'type': brick_type,
                    'hits': brick_hits,
                    'points': (row + 1) * 10 * self.score_multiplier
                })

    def resetBall(self):
        """Resets the ball to its starting position and state."""
        self.balls = []
        ball = QRect(self.paddle.center().x() - self.BALL_DIAMETER // 2,
                     self.paddle.top() - self.BALL_DIAMETER,
                     self.BALL_DIAMETER, self.BALL_DIAMETER)
        
        # Set initial direction with difficulty-based speed
        base_speed = 1
        self.balls.append({
            'rect': ball,
            'xDir': int(base_speed * self.ball_speed_multiplier),
            'yDir': int(-base_speed * self.ball_speed_multiplier)
        })

    def create_particles(self, x, y, color, count=8):
        """Create particle effects at the given position."""
        for _ in range(count):
            particle = Particle(x, y, color)
            self.particles.append(particle)

    def spawn_power_up(self, x, y):
        """Randomly spawn a power-up at the given position."""
        if random.random() < 0.25:  # 25% chance (increased from 15%)
            power_types = ['multiball', 
                           'big_paddle', 
                           'small_paddle', 
                           'laser', 
                           'slow_ball', 
                           'fast_ball',
                           'reverse_paddle']
            
            if self.current_level >= 3:
                power_types.append('extra_life')
                
            power_type = random.choice(power_types)
            power_up = PowerUp(x, y, power_type)
            self.power_ups.append(power_up)

    def apply_power_up(self, power_type):
        """Apply the effect of a power-up."""
        if power_type == 'multiball':
            # Create additional balls
            for _ in range(2):
                if self.balls:
                    original_ball = self.balls[0]
                    new_ball = {
                        'rect': QRect(original_ball['rect']),
                        'xDir': original_ball['xDir'] + random.randint(-2, 2),
                        'yDir': original_ball['yDir']
                    }
                    self.balls.append(new_ball)
                    
        elif power_type == 'big_paddle':
            self.paddle = QRect(self.paddle.x(), 
                                self.paddle.y(), 
                                min(self.original_paddle_width * 2, 150), 
                                self.PADDLE_HEIGHT)
            
            self.active_power_ups['big_paddle'] = True
            self.power_up_timers['big_paddle'] = 500  # 5 seconds at 100ms intervals
            
        elif power_type == 'small_paddle':
            self.paddle = QRect(self.paddle.x(), 
                                self.paddle.y(), 
                                max(self.original_paddle_width // 2, 30), 
                                self.PADDLE_HEIGHT)
            
            self.active_power_ups['small_paddle'] = True
            self.power_up_timers['small_paddle'] = 500
            
        elif power_type == 'laser':
            self.active_power_ups['laser'] = True
            self.power_up_timers['laser'] = 1000
            
        elif power_type == 'slow_ball':
            # Store original speeds before slowing down
            if 'slow_ball' not in self.active_power_ups:
                self.original_ball_speeds = []
                for ball in self.balls:
                    self.original_ball_speeds.append({'xDir': ball['xDir'], 'yDir': ball['yDir']})
                    
            # Apply slow effect to all balls
            for ball in self.balls:
                new_x_dir = int(ball['xDir'] * 0.5)
                new_y_dir = int(ball['yDir'] * 0.5)
                
                # Ensure minimum movement to prevent ball from stopping
                if abs(new_x_dir) < 1:
                    new_x_dir = 1 if ball['xDir'] >= 0 else -1
                if abs(new_y_dir) < 1:
                    new_y_dir = 1 if ball['yDir'] >= 0 else -1
                    
                ball['xDir'] = new_x_dir
                ball['yDir'] = new_y_dir
                
            self.active_power_ups['slow_ball'] = True
            self.power_up_timers['slow_ball'] = 1000
            
        elif power_type == 'fast_ball':
            # Store original speeds before speeding up
            if 'fast_ball' not in self.active_power_ups:
                self.original_ball_speeds = []
                for ball in self.balls:
                    self.original_ball_speeds.append({'xDir': ball['xDir'], 'yDir': ball['yDir']})
                    
            # Apply fast effect to all balls
            for ball in self.balls:
                new_x_dir = int(ball['xDir'] * 2.0)  # Reduced from 3.0 to 2.0 for better gameplay
                new_y_dir = int(ball['yDir'] * 2.0)
                
                # Ensure reasonable maximum speed
                max_speed = int(6 * self.ball_speed_multiplier)  # Increased max for fast ball
                new_x_dir = max(-max_speed, min(max_speed, new_x_dir))
                new_y_dir = max(-max_speed, min(max_speed, new_y_dir))
                
                # Ensure minimum movement
                if abs(new_x_dir) < 2:  # Higher minimum for fast ball
                    new_x_dir = 2 if ball['xDir'] >= 0 else -2
                if abs(new_y_dir) < 2:
                    new_y_dir = 2 if ball['yDir'] >= 0 else -2
                    
                ball['xDir'] = new_x_dir
                ball['yDir'] = new_y_dir
                
            self.active_power_ups['fast_ball'] = True
            self.power_up_timers['fast_ball'] = 500
            
        elif power_type == 'reverse_paddle':
            self.active_power_ups['reverse_paddle'] = True
            self.power_up_timers['reverse_paddle'] = 800  # 8 seconds
            
        elif power_type == 'extra_life':
            self.lives += 1
            
    def update_power_up_effects(self):
        """Update power-up timers and remove expired effects."""
        expired_effects = []
        
        for effect, timer in self.power_up_timers.items():
            self.power_up_timers[effect] -= 1
            if self.power_up_timers[effect] <= 0:
                expired_effects.append(effect)
                
        for effect in expired_effects:
            self.remove_power_up_effect(effect)
            
    def remove_power_up_effect(self, effect):
        """Remove a power-up effect."""
        if effect in self.active_power_ups:
            del self.active_power_ups[effect]
        if effect in self.power_up_timers:
            del self.power_up_timers[effect]
            
        if effect in ['big_paddle', 'small_paddle']:
            # Reset paddle to original size
            self.paddle = QRect(self.paddle.x(), self.paddle.y(), 
                                self.original_paddle_width, 
                                self.PADDLE_HEIGHT)
        elif effect in ['slow_ball', 'fast_ball']:
            # Restore original ball speeds
            if hasattr(self, 'original_ball_speeds') and self.original_ball_speeds:
                for i, ball in enumerate(self.balls):
                    if i < len(self.original_ball_speeds):
                        # Restore the original direction and speed magnitude
                        original = self.original_ball_speeds[i]
                        ball['xDir'] = original['xDir']
                        ball['yDir'] = original['yDir']
                        
                # Clear the stored speeds
                self.original_ball_speeds = []

    def paintEvent(self, event):
        """Handles all the drawing."""
        painter = QPainter(self)
        
        # Draw white background
        painter.fillRect(self.rect(), QColor(255, 255, 255))
        self.drawObjects(painter)

    def drawObjects(self, painter):
        """Draws all game objects."""
        # Draw UI info
        painter.setPen(QColor(50, 50, 50))
        painter.setFont(QFont('Arial', 12))
        
        # Top row of info
        painter.drawText(10, 20, f"Level: {self.current_level}")
        painter.drawText(10, 40, f"Score: {self.score}")
        painter.drawText(10, 60, f"Lives: {self.lives}")
        
        # Difficulty and high score
        painter.drawText(150, 20, f"Difficulty: {self.difficulty}")
        high_score = self.high_score_manager.get_high_score(self.difficulty)
        painter.drawText(150, 40, f"High Score: {high_score}")
        
        # Active power-ups indicator
        if self.active_power_ups:
            painter.drawText(350, 20, "Active Power-ups:")
            y_offset = 40
            for power_up in self.active_power_ups.keys():
                painter.drawText(350, y_offset, f"• {power_up.replace('_', ' ').title()}")
                y_offset += 15
        
        # Draw Paddle
        paddle_color = QColor(0, 180, 255)
        if 'laser' in self.active_power_ups:
            paddle_color = QColor(255, 0, 0)  # Red for laser mode
        elif 'sticky_paddle' in self.active_power_ups:
            paddle_color = QColor(255, 255, 0)  # Yellow for sticky
            
        painter.setBrush(paddle_color)
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.paddle)

        # Draw Trails (behind balls)
        for trail in self.trails:
            trail.draw(painter)
            
        # Draw Balls
        for ball in self.balls:
            painter.setBrush(QColor(50, 50, 50))
            painter.drawEllipse(ball['rect'])

        # Draw Bricks
        for brick in self.bricks:
            painter.setBrush(brick['color'])
            painter.setPen(QPen(QColor(30, 30, 30), 2))
            painter.drawRect(brick['rect'])
            
            # Draw hit indicator for strong bricks
            if brick['type'] == 'strong' and brick['hits'] > 1:
                painter.setPen(QPen(QColor(255, 255, 255), 2))
                painter.setFont(QFont('Arial', 8, QFont.Bold))
                painter.drawText(brick['rect'], Qt.AlignCenter, str(brick['hits']))
                
        # Draw Power-ups
        for power_up in self.power_ups:
            power_up.draw(painter)
            
        # Draw Lasers
        for laser in self.lasers:
            laser.draw(painter)
            
        # Draw Particles
        for particle in self.particles:
            particle.draw(painter)
            
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
        old_paddle_x = self.paddle.x()

        # Check if reverse paddle is active
        is_reversed = 'reverse_paddle' in self.active_power_ups

        if key == Qt.Key_A:
            # Move left normally, or right if reversed
            if is_reversed:
                if self.paddle.right() < self.WIDTH:
                    self.paddle.translate(self.PADDLE_SPEED, 0)
                    self.move_stuck_balls_with_paddle(old_paddle_x)
                    # Also move ball if game hasn't started
                    if not self.isStarted:
                        self.move_ball_with_paddle_before_start(old_paddle_x)
            else:
                if self.paddle.left() > 0:
                    self.paddle.translate(-self.PADDLE_SPEED, 0)
                    self.move_stuck_balls_with_paddle(old_paddle_x)
                    # Also move ball if game hasn't started
                    if not self.isStarted:
                        self.move_ball_with_paddle_before_start(old_paddle_x)
        elif key == Qt.Key_D:
            # Move right normally, or left if reversed
            if is_reversed:
                if self.paddle.left() > 0:
                    self.paddle.translate(-self.PADDLE_SPEED, 0)
                    self.move_stuck_balls_with_paddle(old_paddle_x)
                    # Also move ball if game hasn't started
                    if not self.isStarted:
                        self.move_ball_with_paddle_before_start(old_paddle_x)
            else:
                if self.paddle.right() < self.WIDTH:
                    self.paddle.translate(self.PADDLE_SPEED, 0)
                    self.move_stuck_balls_with_paddle(old_paddle_x)
                    # Also move ball if game hasn't started
                    if not self.isStarted:
                        self.move_ball_with_paddle_before_start(old_paddle_x)
        elif key == Qt.Key_Space:
            if not self.isStarted and not self.isPaused:
                self.startGame()
            elif self.isStarted and not self.isPaused:
                # Check if we have stuck balls (from sticky paddle)
                stuck_balls = [ball for ball in self.balls if ball['xDir'] == 0 and ball['yDir'] == 0]
                if stuck_balls:
                    # Release stuck balls
                    self.release_stuck_balls()
                elif 'laser' in self.active_power_ups:
                    # Shoot laser
                    laser_x = self.paddle.center().x() - 1
                    laser_y = self.paddle.top() - 10
                    laser = Laser(laser_x, laser_y)
                    self.lasers.append(laser)
        elif key == Qt.Key_P:
            self.pauseGame()
        
        self.update()  # Redraw after key press

    def move_ball_with_paddle_before_start(self, old_paddle_x):
        """Move the ball with the paddle when the game hasn't started yet."""
        if not self.isStarted and self.balls:
            paddle_movement = self.paddle.x() - old_paddle_x
            # Move the ball to stay centered on the paddle
            for ball in self.balls:
                ball['rect'].translate(paddle_movement, 0)
                
    def move_stuck_balls_with_paddle(self, old_paddle_x):
        """Move any stuck balls along with the paddle movement."""
        if 'sticky_paddle' not in self.active_power_ups:
            return
            
        paddle_movement = self.paddle.x() - old_paddle_x
        
        for ball in self.balls:
            # Only move balls that are stuck (velocity is 0)
            if ball['xDir'] == 0 and ball['yDir'] == 0:
                ball['rect'].translate(paddle_movement, 0)
                
    def release_stuck_balls(self):
        """Release all stuck balls from the sticky paddle."""
        for ball in self.balls:
            if ball['xDir'] == 0 and ball['yDir'] == 0:
                # Calculate release angle based on ball position relative to paddle center
                paddle_center = self.paddle.center().x()
                ball_center = ball['rect'].center().x()
                
                # Determine angle based on position on paddle
                relative_position = (ball_center - paddle_center) / (self.paddle.width() / 2)
                # Clamp to reasonable range
                relative_position = max(-1, min(1, relative_position))
                
                # Convert to launch angle (between -60 and 60 degrees)
                launch_angle = relative_position * 60  # degrees
                
                # Set ball velocity
                base_speed = max(2, int(self.ball_speed_multiplier))
                ball['xDir'] = int(base_speed * math.sin(math.radians(launch_angle)))
                ball['yDir'] = int(-base_speed * math.cos(math.radians(launch_angle)))
                
                # Ensure minimum movement
                if abs(ball['xDir']) < 1:
                    ball['xDir'] = 1 if ball['xDir'] >= 0 else -1
                if abs(ball['yDir']) < 1:
                    ball['yDir'] = -1

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
            self.moveBalls()
            self.updatePowerUps()
            self.updateLasers()
            self.updateParticles()
            self.updateTrails()
            self.checkCollision()
            self.update()

    def moveBalls(self):
        """Moves all balls according to their direction vectors."""
        if self.isStarted and not self.isPaused:
            for ball in self.balls:
                # Create trail at current position before moving
                ball_center = ball['rect'].center()
                trail = BallTrail(ball_center.x() - self.BALL_DIAMETER // 4, 
                                  ball_center.y() - self.BALL_DIAMETER // 4, 
                                  self.BALL_DIAMETER // 2)
                self.trails.append(trail)
                
                # Move the ball
                ball['rect'].translate(ball['xDir'], ball['yDir'])

    def updatePowerUps(self):
        """Update power-up positions and remove those that fall off screen."""
        for power_up in self.power_ups[:]:
            power_up.update()
            if power_up.rect.top() > self.HEIGHT:
                self.power_ups.remove(power_up)
                
    def updateLasers(self):
        """Update laser positions and remove those that go off screen."""
        for laser in self.lasers[:]:
            laser.update()
            if laser.rect.bottom() < 0:
                self.lasers.remove(laser)
                
    def updateParticles(self):
        """Update particle effects."""
        for particle in self.particles[:]:
            if not particle.update():
                self.particles.remove(particle)

    def updateTrails(self):
        """Update trail effects."""
        for trail in self.trails[:]:
            if not trail.update():
                self.trails.remove(trail)

    def checkCollision(self):
        """Checks for and handles collisions with comprehensive safeguards."""
        balls_to_remove = []
        
        for i, ball in enumerate(self.balls):
            # Enforce strict boundary safeguards to prevent getting stuck
            self.enforce_ball_boundaries(ball)
            
            # Ball hits bottom wall -> Lose a life
            if ball['rect'].bottom() > self.HEIGHT:
                balls_to_remove.append(i)
                continue

            # Ball hits walls with enhanced collision detection
            wall_collision = False
            
            # Left wall collision
            if ball['rect'].left() <= 0:
                ball['rect'].moveLeft(1)  # Move ball away from wall
                ball['xDir'] = abs(ball['xDir'])  # Ensure positive direction
                wall_collision = True
                
            # Right wall collision  
            elif ball['rect'].right() >= self.WIDTH:
                ball['rect'].moveRight(self.WIDTH - 1)  # Move ball away from wall
                ball['xDir'] = -abs(ball['xDir'])  # Ensure negative direction
                wall_collision = True
                
            # Top wall collision
            if ball['rect'].top() <= 0:
                ball['rect'].moveTop(1)  # Move ball away from wall
                ball['yDir'] = abs(ball['yDir'])  # Ensure positive direction (downward)
                wall_collision = True
                
            # If we had a wall collision, increase speed and ensure minimum speed
            if wall_collision:
                self.increase_ball_speed(ball)
                self.ensure_minimum_ball_speed(ball)

            # Ball hits paddle
            if ball['rect'].intersects(self.paddle) and ball['yDir'] > 0:
                # Move ball above paddle to prevent sticking
                ball['rect'].moveBottom(self.paddle.top() - 2)  # Extra margin
                
                if 'sticky_paddle' in self.active_power_ups:
                    # Sticky paddle - stop the ball
                    ball['xDir'] = 0
                    ball['yDir'] = 0
                else:
                    # Complex paddle collision for better control
                    paddle_center = self.paddle.center().x()
                    ball_center = ball['rect'].center().x()
                    
                    # Change horizontal direction based on where it hits the paddle
                    hit_factor = (ball_center - paddle_center) / (self.paddle.width() / 2)
                    max_speed = int(2 * self.ball_speed_multiplier)
                    ball['xDir'] = max(-max_speed, min(max_speed, int(hit_factor * max_speed)))
                    if ball['xDir'] == 0:
                        if hit_factor >= 0:
                            ball['xDir'] = int(self.ball_speed_multiplier)
                        else:
                            ball['xDir'] = int(-self.ball_speed_multiplier)
                    ball['yDir'] = int(-1 * self.ball_speed_multiplier)  # Always bounce up
                    
                    # Increase speed after paddle collision and ensure minimum speed
                    self.increase_ball_speed(ball)
                    self.ensure_minimum_ball_speed(ball)

            # Ball hits a brick
            brick_to_remove = -1
            for j, brick in enumerate(self.bricks):
                if ball['rect'].intersects(brick['rect']):
                    # Handle collision
                    self.handleBrickCollision(ball, brick, j)
                    brick_to_remove = j
                    break

            # Remove the brick after bouncing
            if brick_to_remove != -1:
                brick = self.bricks[brick_to_remove]
                
                # Create particles
                self.create_particles(
                    brick['rect'].center().x(),
                    brick['rect'].center().y(),
                    brick['color'],
                    count=4
                )
                
                # Spawn power-up chance
                self.spawn_power_up(brick['rect'].center().x(), brick['rect'].center().y())
                
                # Add score with combo multiplier
                self.combo_count += 1
                score_bonus = int(brick['points'] * (1 + self.combo_count * 0.1))
                self.score += score_bonus
                
                # Remove brick if it's destroyed
                if brick['hits'] <= 1:
                    del self.bricks[brick_to_remove]
                
        # Remove balls that fell off screen
        for i in reversed(balls_to_remove):
            del self.balls[i]
            
        # Check if all balls are gone
        if not self.balls:
            self.lives -= 1
            self.combo_count = 0  # Reset combo
            if self.lives <= 0:
                self.timer.stop()
                self.parent_window.game_over(won=False, score=self.score)
                return
            else:
                # Reset ball for next life
                self.resetBall()
                self.isStarted = False
                
        # Check for power-up collection
        for power_up in self.power_ups[:]:
            if power_up.rect.intersects(self.paddle):
                self.apply_power_up(power_up.power_type)
                self.power_ups.remove(power_up)
                
        # Check laser collisions with bricks
        for laser in self.lasers[:]:
            for j, brick in enumerate(self.bricks):
                if laser.rect.intersects(brick['rect']):
                    # Laser hits brick
                    self.create_particles(
                        brick['rect'].center().x(),
                        brick['rect'].center().y(),
                        brick['color'],
                        count=4
                    )
                    
                    # Add score
                    self.score += int(brick['points'] * 0.5)  # Half points for laser
                    
                    # Damage brick
                    brick['hits'] -= 1
                    if brick['hits'] <= 0:
                        del self.bricks[j]
                        
                    # Remove laser
                    self.lasers.remove(laser)
                    break

        # Check for win condition
        if not self.bricks:
            self.timer.stop()
            self.advance_level()

    def enforce_ball_boundaries(self, ball):
        """Enforce strict boundaries to prevent ball from getting stuck."""
        # Define safety margins
        margin = 2
        
        # Check if ball is outside boundaries and correct position
        if ball['rect'].left() < 0:
            ball['rect'].moveLeft(margin)
            if ball['xDir'] <= 0:
                ball['xDir'] = max(1, abs(ball['xDir']))
                
        elif ball['rect'].right() > self.WIDTH:
            ball['rect'].moveRight(self.WIDTH - margin)
            if ball['xDir'] >= 0:
                ball['xDir'] = min(-1, -abs(ball['xDir']))
                
        if ball['rect'].top() < 0:
            ball['rect'].moveTop(margin)
            if ball['yDir'] <= 0:
                ball['yDir'] = max(1, abs(ball['yDir']))
                
        # Detect if ball is moving too slowly or stopped (potential stuck situation)
        if abs(ball['xDir']) < 1 and abs(ball['yDir']) < 1:
            # Ball is essentially stopped, give it some momentum
            angle = random.uniform(30, 150)  # Random angle between 30-150 degrees
            speed = max(2, int(self.ball_speed_multiplier))
            ball['xDir'] = int(speed * math.cos(math.radians(angle)))
            ball['yDir'] = int(speed * math.sin(math.radians(angle)))
            
        # Detect if ball is in a corner and might be stuck
        corner_margin = 20
        if ((ball['rect'].left() < corner_margin and ball['rect'].top() < corner_margin) or
            (ball['rect'].right() > self.WIDTH - corner_margin and ball['rect'].top() < corner_margin)):
            # Ball is in a top corner, push it toward center
            ball['rect'].moveCenter(QPoint(self.WIDTH // 2, ball['rect'].center().y() + 20))
            ball['xDir'] = random.choice([-1, 1]) * max(1, abs(ball['xDir']))
            ball['yDir'] = max(1, abs(ball['yDir']))

    def ensure_minimum_ball_speed(self, ball):
        """Ensure ball maintains minimum speed to prevent getting stuck."""
        # Check if we have active speed modifiers
        if 'slow_ball' in self.active_power_ups:
            min_speed = max(1, int(self.ball_speed_multiplier * 0.3))  # Lower minimum for slow ball
        elif 'fast_ball' in self.active_power_ups:
            min_speed = max(2, int(self.ball_speed_multiplier * 0.8))  # Higher minimum for fast ball
        else:
            min_speed = max(1, int(self.ball_speed_multiplier * 0.5))
        
        # Ensure minimum horizontal speed
        if abs(ball['xDir']) < min_speed:
            ball['xDir'] = min_speed if ball['xDir'] >= 0 else -min_speed
            
        # Ensure minimum vertical speed  
        if abs(ball['yDir']) < min_speed:
            ball['yDir'] = min_speed if ball['yDir'] >= 0 else -min_speed
            
        # Set maximum speed based on active power-ups
        if 'fast_ball' in self.active_power_ups:
            max_speed = int(6 * self.ball_speed_multiplier)  # Higher max for fast ball
        elif 'slow_ball' in self.active_power_ups:
            max_speed = int(2 * self.ball_speed_multiplier)  # Lower max for slow ball
        else:
            max_speed = int(4 * self.ball_speed_multiplier)
            
        ball['xDir'] = max(-max_speed, min(max_speed, ball['xDir']))
        ball['yDir'] = max(-max_speed, min(max_speed, ball['yDir']))

    def handleBrickCollision(self, ball, brick, brick_index):
        """Handle collision between ball and brick with enhanced anti-stuck measures."""
        # Get the overlap amount to determine bounce direction more accurately
        ball_rect = ball['rect']
        brick_rect = brick['rect']
        
        # Calculate overlap on each side
        overlap_left = ball_rect.right() - brick_rect.left()
        overlap_right = brick_rect.right() - ball_rect.left()
        overlap_top = ball_rect.bottom() - brick_rect.top()
        overlap_bottom = brick_rect.bottom() - ball_rect.top()
        
        # Find the minimum overlap to determine collision side
        min_overlap = min(overlap_left, overlap_right, overlap_top, overlap_bottom)
        
        # Add safety margin for ball repositioning
        safety_margin = 3
        
        # Bounce based on the side with minimum overlap
        if min_overlap == overlap_left and ball['xDir'] > 0:
            ball['xDir'] = -abs(ball['xDir'])  # Ensure negative direction
            ball['rect'].moveRight(brick_rect.left() - safety_margin)
        elif min_overlap == overlap_right and ball['xDir'] < 0:
            ball['xDir'] = abs(ball['xDir'])  # Ensure positive direction
            ball['rect'].moveLeft(brick_rect.right() + safety_margin)
        elif min_overlap == overlap_top and ball['yDir'] > 0:
            ball['yDir'] = -abs(ball['yDir'])  # Ensure negative direction
            ball['rect'].moveBottom(brick_rect.top() - safety_margin)
        elif min_overlap == overlap_bottom and ball['yDir'] < 0:
            ball['yDir'] = abs(ball['yDir'])  # Ensure positive direction
            ball['rect'].moveTop(brick_rect.bottom() + safety_margin)
        else:
            # Fallback: if we can't determine collision side clearly, 
            # reverse both directions and move ball away from brick center
            ball['xDir'] = -ball['xDir']
            ball['yDir'] = -ball['yDir']
            
            # Move ball away from brick center
            brick_center = brick_rect.center()
            ball_center = ball_rect.center()
            
            if ball_center.x() < brick_center.x():
                ball['rect'].moveRight(brick_rect.left() - safety_margin)
            else:
                ball['rect'].moveLeft(brick_rect.right() + safety_margin)
                
            if ball_center.y() < brick_center.y():
                ball['rect'].moveBottom(brick_rect.top() - safety_margin)
            else:
                ball['rect'].moveTop(brick_rect.bottom() + safety_margin)
                
        # Ensure minimum speed after collision
        self.ensure_minimum_ball_speed(ball)
        
        # Increase speed after brick collision for progressive difficulty
        self.increase_ball_speed(ball)
            
        # Damage the brick
        brick['hits'] -= 1
        if brick['hits'] <= 0:
            # Brick is destroyed, will be removed by caller
            pass
        else:
            # Brick still has hits, make it darker
            brick['color'] = brick['color'].darker(120)

    def increase_ball_speed(self, ball):
        """Increase ball speed after collision for progressive difficulty."""
        # Calculate current speed magnitude
        current_speed = math.sqrt(ball['xDir']**2 + ball['yDir']**2)
        
        # Check if we haven't exceeded the maximum speed
        max_allowed_speed = int(4 * self.ball_speed_multiplier * self.max_speed_multiplier)
        
        if current_speed < max_allowed_speed:
            # Increase speed by the factor
            ball['xDir'] = int(ball['xDir'] * self.speed_increase_factor)
            ball['yDir'] = int(ball['yDir'] * self.speed_increase_factor)
            
            # Ensure we don't exceed maximum speed
            new_speed = math.sqrt(ball['xDir']**2 + ball['yDir']**2)
            if new_speed > max_allowed_speed:
                # Scale back to maximum allowed speed
                scale_factor = max_allowed_speed / new_speed
                ball['xDir'] = int(ball['xDir'] * scale_factor)
                ball['yDir'] = int(ball['yDir'] * scale_factor)
                
        # Ensure minimum speed is maintained
        self.ensure_minimum_ball_speed(ball)

    def advance_level(self):
        """Advance to the next level."""
        self.current_level += 1
        
        # Bonus points for completing level
        level_bonus = self.current_level * 500 * self.score_multiplier
        self.score += int(level_bonus)
        
        # Show level complete message
        QMessageBox.information(
            self,
            "Level Complete!",
            f"Level {self.current_level - 1} Complete!\n"
            f"Level Bonus: {int(level_bonus)} points\n"
            f"Current Score: {self.score}\n\n"
            f"Starting Level {self.current_level}..."
        )
        
        # Reset for next level
        self.initBricks()
        self.resetBall()
        self.isStarted = False
        self.combo_count = 0
        
        # Clear active power-ups for new level
        self.active_power_ups.clear()
        self.power_up_timers.clear()
        self.power_ups.clear()
        self.lasers.clear()
        
        # Reset paddle to original size
        self.paddle = QRect(self.paddle.x(), self.paddle.y(), 
                            self.original_paddle_width, self.PADDLE_HEIGHT)


class BreakoutGame(QMainWindow):
    """
    Main application window for the Breakout game.
    """
    def __init__(self, main_window, parent=None):
        super().__init__()
        self.main_window = main_window
        self.high_score_manager = HighScoreManager()
        
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
                "Features:\n"
                " - Multiple difficulty levels with score multipliers\n"
                " - Power-ups: Multiball, Paddle size, Sticky paddle, Lasers, etc.\n"
                " - Particle effects and visual enhancements\n"
                " - High score tracking\n"
                " - Progressive levels with increasing difficulty\n"
                " - Lives system and combo scoring\n\n"
                "Controls:\n"
                " - Use 'A' to move paddle left\n"
                " - Use 'D' to move paddle right\n"
                " - Press SPACE to start game or shoot lasers\n"
                " - Press P to pause/unpause\n\n"
                "Click 'OK' to start playing!"
            )
            if QMessageBox.information(self, 
                                       "Welcome to Breakout", 
                                       welcome_msg, QMessageBox.Ok) == QMessageBox.Ok:
                self.init_game()                   # Create board and set dimensions.
                self.init_ui()                     # Set up UI dimensions.
        else:
            self.close()

    def end_game(self):
        """End the game by stopping the timer and closing the window."""
        if self.board:
            self.board.timer.stop()
            if hasattr(self.board, 'effect_timer'):
                self.board.effect_timer.stop()
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

    def game_over(self, won=False, score=0):
        """Display game over message and handle restart/close."""
        # Check if it's a high score
        is_high_score = self.high_score_manager.add_score(self.difficulty, score)
        
        if won:
            title = "Congratulations!"
            message = f"You completed all levels on {self.difficulty} mode!\n\n"
        else:
            title = "Game Over"
            message = f"Game over on {self.difficulty} mode!\n\n"
            
        message += f"Final Score: {score}\n"
        message += f"Level Reached: {self.board.current_level}\n"
        
        if is_high_score:
            message += "\n🎉 NEW HIGH SCORE! 🎉\n"
            
        message += "\nDo you want to play again?"
            
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