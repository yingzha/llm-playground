"""
Sprite animation system for the Live Dashboard Wallpaper.
Handles sprite sheet loading and frame-by-frame animation.
"""

import pygame
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional


class AnimationState(Enum):
    IDLE = "idle"
    WALKING_LEFT = "walking_left"
    WALKING_RIGHT = "walking_right"
    SITTING = "sitting"


@dataclass
class SpriteFrame:
    """Represents a single frame from a sprite sheet."""
    surface: pygame.Surface
    duration_ms: int = 200  # How long to display this frame


class AnimatedSprite:
    """
    Handles sprite sheet animation for characters like Chunhua.

    Sprite sheet layout expected:
    - Top row: Standing/idle poses (3 frames)
    - Bottom row: Walking animation frames (3 frames)
    """

    def __init__(
        self,
        sprite_sheet_path: str,
        frame_width: int,
        frame_height: int,
        scale: float = 1.0
    ):
        self.sprite_sheet = pygame.image.load(sprite_sheet_path).convert_alpha()
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.scale = scale

        # Current state
        self.position = pygame.Vector2(0, 0)
        self.state = AnimationState.IDLE
        self.current_frame_index = 0
        self.frame_timer = 0
        self.facing_right = True

        # Movement
        self.target_position: Optional[pygame.Vector2] = None
        self.move_speed = 50  # pixels per second

        # Extract frames from sprite sheet
        self.idle_frames: List[SpriteFrame] = []
        self.walk_frames: List[SpriteFrame] = []
        self._extract_frames()

    def _extract_frames(self):
        """Extract individual frames from the sprite sheet."""
        # Top row: idle/standing frames (columns 0, 1, 2)
        for col in range(3):
            frame = self._get_frame(col, 0)
            self.idle_frames.append(SpriteFrame(frame, duration_ms=500))

        # Bottom row: walking frames (columns 0, 1, 2)
        for col in range(3):
            frame = self._get_frame(col, 1)
            self.walk_frames.append(SpriteFrame(frame, duration_ms=150))

    def _get_frame(self, col: int, row: int) -> pygame.Surface:
        """Extract a single frame from the sprite sheet."""
        x = col * self.frame_width
        y = row * self.frame_height

        frame = pygame.Surface((self.frame_width, self.frame_height), pygame.SRCALPHA)
        frame.blit(self.sprite_sheet, (0, 0), (x, y, self.frame_width, self.frame_height))

        if self.scale != 1.0:
            new_size = (int(self.frame_width * self.scale), int(self.frame_height * self.scale))
            frame = pygame.transform.scale(frame, new_size)

        return frame

    def set_position(self, x: float, y: float):
        """Set the sprite's current position."""
        self.position = pygame.Vector2(x, y)

    def set_facing(self, right: bool):
        """Manually set the facing direction."""
        self.facing_right = right

    def is_moving(self) -> bool:
        """Check if the sprite is currently moving to a target."""
        return self.target_position is not None

    def move_to(self, x: float, y: float):
        """Start moving the sprite to a target position."""
        self.target_position = pygame.Vector2(x, y)

        # Determine walking direction
        if self.target_position.x > self.position.x:
            self.state = AnimationState.WALKING_RIGHT
            self.facing_right = True
        else:
            self.state = AnimationState.WALKING_LEFT
            self.facing_right = False

    def update(self, dt_ms: int):
        """Update animation and movement."""
        # Update frame animation
        self.frame_timer += dt_ms
        current_frames = self._get_current_frames()

        if current_frames:
            current_frame = current_frames[self.current_frame_index]
            if self.frame_timer >= current_frame.duration_ms:
                self.frame_timer = 0
                self.current_frame_index = (self.current_frame_index + 1) % len(current_frames)

        # Update movement
        if self.target_position:
            direction = self.target_position - self.position
            distance = direction.length()

            if distance < 2:  # Close enough, stop moving
                self.position = self.target_position
                self.target_position = None
                self.state = AnimationState.IDLE
                self.current_frame_index = 0
            else:
                # Move towards target
                direction = direction.normalize()
                move_amount = self.move_speed * (dt_ms / 1000.0)
                self.position += direction * min(move_amount, distance)

    def _get_current_frames(self) -> List[SpriteFrame]:
        """Get the frame list for the current animation state."""
        if self.state == AnimationState.IDLE or self.state == AnimationState.SITTING:
            return self.idle_frames
        else:
            return self.walk_frames

    def draw(self, surface: pygame.Surface):
        """Draw the current frame to the surface."""
        current_frames = self._get_current_frames()
        if not current_frames:
            return

        frame = current_frames[self.current_frame_index].surface

        # Flip horizontally if facing left
        if not self.facing_right:
            frame = pygame.transform.flip(frame, True, False)

        # Draw centered on position
        rect = frame.get_rect(midbottom=(int(self.position.x), int(self.position.y)))
        surface.blit(frame, rect)

    def get_rect(self) -> pygame.Rect:
        """Get the bounding rectangle of the sprite."""
        current_frames = self._get_current_frames()
        if current_frames:
            frame = current_frames[0].surface
            return frame.get_rect(midbottom=(int(self.position.x), int(self.position.y)))
        return pygame.Rect(0, 0, 0, 0)
