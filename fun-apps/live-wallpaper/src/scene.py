"""
Scene manager for the Live Dashboard Wallpaper.
Handles background, day/night cycle, and sprite management.
"""

import pygame
from datetime import datetime
from typing import List, Tuple, Optional
from dataclasses import dataclass
import random


@dataclass
class WaypointConfig:
    """Defines a location where Chunhua can move to."""
    name: str
    position: Tuple[int, int]  # (x, y) coordinates
    weight: float = 1.0  # Probability weight for random selection
    facing_right: bool = True  # Direction cat faces when idle at this location
    min_stay_ms: int = 30000  # Minimum stay duration (30 seconds default)
    max_stay_ms: int = 60000  # Maximum stay duration (60 seconds default)


@dataclass
class TimeOfDay:
    """Represents different times of day for visual changes."""
    DAWN = "dawn"      # 5:00 - 7:00
    DAY = "day"        # 7:00 - 17:00
    DUSK = "dusk"      # 17:00 - 19:00
    NIGHT = "night"    # 19:00 - 5:00


class SceneManager:
    """
    Manages the wallpaper scene including:
    - Base background image
    - Day/night window overlay
    - Sprite waypoints and movement scheduling
    """

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        base_scene_path: str,
        window_day_path: Optional[str] = None,
        window_night_path: Optional[str] = None,
        window_rect: Optional[Tuple[int, int, int, int]] = None
    ):
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Load base scene
        self.base_scene = pygame.image.load(base_scene_path).convert_alpha()
        self.base_scene = pygame.transform.scale(
            self.base_scene, (screen_width, screen_height)
        )

        # Window overlays for day/night cycle
        self.window_day: Optional[pygame.Surface] = None
        self.window_night: Optional[pygame.Surface] = None
        self.window_rect = window_rect  # (x, y, width, height) of window area

        if window_day_path:
            self.window_day = pygame.image.load(window_day_path).convert_alpha()
        if window_night_path:
            self.window_night = pygame.image.load(window_night_path).convert_alpha()

        # Waypoints where Chunhua can move
        self.waypoints: List[WaypointConfig] = []

        # Current time of day
        self.current_time_of_day = self._get_time_of_day()

        # Tint colors for different times
        self.time_tints = {
            TimeOfDay.DAWN: (255, 220, 200, 30),    # Warm orange tint
            TimeOfDay.DAY: (255, 255, 255, 0),      # No tint
            TimeOfDay.DUSK: (255, 180, 150, 40),    # Orange/pink tint
            TimeOfDay.NIGHT: (100, 120, 180, 60),   # Blue tint
        }

    def _get_time_of_day(self) -> str:
        """Determine current time of day based on system clock."""
        hour = datetime.now().hour

        if 5 <= hour < 7:
            return TimeOfDay.DAWN
        elif 7 <= hour < 17:
            return TimeOfDay.DAY
        elif 17 <= hour < 19:
            return TimeOfDay.DUSK
        else:
            return TimeOfDay.NIGHT

    def add_waypoint(
        self,
        name: str,
        x: int,
        y: int,
        weight: float = 1.0,
        facing_right: bool = True,
        min_stay_ms: int = 30000,
        max_stay_ms: int = 60000
    ):
        """Add a waypoint where Chunhua can visit."""
        self.waypoints.append(WaypointConfig(
            name=name,
            position=(x, y),
            weight=weight,
            facing_right=facing_right,
            min_stay_ms=min_stay_ms,
            max_stay_ms=max_stay_ms
        ))

    def get_random_waypoint(self, exclude_current: Optional[Tuple[int, int]] = None) -> WaypointConfig:
        """Get a random waypoint, optionally excluding the current position."""
        available = [wp for wp in self.waypoints]

        if exclude_current and len(available) > 1:
            available = [
                wp for wp in available
                if abs(wp.position[0] - exclude_current[0]) > 50 or
                   abs(wp.position[1] - exclude_current[1]) > 50
            ]

        if not available:
            available = self.waypoints

        # Weighted random selection
        total_weight = sum(wp.weight for wp in available)
        r = random.uniform(0, total_weight)
        cumulative = 0

        for wp in available:
            cumulative += wp.weight
            if r <= cumulative:
                return wp

        return available[-1]

    def update(self):
        """Update scene state (e.g., check for time changes)."""
        new_time = self._get_time_of_day()
        if new_time != self.current_time_of_day:
            self.current_time_of_day = new_time
            # Could trigger events here (e.g., Chunhua yawns at night)

    def draw(self, surface: pygame.Surface):
        """Draw the scene to the surface."""
        # Draw base scene
        surface.blit(self.base_scene, (0, 0))

        # Draw window overlay based on time of day
        if self.window_rect:
            self._draw_window_overlay(surface)

        # Apply time-of-day tint
        self._apply_time_tint(surface)

    def _draw_window_overlay(self, surface: pygame.Surface):
        """Draw the appropriate window view based on time."""
        if not self.window_rect:
            return

        x, y, w, h = self.window_rect

        if self.current_time_of_day == TimeOfDay.NIGHT:
            if self.window_night:
                window_img = pygame.transform.scale(self.window_night, (w, h))
                surface.blit(window_img, (x, y))
        else:
            if self.window_day:
                window_img = pygame.transform.scale(self.window_day, (w, h))
                surface.blit(window_img, (x, y))

    def _apply_time_tint(self, surface: pygame.Surface):
        """Apply a subtle color tint based on time of day."""
        tint = self.time_tints.get(self.current_time_of_day)
        if tint and tint[3] > 0:  # If there's any alpha
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill(tint)
            surface.blit(overlay, (0, 0))


class MovementScheduler:
    """
    Schedules Chunhua's movements around the room.
    """

    def __init__(
        self,
        min_wait_ms: int = 30000,   # 30 seconds minimum wait
        max_wait_ms: int = 120000,  # 2 minutes maximum wait
    ):
        self.min_wait = min_wait_ms
        self.max_wait = max_wait_ms
        self.time_until_next_move = self._get_random_wait()

    def _get_random_wait(self) -> int:
        """Get a random wait time in milliseconds."""
        return random.randint(self.min_wait, self.max_wait)

    def update(self, dt_ms: int) -> bool:
        """
        Update the scheduler.
        Returns True if it's time to trigger a new movement.
        """
        self.time_until_next_move -= dt_ms

        if self.time_until_next_move <= 0:
            self.time_until_next_move = self._get_random_wait()
            return True

        return False

    def force_next_move(self):
        """Force the next movement to happen immediately."""
        self.time_until_next_move = 0
