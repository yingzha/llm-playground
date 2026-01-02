"""
Live Dashboard Wallpaper - Main Entry Point

A cozy living room wallpaper featuring Chunhua the cat
moving around with day/night cycle support.
"""

import pygame
import sys
import random
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from scene import SceneManager, MovementScheduler
from sprite import AnimatedSprite


# Configuration
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
FPS = 60
WINDOW_TITLE = "Live Dashboard Wallpaper - Chunhua's Room"

# Asset paths
ASSETS_DIR = Path(__file__).parent / "assets"
BACKGROUND_PATH = ASSETS_DIR / "backgrounds" / "background.jpg"
SPRITE_PATH = ASSETS_DIR / "sprites" / "chunhua_fixed.png"


def get_sprite_frame_dimensions(sprite_path: Path) -> tuple[int, int]:
    """Calculate frame dimensions from sprite sheet (3 cols x 2 rows)."""
    temp_surface = pygame.image.load(str(sprite_path))
    sheet_width = temp_surface.get_width()
    sheet_height = temp_surface.get_height()
    return sheet_width // 3, sheet_height // 2


def setup_waypoints(scene: SceneManager):
    """Define locations where Chunhua can visit - fireplace area only."""
    # In front of fireplace - warming up
    scene.add_waypoint(
        name="fireplace_front",
        x=1050,
        y=850,
        weight=1.5,
        facing_right=False,  # Face the fireplace
        min_stay_ms=30000,
        max_stay_ms=60000
    )

    # Left side of fireplace
    scene.add_waypoint(
        name="fireplace_left",
        x=850,
        y=820,
        weight=1.0,
        facing_right=True,  # Face toward fireplace
        min_stay_ms=20000,
        max_stay_ms=40000
    )

    # Right side of fireplace (near plants)
    scene.add_waypoint(
        name="fireplace_right",
        x=1250,
        y=830,
        weight=1.0,
        facing_right=False,  # Face toward fireplace
        min_stay_ms=20000,
        max_stay_ms=40000
    )

    # On the rug near fireplace
    scene.add_waypoint(
        name="fireplace_rug",
        x=1000,
        y=920,
        weight=1.2,
        facing_right=False,
        min_stay_ms=25000,
        max_stay_ms=50000
    )


class LiveWallpaper:
    """Main application class for the live wallpaper."""

    def __init__(self):
        pygame.init()

        # Create display (use fullscreen with F key toggle)
        self.fullscreen = False
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(WINDOW_TITLE)
        self.clock = pygame.time.Clock()

        # Validate assets exist
        if not BACKGROUND_PATH.exists():
            raise FileNotFoundError(
                f"Background image not found: {BACKGROUND_PATH}\n"
                "Please save the living room image to this location."
            )
        if not SPRITE_PATH.exists():
            raise FileNotFoundError(
                f"Sprite sheet not found: {SPRITE_PATH}\n"
                "Please save the cat sprite sheet to this location."
            )

        # Initialize scene
        self.scene = SceneManager(
            screen_width=SCREEN_WIDTH,
            screen_height=SCREEN_HEIGHT,
            base_scene_path=str(BACKGROUND_PATH)
        )
        setup_waypoints(self.scene)

        # Initialize sprite
        frame_w, frame_h = get_sprite_frame_dimensions(SPRITE_PATH)
        # Scale the cat to be appropriately sized for the scene
        # Adjust scale as needed based on how the cat looks
        self.cat = AnimatedSprite(
            sprite_sheet_path=str(SPRITE_PATH),
            frame_width=frame_w,
            frame_height=frame_h,
            scale=0.4  # Scale down if sprite is too large
        )

        # Start at the fireplace
        start_waypoint = next(
            (wp for wp in self.scene.waypoints if wp.name == "fireplace_front"),
            self.scene.waypoints[0]
        )
        self.cat.set_position(start_waypoint.position[0], start_waypoint.position[1])
        self.cat.set_facing(start_waypoint.facing_right)

        # Movement control
        self.current_waypoint = start_waypoint
        self.stay_timer = random.randint(
            start_waypoint.min_stay_ms,
            start_waypoint.max_stay_ms
        )
        self.waiting_at_waypoint = True

        # Running state
        self.running = True

    def handle_events(self):
        """Process pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    # Force movement on spacebar (for testing)
                    self.trigger_movement()
                elif event.key == pygame.K_f:
                    # Toggle fullscreen
                    self.fullscreen = not self.fullscreen
                    if self.fullscreen:
                        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                    else:
                        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    def trigger_movement(self):
        """Start moving to a new waypoint. (Disabled for now)"""
        # Walking disabled - just stay idle
        pass

    def update(self, dt_ms: int):
        """Update game state."""
        # Update scene (time-of-day checks)
        self.scene.update()

        # Update cat animation and movement
        self.cat.update(dt_ms)

        # Handle waypoint arrival and stay timing
        if self.waiting_at_waypoint:
            # Count down stay timer
            self.stay_timer -= dt_ms
            if self.stay_timer <= 0:
                self.trigger_movement()
        else:
            # Check if cat has arrived at destination
            if not self.cat.is_moving():
                # Arrived! Set facing and start stay timer
                self.cat.set_facing(self.current_waypoint.facing_right)
                self.stay_timer = random.randint(
                    self.current_waypoint.min_stay_ms,
                    self.current_waypoint.max_stay_ms
                )
                self.waiting_at_waypoint = True

    def draw(self):
        """Render the scene."""
        # Draw background with day/night effects
        self.scene.draw(self.screen)

        # Draw the cat
        self.cat.draw(self.screen)

        # Update display
        pygame.display.flip()

    def run(self):
        """Main game loop."""
        print(f"Starting Live Wallpaper at {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
        print("Controls:")
        print("  F     - Toggle fullscreen")
        print("  ESC   - Quit")

        while self.running:
            dt_ms = self.clock.tick(FPS)

            self.handle_events()
            self.update(dt_ms)
            self.draw()

        pygame.quit()
        print("Wallpaper closed.")


def main():
    """Entry point."""
    try:
        wallpaper = LiveWallpaper()
        wallpaper.run()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        pygame.quit()
        sys.exit(1)


if __name__ == "__main__":
    main()
