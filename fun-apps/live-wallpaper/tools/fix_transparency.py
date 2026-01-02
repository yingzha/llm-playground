"""
Fix checkered transparency pattern in sprite sheets.
Converts the checkered "fake transparency" to real PNG transparency.
"""

from PIL import Image
import sys
from pathlib import Path


def fix_checkered_transparency(input_path: str, output_path: str = None):
    """
    Remove checkered background pattern and replace with true transparency.
    """
    img = Image.open(input_path).convert("RGBA")
    pixels = img.load()
    width, height = img.size

    # Sample a larger area from the corners to detect checker colors
    color_samples = []
    sample_regions = [
        (0, 0, 50, 50),  # top-left
        (width-50, 0, width, 50),  # top-right
    ]

    for x1, y1, x2, y2 in sample_regions:
        for y in range(y1, min(y2, height)):
            for x in range(x1, min(x2, width)):
                r, g, b, a = pixels[x, y]
                color_samples.append((r, g, b))

    # Find the two most common colors (checker pattern)
    from collections import Counter
    color_counts = Counter(color_samples)
    top_colors = color_counts.most_common(10)

    print(f"Top colors detected: {top_colors[:5]}")

    # The checker colors should be the two most common
    if len(top_colors) >= 2:
        checker_colors = [top_colors[0][0], top_colors[1][0]]
        print(f"Checker colors identified: {checker_colors}")
    else:
        print("Could not identify checker colors")
        return

    # Process all pixels - remove any gray-ish colors in the checker range
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]

            # Check if pixel is gray (r ≈ g ≈ b) and in the checker color range
            is_gray = abs(r - g) < 15 and abs(g - b) < 15 and abs(r - b) < 15
            in_checker_range = 80 < r < 200 and 80 < g < 200 and 80 < b < 200

            if is_gray and in_checker_range:
                pixels[x, y] = (r, g, b, 0)  # Make transparent

    # Save result
    if output_path is None:
        output_path = input_path.replace(".png", "_fixed.png")

    img.save(output_path, "PNG")
    print(f"Saved fixed sprite sheet to: {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_transparency.py <sprite_sheet.png> [output.png]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    fix_checkered_transparency(input_file, output_file)
