"""
Demo data generation for testing motion estimation algorithms.
"""

import numpy as np


def generate_cell(
    center: tuple[float, float],
    radius: float,
    intensity: float,
    image_size: tuple[int, int] = (512, 512),
    seed: int = None,
) -> np.ndarray:
    """
    Generate a synthetic cell in a 2D image.

    Args:
        center: (y, x) coordinates of cell center
        radius: Cell radius
        intensity: Cell intensity
        image_size: Size of output image (height, width)

    Returns:
        2D array containing the cell
    """
    y, x = np.mgrid[: image_size[0], : image_size[1]]
    y_center, x_center = center

    # Create sharper Gaussian cell with higher intensity
    cell = np.exp(-((y - y_center) ** 2 + (x - x_center) ** 2) / (2 * (radius / 2) ** 2))
    cell = cell * intensity * 2  # Double the intensity

    return cell


def generate_cell_movement(
    num_frames: int = 5,
    image_size: tuple[int, int] = (512, 512),
    num_cells: int = 3,
    max_displacement: float = 50.0,
    radius: float = 5.0,
    displacement: tuple[float, float] | None = None,
    seed: int | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Generate a sequence of images with moving cells.

    Args:
        num_frames: Number of frames to generate
        image_size: Size of each frame (height, width)
        num_cells: Number of cells to generate
        max_displacement: Maximum cell displacement between frames
        radius: Radius of the cells
        displacement: Optional fixed displacement (dy, dx)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (frames, ground_truth_motion)
    """
    if seed is not None:
        np.random.seed(seed)

    frames = []
    motion_field = np.zeros((*image_size, 2), dtype=np.float32)

    # Generate initial cell positions
    centers = []
    for _ in range(num_cells):
        center = (
            np.random.uniform(image_size[0] * 0.2, image_size[0] * 0.8),
            np.random.uniform(image_size[1] * 0.2, image_size[1] * 0.8),
        )
        centers.append(center)

    # Generate frames
    for frame_idx in range(num_frames):
        frame = np.zeros(image_size, dtype=np.float32)

        # Add cells to frame
        for i, center in enumerate(centers):
            # Add some random movement (fresh draw per cell per frame unless
            # the caller fixed a displacement)
            if frame_idx > 0:
                if displacement is None:
                    step = np.random.uniform(-max_displacement, max_displacement, size=2)
                else:
                    step = displacement

                centers[i] = (center[0] + step[0], center[1] + step[1])
                # Update motion field
                y, x = np.mgrid[: image_size[0], : image_size[1]]
                dist = np.sqrt((y - center[0]) ** 2 + (x - center[1]) ** 2)
                mask = dist < 20  # Only update motion near cell
                motion_field[mask, 0] = step[0]
                motion_field[mask, 1] = step[1]

            # Generate cell with higher intensity and sharper profile
            cell = generate_cell(centers[i], radius=radius, intensity=2.0, image_size=image_size)
            frame += cell

        # Add minimal noise
        frame += np.random.normal(0, 0.05, frame.shape)

        # Normalize frame to [0, 1] range
        frame = (frame - frame.min()) / (frame.max() - frame.min())

        # Increase contrast
        frame = frame * 1.5
        frame = np.clip(frame, 0, 1)

        frames.append(frame)

    return frames, motion_field


def generate_motion_field(
    image_size: tuple[int, int] = (512, 512),
    max_displacement: float = 50.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate a synthetic motion field.

    Args:
        image_size: Size of the motion field (height, width)
        max_displacement: Maximum displacement in pixels
        seed: Random seed for reproducibility

    Returns:
        Motion field as a 3D array (height, width, 2)
    """
    if seed is not None:
        np.random.seed(seed)

    # Create coordinate grid
    y, x = np.mgrid[: image_size[0], : image_size[1]]

    # Center coordinates
    y_center = image_size[0] / 2
    x_center = image_size[1] / 2

    # Create radial motion field
    r = np.sqrt((y - y_center) ** 2 + (x - x_center) ** 2)
    theta = np.arctan2(y - y_center, x - x_center)

    # Radial displacement
    dr = max_displacement * np.exp(-r / 100)

    # Convert to Cartesian coordinates
    motion_field = np.zeros((*image_size, 2), dtype=np.float32)
    motion_field[..., 0] = dr * np.cos(theta)  # x component
    motion_field[..., 1] = dr * np.sin(theta)  # y component

    return motion_field
