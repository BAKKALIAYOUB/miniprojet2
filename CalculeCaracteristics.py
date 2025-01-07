import trimesh
from scipy.special import sph_harm
import numpy as np


def carac_fourier_3D(path: str) -> np.ndarray:
    mesh = trimesh.load(path)

    # Normalization (translation, échelle)
    mesh.apply_translation(-mesh.center_mass)  # Centrer à l'origine
    mesh.apply_scale(1 / mesh.scale)  # Normaliser l'échelle

    # Transformation de Fourier
    fourier_coeff = np.fft.fftn(mesh.vertices)  # Return complex numbers
    fourier_magnitudes = np.abs(fourier_coeff)  # Magnitude invariant a la rotation

    return fourier_magnitudes


def factorial(n):
    """Calculate factorial of a number using recursion."""
    if n == 0:
        return 1
    return n * factorial(n - 1)


def calc_3d_zernike_basis(n, l, m, r, theta, phi):
    """
    Calculate the 3D Zernike basis function.

    Args:
        n (int): Order of the Zernike polynomial.
        l (int): Degree of the Zernike polynomial.
        m (int): Order of the spherical harmonic.
        r (np.ndarray): Radial coordinates.
        theta (np.ndarray): Polar angles.
        phi (np.ndarray): Azimuthal angles.

    Returns:
        np.ndarray: Values of the 3D Zernike basis function.
    """
    # Radial polynomial
    R_nl = 0
    for k in range((n - l) // 2 + 1):
        coef = ((-1) ** k * factorial(n - k)) / (
                factorial(k) * factorial((n + l) // 2 - k) * factorial((n - l) // 2 - k))
        R_nl += coef * r ** (n - 2 * k)

    # Spherical harmonics
    Y_lm = sph_harm(m, l, phi, theta)

    return R_nl * Y_lm


def calculate_3d_zernike_moments(mesh, max_order=10):
    """
    Calculate 3D Zernike moments for a mesh.

    Args:
        mesh (trimesh.Trimesh): The 3D mesh object.
        max_order (int): Maximum order of Zernike moments to compute.

    Returns:
        dict: Dictionary containing Zernike moments indexed by (n, l, m).
    """
    # Get vertices and convert to spherical coordinates
    vertices = mesh.vertices

    # Normalize coordinates to fit in unit sphere
    center = np.mean(vertices, axis=0)
    vertices = vertices - center
    max_radius = np.max(np.linalg.norm(vertices, axis=1))
    vertices = vertices / max_radius

    # Convert to spherical coordinates
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / np.where(r != 0, r, 1))  # Avoid division by zero
    phi = np.arctan2(y, x)

    # Initialize moments dictionary
    moments = {}

    # Calculate moments for each order n
    for n in range(max_order + 1):
        for l in range(n + 1):
            # l must have same parity as n
            if (n - l) % 2 != 0:
                continue

            for m in range(-l, l + 1):
                # Calculate basis function values for all vertices
                basis_vals = calc_3d_zernike_basis(n, l, m, r, theta, phi)

                # Weight by vertex area (approximate as 1/num_vertices for simplicity)
                # For more accuracy, you could use actual vertex areas
                weight = 1.0 / len(vertices)

                # Calculate moment by numerical integration
                moment = np.sum(basis_vals * weight)
                moments[(n, l, m)] = np.abs(moment)

    return moments


def analyze_shape(mesh_path, max_order=10):
    """
    Load mesh and calculate its 3D Zernike moments.

    Args:
        mesh_path (str): Path to the mesh file.
        max_order (int): Maximum order of Zernike moments to compute.

    Returns:
        dict: Dictionary containing Zernike moments indexed by (n, l, m).
    """
    # Load mesh
    mesh = trimesh.load(mesh_path)

    # Ensure mesh is watertight and oriented
    if not mesh.is_watertight:
        print("Warning: Mesh is not watertight, results may be inaccurate")

    # Calculate moments
    moments = calculate_3d_zernike_moments(mesh, max_order)

    return moments


if __name__ == '__main__':
    # Example usage
    file = "3D Models/All Models/ampho.obj"
    moments = analyze_shape(file, max_order=10)
    print("3D Zernike Moments:")
    for key, value in moments.items():
        print(f"Order (n, l, m) = {key}: Moment = {value}")