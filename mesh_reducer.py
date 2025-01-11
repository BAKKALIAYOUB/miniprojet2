import trimesh


def reduce_mesh(input_file, reduction_factor=0.5):
    """
    Reads an .obj file, reduces the mesh complexity, and returns the simplified mesh.

    Parameters:
    - input_file (str): Path to the input .obj file.
    - reduction_factor (float): Fraction of faces to retain (0.0 to 1.0).

    Returns:
    - simplified_mesh (trimesh.Trimesh): The simplified mesh object.
    """
    # Load the mesh from the .obj file
    mesh = trimesh.load(input_file)

    # Ensure the mesh is watertight (optional, depending on your use case)
    if not mesh.is_watertight:
        print("Mesh is not watertight. Attempting to fix...")
        mesh.fill_holes()

    # Simplify the mesh
    simplified_mesh = mesh.simplify_quadric_decimation(reduction_factor)


    # Return the simplified mesh instead of saving it
    return simplified_mesh


if __name__ == "__main__":
    file = "All Models/ampho.obj"

    # Call the function and get the reduced mesh
    simplified_mesh = reduce_mesh(file, reduction_factor=0.5)

    print()
