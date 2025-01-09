import trimesh

def reduce_mesh(input_file, output_file, reduction_factor=0.5):
    """
    Reads an .obj file, reduces the mesh complexity, and saves the simplified mesh.

    Parameters:
    - input_file (str): Path to the input .obj file.
    - output_file (str): Path to save the reduced .obj file.
    - reduction_factor (float): Fraction of faces to retain (0.0 to 1.0).
    """
    # Load the mesh from the .obj file
    mesh = trimesh.load(input_file)

    # Ensure the mesh is watertight (optional, depending on your use case)
    if not mesh.is_watertight:
        print("Mesh is not watertight. Attempting to fix...")
        mesh.fill_holes()

    # Simplify the mesh
    simplified_mesh = mesh.simplify_quadric_decimation(reduction_factor)

    # Export the simplified mesh to a new .obj file
    simplified_mesh.export(output_file)
    print(f"Simplified mesh saved to {output_file}")


if __name__ == "__main__":
    file = "3D Models/All Models/ampho.obj"
    output = "output.obj"

    reduce_mesh(file, output, reduction_factor=0.5)
