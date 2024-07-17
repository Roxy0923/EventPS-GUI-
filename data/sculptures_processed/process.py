import os
import sys
import glob
import numpy as np
import pymeshlab

def process(file_input, file_output):
  ms = pymeshlab.MeshSet()
  ms.load_new_mesh(file_input)
  m = ms.current_mesh()
  ms.apply_coord_laplacian_smoothing(stepsmoothnum=1)
  ms.meshing_decimation_quadric_edge_collapse(targetfacenum=100000)
  ms.meshing_repair_non_manifold_vertices()
  ms.meshing_repair_non_manifold_edges(method="Remove Faces")
  ms.meshing_remove_connected_component_by_face_number(mincomponentsize=10000)
  ms.apply_normal_normalization_per_vertex()
  ms.compute_texcoord_parametrization_flat_plane_per_wedge(projectionplane="XY")
  # ms.generate_voronoi_atlas_parametrization()
  # ms.set_current_mesh(1)
  v_matrix = m.vertex_matrix()
  radius = np.linalg.norm(v_matrix, ord=2, axis=1).max()
  print("radius", radius)
  ms.compute_matrix_from_translation_rotation_scale(scalex=1/radius, scaley=1/radius, scalez=1/radius)
  ms.meshing_merge_close_vertices()
  ms.compute_normal_per_vertex(weightmode="By Area")
  ms.save_current_mesh(file_output)

def main():
  for i, filename in enumerate(sorted(glob.glob(os.path.join(sys.argv[1], "*.obj")))):
    filename = os.path.basename(filename)
    print("Processing", filename)
    process(os.path.join(sys.argv[1], filename), os.path.join(sys.argv[2], "%06d.obj" % i))

if __name__ == "__main__":
  main()
 
