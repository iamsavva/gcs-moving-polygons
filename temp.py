from gcs_for_blocks.set_tesselation_2d import SetTesselation
from gcs_for_blocks.gcs_options import GCSforAutonomousBlocksOptions
from gcs_for_blocks.gcs_auto_blocks import GCSAutonomousBlocks
from gcs_for_blocks.util import timeit, INFO, all_possible_combinations_of_items
# from gcs_for_blocks.hierarchical_gcs_ab import HierarchicalGraph, HierarchicalGCSAB

import numpy as np

from pydrake.geometry.optimization import (  # pylint: disable=import-error
    Point,
    HPolyhedron,
    ConvexSet,
)


from draw_2d import Draw2DSolution


if __name__ == "__main__":
    nb = 2
    ubf = 2.0
    start_point = Point( np.array([0,0, 2,2]))
    target_point = Point(np.array([2,2, 0,0]))

    # nb = 2
    # ubf = 2.0
    # start_point = Point( np.array([0.5,0.5, 1.5,1.5]))
    # target_point = Point(np.array([1.5,1.5, 0.5,0.5]))


    # nb = 3
    # ubf = 4.0
    # start_point = Point(np.array([1, 1, 1, 2, 1, 3]))
    # target_point = Point(np.array([3, 3, 3, 1, 3, 2]))

    # # # 5.31 5.27

    # nb = 4
    # ubf = 4.0
    # start_point = Point(np.array([1,1, 1,2, 1,3, 1,4]))
    # target_point = Point(np.array([3,4, 3,3, 3,2, 3,1]))

    options = GCSforAutonomousBlocksOptions(num_blocks = nb, num_sides = 50, ubf=ubf)
    options.use_convex_relaxation = False
    options.max_rounded_paths = 30
    options.problem_complexity = "collision-free-all-moving"
    # options.edge_gen = "binary_tree_down"  # binary_tree_down
    # options.rounding_seed = 1
    # options.custom_rounding_paths = 0

    x = timeit()
    gcs = GCSAutonomousBlocks(options)
    gcs.build_the_graph_simple(start_point, target_point)
    gcs.opt.num_gcs_sets = len(gcs.set_gen.rels2set)
    gcs.solve(show_graph=False, verbose=True)

    modes, vertices = gcs.get_solution_path()

    drawer = Draw2DSolution(nb, np.array([ubf,ubf]), modes, vertices, target_point.x(), fast = False, no_arm = True, draw_circles=True)
    drawer.draw_solution_no_arm()
