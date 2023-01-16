import typing as T
import numpy.typing as npt
import numpy as np

class EdgeOptAB:
    """
    Option class for edge connectivity.
    """
    # right point belongs to same set as left point
    add_set_transition_constraint = False
    # right point equal to left point
    add_equality_constraint = False
    # L2 norm on gripper movement
    add_each_block_movement_cost = False

    def __init__(
        self,
        add_set_transition_constraint: bool,
        add_equality_constraint: bool,
        add_each_block_movement_cost: bool,
    ):
        self.add_set_transition_constraint = add_set_transition_constraint
        self.add_equality_constraint = add_equality_constraint
        self.add_each_block_movement_cost = add_each_block_movement_cost

    @staticmethod
    def move_edge() -> "EdgeOptAB":
        return EdgeOptAB(True, False, True)

    @staticmethod
    def equality_edge() -> "EdgeOptAB":
        return EdgeOptAB(False, True, False)

    @staticmethod
    def target_edge() -> "EdgeOptAB":
        return EdgeOptAB(False, False, True)


class GCSforAutonomousBlocksOptions(GCSforBlocksOptions):

    @property
    def rel_iter(self, relation) -> int:
        return (relation+1) % self.number_of_relations

    @property
    def number_of_relations(self) -> int:
        return len(self.rels)

    @property
    def rels_len(self) -> int:
        """ Number of relations that define a free-space convex set"""
        return int((self.num_blocks - 1) * self.num_blocks / 2)

    @property
    def num_modes(self) -> int:
        """
        Number of modes. For the case with no pushing, we have 1 mode for free motion and a mode
        per block for when grasping that block.
        The case with pushing will have many more modes; not implemented.
        """
        return self.num_blocks

    def __init__(
        self,
        num_blocks: int,
        num_sides: int,
        block_radius: float = 1.0,
        max_rounded_paths: int = 40,
        use_convex_relaxation: bool = True,
        lb: T.List = None,
        ub: T.List = None,
        lbf: int = None,
        ubf: int = None,
    ):
        self.block_dim = 2
        self.num_sides = num_sides
        self.num_blocks = num_blocks
        self.block_radius = block_radius
        self.max_rounded_paths = max_rounded_paths
        self.use_convex_relaxation = use_convex_relaxation
        self.num_gcs_sets = -1 # uninitialized yet

        # other blocks are located in relation to the current block: left / right / above.
        # but since we have K sides, other blocks are in relation Dk to some block 
        self.rels = [i for i in range(self.num_sides)]

        if lb is not None:
            assert (
                len(lb) == self.block_dim
            ), "Dimension for lower bound constructor must be block_dim"
            self.lb = np.tile(lb, self.num_modes)
        elif lbf is not None:
            self.lb = np.ones(self.state_dim) * lbf
        else:
            self.lb = np.zeros(self.state_dim)

        if ub is not None:
            assert (
                len(ub) == self.block_dim
            ), "Dimension for upper bound constructor must be block_dim"
            self.ub = np.tile(ub, self.num_modes)
        elif ubf is not None:
            self.ub = np.ones(self.state_dim) * ubf
        else:
            self.ub = np.ones(self.state_dim) * 10.0
