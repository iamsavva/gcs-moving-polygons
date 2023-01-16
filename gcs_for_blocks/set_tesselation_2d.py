#!/usr/bin/env python3
# pyright: reportMissingImports=false
import typing as T

import numpy as np
import numpy.typing as npt

from pydrake.geometry.optimization import HPolyhedron

from .gcs_options import GCSforAutonomousBlocksOptions
from .util import (
    WARN,
    INFO,
    all_possible_combinations_of_items,
    timeit,
    ChebyshevCenter,
)

from tqdm import tqdm


class SetTesselation:
    def __init__(self, options: GCSforAutonomousBlocksOptions):
        self.opt = options
        self.sets_in_rels_representation = self.get_sets_in_rels_representation()

        self.index2relation = dict()  # T.Dict[int, (int,int)]
        self.make_index_to_block_relation()

        self.rels2set_dict = dict()  # T.Dict[str, HPolyhedron]
        self.generate_sets()

    def rels2set(self, rel_string:str):
        if self.opt.lazy_set_construction:
            return self.get_set_for_rels( self.string_to_rel(rel_string) )
        else:
            return self.rels2set_dict[rel_string]

    def rel_name(self, rels):
        assert type(rels) == list
        return "D_" + "_".join([str(x) for x in rels])

    def string_to_rel(self, rel_string):
        return [int(x) for x in rel_string[2:].split("_")]

    def get_sets_in_rels_representation(self):
        # all possible combinations of relations are teh possible sets
        return all_possible_combinations_of_items(self.opt.rels, self.opt.rels_len)

    def generate_sets(self):
        if not self.opt.lazy_set_construction:
            num_sets = len(self.sets_in_rels_representation)
            for i in tqdm(range(num_sets), "Set generation"):
                rels_rep = self.sets_in_rels_representation[i]
                # get set
                set_for_rels_rep = self.get_set_for_rels(rels_rep)
                # DO NOT reduce iequalities, some of these sets are empty
                # reducing inequalities is also extremely time consuming
                # set_for_rels_rep = set_for_rels_rep.ReduceInequalities()

                # check that it's non-empty
                solved, _, r = ChebyshevCenter(set_for_rels_rep)
                if solved and r >= 0.00001:
                    self.rels2set_dict[self.rel_name(rels_rep)] = set_for_rels_rep

    def get_set_for_rels(self, rels: T.List[int]) -> HPolyhedron:
        A, b = self.get_bounding_box_constraint()
        for relation_index, relation in enumerate(rels):
            i, j = self.index2relation[relation_index]
            A_relation, b_relation = self.get_constraints_for_relation(relation, i, j)
            A = np.vstack((A, A_relation))
            b = np.hstack((b, b_relation))
        return HPolyhedron(A, b)

    def get_constraints_for_relation(self, relation: int, i: int, j: int):
        return self.get_constraints_for_relation(relation, i, j)

    def get_constraints_for_relation(self, relation: int, i, j):
        """
        3 half planes that define that object j is in direction-relation relative to object i
        """
        r = self.opt.block_radius
        bd = self.opt.block_dim
        sd = self.opt.state_dim
        xi, yi = i * bd, i * bd + 1
        xj, yj = j * bd, j * bd + 1
        theta = 2*np.pi / self.opt.num_sides

        a0, a1, a2 = np.zeros(sd), np.zeros(sd), np.zeros(sd)
        k = relation
        angle = theta*(k+1)
        a0[xi], a0[yi], a0[xj], a0[yj] = np.sin(angle), -np.cos(angle), -np.sin(angle), np.cos(angle)
        angle = theta*(k)
        a1[xi], a1[yi], a1[xj], a1[yj] = -np.sin(angle), np.cos(angle), np.sin(angle), -np.cos(angle)
        angle = theta*(k+0.5)
        a2[xi], a2[yi], a2[xj], a2[yj] = np.cos(angle), np.sin(angle), -np.cos(angle), -np.sin(angle)
        A = np.vstack((a0, a1, a2))
        b = np.array([0, 0, -r])
        return A, b

    def get_bounding_box_constraint(self) -> T.Tuple[npt.NDArray, npt.NDArray]:
        A = np.vstack((np.eye(self.opt.state_dim), -np.eye(self.opt.state_dim)))
        b = np.hstack((self.opt.ub, -self.opt.lb))
        return A, b

    def make_index_to_block_relation(self):
        """
        Imagine the matrix of relations: 1-n against 1-n
        There are a total of n-1 relations
        0,1  0,2  0,3 .. 0,n-1
             1,2  1,3 .. 1,n-1
                      ..
                      ..
                         n-2,n-1
        index of the relation is sequential, goes left to right and down.
        """
        st = [0, 1]
        index = 0
        while index < self.opt.rels_len:
            self.index2relation[index] = (st[0], st[1])
            index += 1
            st[1] += 1
            if st[1] == self.opt.num_blocks:
                st[0] += 1
                st[1] = st[0] + 1
        assert st == [self.opt.num_blocks - 1, self.opt.num_blocks], "checking my math"

    def construct_rels_representation_from_point(self, point: npt.NDArray) -> T.List[int]:
        """
        Given a point, find a list of relations for it
        """
        rels_representation = []
        for index in range(self.opt.rels_len):
            i, j = self.index2relation[index]
            for relation in self.opt.rels:
                A, b = self.get_constraints_for_relation(relation, i, j)
                if np.all(A.dot(point) <= b):
                    rels_representation += [relation]
                    break
        # check yourself -- should have n*(n-1)/2 letters in the representation
        assert len(rels_representation) == self.opt.rels_len
        return self.rel_name(rels_representation)

    def get_1_step_neighbours(self, rels_string: str):
        """
        Get all 1 step neighbours
        1-step -- change of a single relation
        """
        rels = self.string_to_rel(rels_string)
        
        nbhd = []
        for i, rel in enumerate(rels):
            other_rel = rels
            nbh_rels_for_i = self.opt.rel_nbhd(rel)
            for nbh_rel in nbh_rels_for_i:
                other_rel[i] = nbh_rel
                if self.opt.lazy_set_construction or self.rel_name(other_rel) in self.rels2set_dict:
                    nbhd += [self.rel_name(other_rel)]
        return nbhd

    def get_useful_1_step_neighbours(self, rels_string: str, target_string: str):
        """
        Get 1-stop neighbours that are relevant given the target node
        1-step -- change in a single relation
        relevant to target -- if relation in relation is already same as in target, don't change it
        """
        rels = self.string_to_rel(rels_string)
        target = self.string_to_rel(target_string)
        assert len(rels) == self.opt.rels_len, "Wrong num of relations: " + rels
        assert len(target) == self.opt.rels_len, "Wrong num of relations in target: " + target

        nbhd = []
        for i, rel in enumerate(rels):
            # same -- don't do anything
            if rel == target[i]:
                continue

            up_dist = (target[i] - rels[i]) % self.opt.number_of_relations
            down_dist = (rels[i] - target[i]) % self.opt.number_of_relations
            if up_dist == down_dist:
                for let in self.opt.rel_nbhd(rel):
                    nbhd += [ self.rel_name(rels[:i] + [let] + rels[i + 1 :]) ]
            elif up_dist < down_dist:
                nbhd += [ self.rel_name(rels[:i] + [(rel+1)%self.opt.number_of_relations] + rels[i + 1 :]) ]
            else:
                nbhd += [ self.rel_name(rels[:i] + [(rel-1)%self.opt.number_of_relations] + rels[i + 1 :]) ]
        return nbhd
