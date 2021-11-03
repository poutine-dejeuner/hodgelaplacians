from functools import lru_cache
from warnings import warn
from itertools import combinations

from scipy.linalg import expm
from scipy.sparse import dok_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigsh, norm
from sympy.combinatorics.permutations import Permutation

import numpy as np


class HodgeLaplacians:
    """Class for construction of Hodge and Bochner Laplacians from either collection of simplices
    or simplicial tree."""
    def __init__(self, simplices, oriented=True, maxdimension=2, mode='normal'):
        self.mode = mode
        self.oriented = oriented
        self.simplices = []

        if mode == 'normal':
            self.import_simplices(simplices=simplices)
            max_simplex_size = len(max(self.face_set, key=lambda el: len(el)))
            if maxdimension > max_simplex_size-1:
                warn(f"Maximal simplex in the collection has size {max_simplex_size}. \n maxdimension is set to {max_simplex_size-1}")
                self.maxdim = max_simplex_size - 1
            elif maxdimension < 0:
                raise ValueError(f"maxdimension should be a positive integer!")
            else:
                self.maxdim = maxdimension

        elif mode == 'gudhi':
            self.simplices = [tuple(a[0]) for a in simplices] # Removing filtration values
            self.face_set = self.simplices # Assume that the set of simplices is full
            max_simplex_size = len(max(self.face_set, key=lambda el: len(el)))
            if maxdimension > max_simplex_size-1:
                warn(f"Maximal simplex in the collection has size {max_simplex_size}. \n maxdimension is set to {max_simplex_size-1}")
                self.maxdim = max_simplex_size - 1
            elif maxdimension < 0:
                raise ValueError(f"maxdimension should be a positive integer!")
            else:
                self.maxdim = maxdimension
        else:
            raise ValueError(f"Désolé... Import modes different from 'normal' and 'gudhi' are not implemented yet...")


    def import_simplices(self, simplices=[]):
        if self.oriented == True:
            self.simplices = tuple(map(lambda simplex: tuple(simplex), simplices))
        elif self.oriented == False:
            self.simplices = tuple(map(lambda simplex: tuple(sorted(simplex)), simplices))
        self.face_set = self._faces(self.simplices)

    def n_faces(self, n):
        return tuple(filter(lambda face: len(face) == n+1, self.face_set))

    def _faces(self, simplices):
        """This function used to return a set but this had the annoying consequence of 
        defining the operators with a different ordering of the nodes thant the natural
        one when the nodes are named by integers. Returning a sorted list solves the problem """
        faceset = set()
        for simplex in simplices:
            numnodes = len(simplex)
            for r in range(numnodes, 0, -1):
                for face in combinations(simplex, r):
                        faceset.add(tuple(face))
        return tuple(sorted(faceset))

    def boundary_operator(self, i):
        source_simplices = self.n_faces(i)
        target_simplices = self.n_faces(i-1)

        if len(target_simplices) == 0:
            S = dok_matrix((1, len(source_simplices)), dtype=np.float64)
            S[0, 0:len(source_simplices)] = 1
        else:
            source_simplices_dict = {source_simplices[j]:
                                     j for j in range(len(source_simplices))}
            target_simplices_dict = {target_simplices[i]:
                                     i for i in range(len(target_simplices))}
            sorted_target_simplices = {tuple(sorted(simplex)):simplex for simplex in target_simplices}

            S = dok_matrix((len(target_simplices),
                            len(source_simplices)),
                           dtype=np.float64)
            for oriented_source in source_simplices:
                for a in range(len(oriented_source)):
                    oriented_face = oriented_source[:a]+oriented_source[(a+1):]  #constructs a simplex with the coordinate a missing
                    sorted_face = tuple(sorted(oriented_face))
                    oriented_target = sorted_target_simplices[sorted_face]
                    i = target_simplices_dict[oriented_target]
                    j = source_simplices_dict[oriented_source]
                    #S[i, j] = -1 if a % 2 == 1 else 1   # S[i, j] = (-1)**a
                    S[i, j] = (-1)**(orientation(oriented_face) + orientation(oriented_target) + a)
        return S

    @lru_cache(maxsize=32)
    def getBoundaryOperator(self,d):
        if d >= 0 and d <= self.maxdim:
            return self.boundary_operator(d).tocsr()
        else:
            raise ValueError(f"d should be not greater than {self.maxdim} (maximal allowed dimension for simplices)")

    @lru_cache(maxsize=32)
    def getHodgeLaplacian(self,d):
        if d == 0:
            L = self.getHodgeLaplacianUp(d)
        elif d < self.maxdim:
            L = self.getHodgeLaplacianUp(d) + self.getHodgeLaplacianDown(d)
        elif d == self.maxdim:
            L = self.getHodgeLaplacianDown(d)
        else:
            raise ValueError(f"d should be not greater than {self.maxdim} (maximal dimension simplices)")
        return L

    @lru_cache(maxsize=32)
    def getHodgeLaplacianUp(self,d):
        if 0 <= d < self.maxdim:
            B_next = self.getBoundaryOperator(d+1)
            Bt_next = B_next.transpose()
            L = B_next.dot(Bt_next)
        elif d == self.maxdim:
            raise ValueError(f"The upper Laplacian in dimension {self.maxdim} is trivial")
        else:
            raise ValueError(f"d should be not greater than {self.maxdim} (maximal dimension simplices)")
        return L 

    @lru_cache(maxsize=32)
    def getHodgeLaplacianDown(self,d):
        if d == 0:
            raise ValueError(f"The lower Laplacian in dimension 0 is trivial")
        elif 0 < d <= self.maxdim:
            B = self.getBoundaryOperator(d)
            Bt = B.transpose()
            L = Bt.dot(B)
        else:
            raise ValueError(f"d should be not greater than {self.maxdim} (maximal dimension simplices)")
        return L

    def getBochnerLaplacian(self,d):
        LB = self.getHodgeLaplacian(d)
        LB = LB - diags(LB.diagonal())
        ddd = norm(LB, 1, 1)
        LB = LB + diags(ddd) # Weitzenböck decomposition
        return LB

    @lru_cache(maxsize=32)
    def getHodgeHeatKernel(self,d,t):
        L = self.getHodgeLaplacian(d)
        L = L.multiply(t)
        return expm(L)

    @lru_cache(maxsize=32)
    def getBochnerHeatKernel(self,d,t):
        LB = self.getBochnerLaplacian(d)
        LB = L.multiply(t)
        return expm(LB)

    def diffuseChainHodge(self, d, chain, t):
        HK = self.getHodgeHeatKernel(d,t)
        diff_chain = HK.dot(chain)
        return diff_chain

    def diffuseChainBochner(self, d, chain, t):
        HK = self.getBochnerHeatKernel(d,t)
        diff_chain = HK.dot(chain)
        return diff_chain

    def getCombinatorialRicci(self,d):
        L = self.getHodgeLaplacian(d)
        LB = self.getBochnerLaplacian(d)
        Ricci = L - LB
        Ricci = Ricci.diagonal()
        return Ricci

    def getHodgeSpectrum(self, d, k=1, around_point=0.01, dense=False):
        """Obtain k eigenvalues and eigenvectors of d-Hodge Laplacian.
        Eigenvalues and eigenvectors are computed sufficently fast 
        using Shift-Invert mode of the ARPACK algorithm in SciPy.
        More info: https://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html"""
        L = self.getHodgeLaplacian(d)
        if dense == False:
            vals, vecs = eigsh(L, k=k, sigma=around_point, which='LM')
        elif dense == True:
            L = L.toarray()
            vals, vecs = np.linalg.eigh(L)
        return vals, vecs
  
    def getBochnerSpectrum(self, d, k=1, around_point=0.01, dense=False):
        """Obtain k eigenvalues and eigenvectors of d-Hodge Laplacian.
        Eigenvalues and eigenvectors are computed sufficently fast 
        using Shift-Invert mode of the ARPACK algorithm in SciPy.
        More info: https://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html"""
        LB = self.getBochnerLaplacian(d)
        if dense == False:
            vals, vecs = eigsh(LB, k=k, sigma=around_point, which='LM')
        elif dense == True:
            LB = LB.toarray()
            vals, vecs = np.linalg.eigh(LB)
        return vals, vecs

    def randomWalkDistribution(self, d, chain_distribution, time):
        """To be implemented based on the work
        S. Mukherjee￼ and J. Steenbergen,
        Random walks on simplicial complexes and harmonics
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5324709/"""
        pass

    def randomWalkChain(self, d, chain, time):
        """To be implemented based on the work
        S. Mukherjee￼ and J. Steenbergen,
        Random walks on simplicial complexes and harmonics
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5324709/"""
        pass

class WeightedHodgeLaplacians( HodgeLaplacians ):
    def __init__(self, weighted_simplices, oriented=True, maxdimension=2):
        '''
        weighted_simplices: dict with simplex keys and weight values
        '''
        self.oriented = oriented
        self.weighted_simplices = weighted_simplices
        self.simplices = tuple(self.weighted_simplices.keys())
        no_vertices = []
        for simplex in self.simplices:
            if type(simplex) != tuple:
                pass
            else:
                no_vertices.append(simplex)
        if no_vertices == []:
            self.maxdim = 0
        else:
            max_simplex_size = max(len(simplex) for simplex in no_vertices)
        self.maxdim = max_simplex_size - 1

    def n_weighted_faces(self, n):
        return {simplex:weight for simplex,weight in self.weighted_simplices.items() if len(simplex)==n+1}

    def n_faces(self, n):
        return tuple(simplex for simplex,weight in self.weighted_simplices.items() if len(simplex)==n+1)

    def n_weights(self, n):
        simplices = self.n_weighted_faces(n)
        if len(simplices) == 0:
            W = 0
        else:
            W = np.array(list(simplices.values()))
        return W

    @lru_cache(maxsize=32)
    def getHodgeLaplacianUp(self,d):
        if 0 <= d < self.maxdim:
            B_dplus1 = self.getBoundaryOperator(d+1)
            Bt_dplus1 = B_dplus1.transpose()
            W_d = diags(self.n_weights(d))
            W_dplus1inv = diags(self.n_weights(d-1)**(-1))
            L = B_dplus1 @ W_dplus1inv @ Bt_dplus1 @ W_d
        elif d == self.maxdim:
            raise ValueError(f"The upper Laplacian in dimension {self.maxdim} is trivial")
        else:
            raise ValueError(f"d should be not greater than {self.maxdim} (maximal dimension simplices)")
        return L 

    @lru_cache(maxsize=32)
    def getHodgeLaplacianDown(self,d):
        if d == 0:
            raise ValueError(f"The lower Laplacian in dimension 0 is trivial")
        elif 0 < d <= self.maxdim:
            B_d = self.getBoundaryOperator(d)
            Bt_d = B_d.transpose()
            W_dminus1 = diags(self.n_weights(d-1))
            W_dinv = diags(self.n_weights(d)**(-1))
            L = W_dinv @ Bt_d @ W_dminus1 @ B_d
        else:
            raise ValueError(f"d should be not greater than {self.maxdim} (maximal dimension simplices)")
        return L

def orientation(ordered_nodes):
    sorted_indices = [ordered_nodes.index(i) for i in sorted(ordered_nodes)]
    orientation = Permutation(sorted_indices).parity()
    return orientation