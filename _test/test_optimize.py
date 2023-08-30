import sys, os
from os.path import dirname
import numpy as np
import pickle
import pytest
import matplotlib.pyplot as plt

sys.path.append(dirname(__file__) + "/../")
root_dir = os.path.abspath(os.path.dirname(__file__) + "/../")
print("appended:", sys.path[-1])

from poly_matrix import PolyMatrix, sorted_dict
from optimize import solve_sdp_mosek, solve_low_rank_sdp

def test_mosek_solve():
    # Test mosek solve on a simple problem
    # Load data from file
    with open(os.path.join(root_dir,"_test","test_prob_1.pkl"),'rb') as file:
        data = pickle.load(file)
    
    # Run mosek solver
    X, cost = solve_sdp_mosek(Q=data['Q'],
                              Constraints=data['Constraints'])
    # Get singular values
    u,s,v = np.linalg.svd(X)
    print(f"SVR:  {s[0]/s[1]}")
    print(f"Cost: {cost} ")
    
    with open(os.path.join(root_dir,"_test","test_prob_1.pkl"),'wb') as file:
        data['X'] = X
        data['cost']=cost
        pickle.dump(data,file) 
    
def test_low_rank_solve(rank=2):
    # Test mosek solve on a simple problem
    # Load data from file
    with open(os.path.join(root_dir,"_test","test_prob_1.pkl"),'rb') as file:
        data = pickle.load(file)
    
    # Feasible initial condition
    y_0 = [1.,1.,0.,0.,0.,1.,0.,0.,0.,1.,0.,0.,0.]
    Y_0 = y_0 * rank
    # Init with solution
    Y, X, cost = solve_low_rank_sdp(Q=data['Q'],
                            Constraints=data['Constraints'],
                            rank=rank,
                            Y_0=Y_0)
    # Check solution rank
    u,s,v = np.linalg.svd(X)
    print(f"SVR:  {s[0]/s[1]}")
    print(f"Cost: {cost} ")
    print(f"MOSEK Cost: {data['cost']}")
    plt.semilogy(s)
    plt.title(f"Rank Restriction: {rank}")
    plt.show()    
    
def test_low_rank_solve_redun(rank=2):
    # Test mosek solve on a simple problem
    # Load data from file
    with open(os.path.join(root_dir,"_test","test_prob_2.pkl"),'rb') as file:
        data = pickle.load(file)
    
    # Feasible initial condition
    y_0 = [1.,1.,0.,0.,0.,1.,0.,0.,0.,1.,0.,0.,0.]
    Y_0 = y_0 * rank
    # Init with solution
    Y, X, cost = solve_low_rank_sdp(Q=data['Q'],
                            Constraints=data['Constraints'],
                            rank=rank,
                            Y_0=Y_0)
    # Check solution rank
    u,s,v = np.linalg.svd(X)
    print(f"SVR:  {s[0]/s[1]}")
    print(f"Cost: {cost} ")
    print(f"MOSEK Cost: {data['cost']}")
    plt.semilogy(s)
    plt.title(f"Rank Restriction: {rank}")
    plt.show()    
    


if __name__ == "__main__":
    # test_mosek_solve()
    test_low_rank_solve()
    
    