import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Callable
import shutil


def lpsp(deficit, demand) -> list[float]:
    """ Loss of Power Supply Probability
    Used by: Bouzidi 2013, Ahmed and Demirci 2022
    """
    assert len(demand) == len(deficit)
    loss = [0] * len(demand)
    for i in range(1,len(demand)):
        loss[i] = sum(deficit[:i])/sum(demand[:i])
    return loss

def lpsp_total(deficit, demand) -> float:
    assert len(demand) == len(deficit)
    return sum(deficit)/sum(demand)


def clpsp(deficit, demand) -> list[float]:
    """Cumulative Loss of Power Supply Probability
    Just made it up ;)"""
    assert len(demand) == len(deficit)
    loss = [0] * len(demand)
    for i in range(len(demand)):
        cur_loss = deficit[i]/demand[i]
        if i > 0:
            cur_loss += loss[i-1]
        loss[i] = cur_loss
    return loss



def plot_loss_function(deficit, demand, loss_fn: Callable):
    loss = loss_fn(deficit=deficit, demand=demand)
    loss_fn_name = loss_fn.__name__
    
    plt.figure(figsize=(8, 6))  
    sns.set_context("talk")
    sns.set_style("whitegrid")  
    
    sns.lineplot(x=range(len(loss)), y=loss, linewidth=2, color="blue")
    plt.title(f"Loss Function: {loss_fn_name}", fontsize=16, fontweight="bold")
    plt.xlabel("Days", fontsize=14)
    plt.ylabel("Loss Value", fontsize=14)
    
    plt.text(0.05, 0.95, f"Deficit: {deficit}\nDemand: {demand}",
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white"))
    
    output_path = f"outputs/{loss_fn_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.clf()


# TODO: shortage days loss function
# add growth term for consecutive days of shortage

if __name__ == "__main__":
    demand = [20] * 14
    deficit = [0,0,0,20,20,20,0,0,0,20,0,0,0,0]
    output_path = Path("outputs")
    shutil.rmtree(output_path)
    output_path.mkdir()
    plot_loss_function(deficit=deficit, demand=demand, loss_fn=lpsp)
    plot_loss_function(deficit=deficit, demand=demand, loss_fn=clpsp)