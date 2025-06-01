import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Node():
  def __init__(self, m, x:np.array, p:np.array):
    self.m = m #mass
    self.x = x
    self.p = p
    self.child = None
    self.size = Size0

  def to_child(self, pos:np.array, node_center):
    """
    Make the Node to child node (reduce the size), and return the index
    * Note : indexing
    0|1
    -+-
    2|3
    """
    idx = 0
    self.size /= 2
    node_center_updated = node_center.copy()
    for dim in range(2):
      if pos[dim] > node_center[dim]:
        idx += 2**dim
        node_center_updated[dim] += self.size/2
      else:
        node_center_updated[dim] -= self.size/2
    return idx, node_center_updated

  def copy(self):
    node_copy = Node(self.m, self.x.copy(), self.p.copy())
    node_copy.child = self.child
    node_copy.size = self.size
    return node_copy

  def dist(self, other):
    return np.linalg.norm(other.x - self.x)
  
  def force_by(self, other, G):
    if self.dist(other) < 1e-2:
      return 0
    else:
      return G*self.m*other.m*(other.x - self.x)/(self.dist(other)**3)

# if Node is None : "Empty Node"


def load(body : Node, node, node_center):
  # if there is no node : set the given node(body) to leaf
  if node is None:
    node_updated = body
    return node_updated

  # else
  node_updated = node
  if node.size > min_node_size: # update until node.size > min_node_size
    # add the node itself into child node
    if node.child is None:
      node_updated = node.copy()
      node_updated.child = [None for _ in range(4)]
      idx, _ = node.to_child(node.x, node_center)
      node_updated.child[idx] = node
    idx, node_center_updated = body.to_child(body.x, node_center)
    node_updated.child[idx] = load(body, node_updated.child[idx], node_center_updated)
  return node_updated

def force_on(body, node, theta, G):
  if node.child is None:
    return body.force_by(node, G)

  if node.size < body.dist(node) * theta:
    return body.force_by(node, G)

  return sum(force_on(body, c, theta, G) for c in node.child if c is not None)

def simulation_step(bodies, theta, G, dt):
  root = None
  for body in bodies:
    root = load(body, root, np.array([0, 0]))
  
  for body in bodies:
    force = force_on(body, root, theta, G)
    body.p = body.p + dt*force
    body.x = body.x + dt * body.p / body.m

def draw(bodies, zoom_out=0.8, cumulative=False, ticks=False):
    for body in bodies:
        x_temp = body.x[0]
        y_temp = body.x[1]
        if body.m > 100:
          plt.plot(x_temp, y_temp, color='red', marker='.', markersize=7)
        else:
          plt.plot(x_temp, y_temp, color='blue', marker='.', markersize=0.1)
    plt.xlim(-Size0/2-zoom_out, Size0/2+zoom_out)
    plt.ylim(-Size0/2-zoom_out, Size0/2+zoom_out)
    if ticks:
      plt.xticks([])
      plt.yticks([])
    if not cumulative:
      plt.show()
      

########## Main Code ##########
# parameter
min_node_size = 1e-4
Size0=1
Theta = 0.7
G = 1.e-5
dt = 1.e-2
N_bodies = 100
N_steps = 1000

# Fix Seed for Initialization
np.random.seed(123)

# Initial Conditions
Masses = np.random.random(N_bodies)*1
X0 = np.random.random(N_bodies) - 0.5
Y0 = np.random.random(N_bodies) - 0.5
PX0 = np.random.random(N_bodies) - 0.5
PY0 = np.random.random(N_bodies) - 0.5

Masses[0] = 1e+4
X0[0] = 0
Y0[0] = 0
PX0[0] = 0
PY0[0] = 0
PX0 = PX0/1e-0
PY0 = PY0/1e-0

# Initialize
Bodies = [Node(mass, np.array([x0, y0]), np.array([px0, py0])) for (x0, y0, px0, py0, mass) in zip(X0, Y0, PX0, PY0, Masses)]

# Main Loop
def Loop_BH(n):
    for i in tqdm(range(n)):
        simulation_step(Bodies, Theta, G, dt)
        draw(Bodies, zoom_out=0.8, cumulative=True)

Loop_BH(N_steps)
plt.show()
