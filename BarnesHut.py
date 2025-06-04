import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

epsilon = 1e-8

class Node():
  def __init__(self, m, x:np.array, p:np.array, size0):
    self.m = m #mass
    self.x = x
    self.p = p
    self.child = None
    self.size = size0

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
    node_copy = Node(self.m, self.x.copy(), self.p.copy(), self.size)
    node_copy.child = self.child
    node_copy.size = self.size
    return node_copy

  def dist(self, other):
    return np.linalg.norm(self.x - other.x + epsilon)

  def force_by(self, other, G):
    if self.dist(other) < 1e-2:
      return 0
    else:
      return -G*self.m*other.m*(self.x - other.x)/(self.dist(other)**3)
  
  # if Node is None : "Empty Node"


class BarnesHut():
  def __init__(self, bodies,
               size0 = 1,
               min_node_size = 1e-4,
               theta = 0.7,
               G=1.e-5, 
               dt = 1.e-2):
    self.bodies = bodies
    self.size0 = size0
    self.min_node_size = min_node_size
    self.theta = theta
    self.G = G
    self.dt = dt

  def set_potential(self, mode, **kwargs):
    if mode == "None":
      return
    
    if mode == "Jacobi_Potential":
      self.Jacobi_Potential = Jacobi_Potential(kwargs["ext_sources"], self.size0, G = self.G, dx = 1.e-2)
      return
    
    else:
      raise NotImplementedError
    
  def load(self, body : Node, node, node_center):
    # if there is no node : set the given node(body) to leaf
    if node is None:
      node_updated = body
      return node_updated

    # else
    node_updated = node
    if node.size > self.min_node_size: # update until node.size > min_node_size
      # add the node itself into child node
      if node.child is None:
        node_updated = node.copy()
        node_updated.child = [None for _ in range(4)]
        idx, _ = node.to_child(node.x, node_center)
        node_updated.child[idx] = node
      idx, node_center_updated = body.to_child(body.x, node_center)
      node_updated.child[idx] = self.load(body, node_updated.child[idx], node_center_updated)
    return node_updated


  def force_on(self, body, node, ext_sources=None, external_force=None):
    ext_force = 0
    if external_force == "jacobi_potential":
      ext_force = self.Jacobi_Potential.ext_force(body)
    
    if node.child is None:
      return body.force_by(node, self.G) + ext_force

    if node.size < body.dist(node) * self.theta:
      return body.force_by(node, self.G) + ext_force

    return sum(self.force_on(body, c) for c in node.child if c is not None) + ext_force

  def simulation_step(self, ext_sources, ext_force):
    root = None
    for body in self.bodies:
      root = self.load(body, root, np.array([0, 0]))

    for body in self.bodies:
      force = self.force_on(body, root, ext_sources, ext_force)
      body.p = body.p + self.dt*force
      body.x = body.x + self.dt * body.p / body.m

  def draw(self, zoom_out=0.8, cumulative=False, ticks=False):
    halfwidth = self.size0/2 + zoom_out
    for body in self.bodies:
        x_temp = body.x[0]
        y_temp = body.x[1]
        if body.m > 100:
          plt.plot(x_temp, y_temp, color='red', marker='.', markersize=7)
        else:
          plt.plot(x_temp, y_temp, color='blue', marker='.', markersize=0.05)
    plt.xlim(-halfwidth, halfwidth)
    plt.ylim(-halfwidth, halfwidth)
    if ticks:
      plt.xticks([])
      plt.yticks([])
    if not cumulative:
      plt.figure(figsize=(10, 10))
      plt.show()

  def run(self, N_steps, zoom_out, ticks, cumulative):
    if cumulative:
      plt.figure(figsize=(10, 10))
      self.Jacobi_Potential.draw(zoom_out=zoom_out, ticks=ticks)
    for i in tqdm(range(N_steps)):
      self.simulation_step(ext_sources, "jacobi_potential")
      self.draw(zoom_out=zoom_out, cumulative=cumulative, ticks=ticks)
      if not cumulative:
        self.Jacobi_Potential.draw(zoom_out=zoom_out, ticks=ticks)


class Jacobi_Potential():
  def __init__(self, ext_sources, size0, G=1.e-5, dx=1.e-2):
    self.ext_sources = ext_sources
    self.size0 = size0
    self.G = G
    self.dx = dx

  def potential_at(self, X, Y):
    if len(self.ext_sources) != 2:
      raise NotImplementedError
    m1 = self.ext_sources[0][0]
    m2 = self.ext_sources[1][0] # m1 > m2
    mu = m2/(m1+m2)
    a = np.linalg.norm(self.ext_sources[0][1] - self.ext_sources[1][1])
    n = G*(m1+m2)/a**3
    r1 = np.hypot(X-self.ext_sources[0][1][0], Y-self.ext_sources[0][1][1]) + epsilon
    r2 = np.hypot(X-self.ext_sources[1][1][0], Y-self.ext_sources[1][1][1]) + epsilon
    com = (m1*self.ext_sources[0][1] + m2*self.ext_sources[1][1])/(m1+m2)

    return - self.G*m1/r1 - self.G*m2/r2 - 0.5*n*np.hypot(X-com[0], Y-com[1])**2

  def ext_force(self, body):
    dx = 1e-4
    pos = body.x
    U0 = self.potential_at(pos[0], pos[1])
    Ux = self.potential_at(pos[0]+self.dx, pos[1])
    Uy = self.potential_at(pos[0], pos[1]+self.dx)
    ax = -(Ux - U0)/self.dx
    ay = -(Uy - U0)/self.dx

    v_3d = np.append(body.p, 0)
    omega = np.sqrt(self.G*self.ext_sources[0][0] + epsilon)/(np.linalg.norm(self.ext_sources[0][1] - self.ext_sources[1][1])**1.5)*np.array([0, 0, 1])
    Corioli_F = 2*np.cross(v_3d, omega)
    return body.m*np.array([ax, ay]) + Corioli_F[:2]

  def draw(self, zoom_out, ticks=False):
    # WARNING : the parameters are tuned about G=1.e-5
    grid_num = int((self.size0+2*zoom_out)//self.dx)
    halfwidth = self.size0/2 + zoom_out
    X, Y = np.meshgrid(np.linspace(-halfwidth, halfwidth, grid_num), np.linspace(-halfwidth, halfwidth, grid_num))
    U = self.potential_at(X, Y)
    plt.imshow(U, alpha=0.2, vmin=-0.28, extent=(-halfwidth, halfwidth, -halfwidth, halfwidth))
    plt.colorbar()
    plt.contour(X, Y, U, levels=np.linspace(-0.23, -0.19, 20), alpha=0.4, extent=(-halfwidth, halfwidth, -halfwidth, halfwidth)) 
    #plt.colorbar()
    if ticks:
      plt.xticks([])
      plt.yticks([])


########## Main Code ##########
# parameter
min_node_size = 1e-4
size0 = 2
theta = 0.7
G = 1.e-5
dt = 1.e-2

N_bodies = 100
N_steps = 5000

# Fix Seed for Initialization
np.random.seed(1)

# Initial Conditions
ext_sources = [(1e+4, np.array([0.0, 0])), (3e+2, np.array([0.8, 0]))]

Masses = np.random.random(N_bodies)*1e-2
X0 = size0*np.random.random(N_bodies) - size0/2
Y0 = size0*np.random.random(N_bodies) - size0/2

X0 = 0.2*np.random.random(N_bodies) + 0.3
Y0 = 0.2*np.random.random(N_bodies) + 0.6

PX0 = (np.random.random(N_bodies) - 0.5)/1e+10
PY0 = (np.random.random(N_bodies) - 0.5)/1e+10

# Initialize
Bodies = [Node(mass, np.array([x0, y0]), np.array([px0, py0]), size0=1) for (x0, y0, px0, py0, mass) in zip(X0, Y0, PX0, PY0, Masses)]

# Main Loop

zoom_out = 0
ticks = False
cumulative = True

simulation = BarnesHut(Bodies, size0 = size0, min_node_size = min_node_size, theta = theta, G = G, dt = dt)
simulation.set_potential(mode = "Jacobi_Potential", ext_sources = ext_sources)
simulation.run(N_steps, zoom_out, ticks, cumulative)
plt.show()
