# dqn_poacher.py

import collections
from types import SimpleNamespace

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from agents.base import ControlPanel
from simulation.reserve import bounds_from_graph
from simulation.weather import WEATHER_STATES

VISIBLE_FACTS = ["pos","Viewshed", "dist_to_water", "centrality"]
HIDDEN_FACTS = ["Type", "dist_to_civ", "dist_to_station", "dist_to_sighting"]
VISIBLE_TYPES = {"mountain", "water", "poacherh"}
EXPERIENCE_FACTS = {"times_caught": 0}

NODE_TYPES = ["poacherh", "rangerh", "civilization", "mountain", "camera", "water"]
DIST_SCALE_M = 100_000
EDGE_SCALE_M = 50_000
HIDDEN_DISTANCES = ["dist_to_civ", "dist_to_station", "dist_to_sighting"]

# Reasoning-bank lessons weigh lesson concerns between 0-1
# Every DQN has these 6 inputs which stay 0 unless they have a reasoning bank, to keep one standard network size
LESSON_CONCERNS = ["avoid_villages", "avoid_station_area", "avoid_high_centrality",
                   "prefer_low_viewshed", "avoid_capture_sites", "avoid_threat_cells"]

class DQN(nn.Module):
    def __init__(self, input_size, output_size=1):
        """
        Initialize feedforward NN
        State propagation to output
        Args:
            input_size (int): situation numbers + one move's numbers
            output_size (int): 1 = the score Q(situation, move) for that one move
        """
        super(DQN, self).__init__()
        # Starting with 3-layer 128 neuron network
        self.layer1 = nn.Linear(input_size, 256) # Input layer to first hidden layer
        self.layer2 = nn.Linear(256, 256)        # Hidden layer
        self.layer3 = nn.Linear(256, 64)         # Hidden layer bottleneck (generalization)
        self.layer4 = nn.Linear(64, output_size) # Final layer to output Q-values

    def forward(self, x):
        """
        Defining how an input state 'x' flows through the network
        """
        # Apply the ReLU activation function after the first two layers
        # Rectified linear unit, piecewise linear function outputs direct linear input if positive
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))

        # Output raw Q representing probability of action
        return self.layer4(x)

class DQNAgent(ControlPanel):
    """Deep Q-network poacher. Operates in simulation through ControlPanel hooks."""

    def __init__(self, graph, rng, learning_rate=0.001,
                 discount_factor=0.99, epsilon_decay=0.995, min_epsilon=0.01,
                 update_target_every=10, batch_size=64, reward_scale=1000,
                 use_knowledge=True, use_threat=True, use_road_distance=False, eval_epsilon=0.0):
        super().__init__(graph, rng)
        # Ablation switches. Toggled off inputs are zeroed to maintain network size cross arms
        self.use_knowledge = use_knowledge      # encode_node's 17 input bitmask about each neighbor
        self.use_threat = use_threat            # threat grid in the situation + threat at the neighbor's cell
        self.use_road_distance = use_road_distance  # knows the road network, aka edge lengths from A* terrain traversals
        # Shortest route (terrain_dist) stored between every pair of nodes
        self.road_dist = dict(nx.all_pairs_dijkstra_path_length(graph, weight="terrain_dist"))
        # Exploration during evaluation (0 = fully greedy)
        self.eval_epsilon = eval_epsilon
        # training = False (evaluation): greedy moves (epsilon 0), no memories stored, no learning
        self.training = True
        # The world's rewards (ex: -5000 caught) are divided by this before learning, so targets stay small
        self.reward_scale = reward_scale
        # Seed torch from this brain's own stream, so network weights remain reproducible
        torch.manual_seed(rng.randrange(2**31))

        self.grid_resolution = 10
        self.threat_grid = np.zeros((self.grid_resolution, self.grid_resolution))
        self.known = self._initial_knowledge()
        self.blocked_today = set() # sorted (u, v) edges agent ran into, blocks re-roll at midnight
        self.replay_memory = collections.deque(maxlen = 10000)
        # Most neighbors any node has: learn() pads every next-node candidate list to this length
        self.max_degree = max(deg for _, deg in graph.degree())

        self.discount_factor = discount_factor
        self.batch_size = batch_size

        # Exploration and episode stats
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.update_target_every = update_target_every
        self.episodes_done = 0

        # learn() adds to these and end_episode() averages them into last_trip_stats
        self.trip_losses = []
        self.trip_qs = []
        self.last_trip_stats: dict[str, float | int | None] = {}

        # The move in progress: [its input row (situation + move), reward collected since]
        self.pending = None
        # Last goal and situation seen in next_move, allows forced move from events to also be recorded
        self.last_goal = None
        self.last_view = None
        # No immediate return: the node just left, and the goal at the time (the rule resets whenever goal changes)
        self.came_from = None
        self.came_from_goal = None

        # The same bounds the world uses worked out from the graph
        self.map_bounds = bounds_from_graph(graph)

        # Largest stored centrality, used to scale to 0-1 (or 1 avoids dividing by zero)
        self.max_centrality = max(attrs.get("centrality", 0) for _, attrs in self.graph.nodes(data=True)) or 1.0

        # Input size worked out from the encoders to keeps up with new features/changes
        u, v = next(iter(graph.edges))
        sample_view = SimpleNamespace(weather=WEATHER_STATES[0], night=False, map_bounds=self.map_bounds)
        self.input_size = len(self.get_state_vector(u, v, sample_view)) + len(self.encode_move(u, v, v))

        self.main_network = DQN(self.input_size)
        self.target_network = DQN(self.input_size)
        self.target_network.load_state_dict(self.main_network.state_dict()) #Clone weights after pass
        self.optimizer = optim.Adam(self.main_network.parameters(), lr = learning_rate)
        # Huber loss: squared for small errors, linear for big ones to hedge against skewed outliers
        self.loss_function = nn.SmoothL1Loss()

    # ControlPanel hooks

    def start_episode(self, node, goal, situation=None):
        self.pending = None
        self.came_from = None

    def next_move(self, node, goal, situation):
        """Choose the next neighbor. Also closes off the previous move as one stored experience."""
        view = SimpleNamespace(weather=situation["weather"], night=situation["night"], map_bounds=self.map_bounds)
        self.last_goal, self.last_view = goal, view
        neighbors, rows = self.candidate_moves(node, goal, view)

        # One memory per move, prev move led us here
        self._close_pending(rows)

        i = self.choose_action(rows, self.epsilon if self.training else self.eval_epsilon, self.move_bonus(node, neighbors))
        self.pending = [rows[i], 0.0]
        self.came_from, self.came_from_goal = node, goal
        # return new options
        return neighbors[i]

    def feedback(self, reward, done):
        if self.pending is None:
            return
        self.pending[1] += reward / self.reward_scale
        if done:
            row, collected = self.pending
            if self.training:
                # no next options, the trip is over
                self.remember(row, collected, None, True)
                self.learn(self.batch_size)
            self.pending = None

    def _close_pending(self, next_rows):
        """Store the move in progress, now that we know the options it led to."""
        if self.pending is not None and self.training:
            row, collected = self.pending
            self.remember(row, collected, next_rows, False)
            self.learn(self.batch_size)
            self.pending = None

    def _forced_move(self, node, to):
        """An event occurnnce sent us to a node without asking, likely got lost. Record the move that happened."""
        
        if self.last_goal is None:
            return
        
        neighbors, rows = self.candidate_moves(node, self.last_goal, self.last_view)
        self._close_pending(rows)
        state = self.get_state_vector(node, self.last_goal, self.last_view)
        move = torch.tensor(self.encode_move(node, to, self.last_goal), dtype=torch.float32)
        self.pending = [torch.cat([state, move]), 0.0]
        self.came_from, self.came_from_goal = node, self.last_goal

    def end_episode(self, outcome):
        
        self.last_trip_stats.clear()
        
        if not self.training:
            return

        if self.trip_losses:
            self.last_trip_stats = {"mean_loss": sum(self.trip_losses) / len(self.trip_losses), "mean_q": sum(self.trip_qs) / len(self.trip_qs)}
        else:
            self.last_trip_stats = {"mean_loss": None, "mean_q": None}
        self.trip_losses.clear()
        self.trip_qs.clear()

        self.episodes_done += 1
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        if self.episodes_done % self.update_target_every == 0:
            self.update_target_network()

    def end_day(self):
       
        self.threat_grid *= 0.90
        self.blocked_today.clear()

    # Agent knowledge 

    def normalize(self, pos, bounds):
        "Normalize x y position to 0-1 range"
        min_x, min_y, max_x, max_y = bounds
        return np.array([(pos[0]-min_x)/(max_x-min_x), (pos[1]-min_y)/(max_y-min_y)])

    def _initial_knowledge(self):
        """
        Initialize the agent's knowledge about the environment.
        """

        known = {}
        for node, attrs in self.graph.nodes(data=True):
            facts = {fact: attrs.get(fact, None) for fact in VISIBLE_FACTS}
            hidden_facts = {fact: None for fact in HIDDEN_FACTS}
            known[node] = {**facts, **hidden_facts, **EXPERIENCE_FACTS}

            if attrs["Type"] in VISIBLE_TYPES:
                known[node]["Type"] = attrs["Type"]

        return known

    def observe(self, event, reserve=None):
        """
        Update the agent's knowledge based on an observed event.
        """
        kind = event["kind"]
        node = event["node"]

        if kind == "arrived":
            node_type = self.graph.nodes[node]["Type"]
            self.known[node]["Type"] = node_type
            # Standing at a civ point means the nearest one is 0 m away
            if node_type == "civilization":
                self.known[node]["dist_to_civ"] = 0.0
            elif node_type == "rangerh":
                self.known[node]["dist_to_station"] = 0.0
        elif kind == "captured":
            self.known[node]["times_caught"] += 1
            pos = event["pos"]
            grid_r, grid_c = self._reserve_to_grid(pos, self.map_bounds)
            self.threat_grid[grid_r, grid_c] = 1.0
        elif kind == "blocked":
            self.blocked_today.add(tuple(sorted(event["edge"])))
        elif kind == "forced_move":
            self._forced_move(node, event["to"])
        elif kind == "returned":
            pass # next_move is asked right after at this node, and closes the blocked move's memory there
        else:
            raise ValueError(f"Unknown event kind: {kind!r}")

    def _reserve_to_grid(self, pos, map_bounds):
        x, y = self.normalize(pos, map_bounds)
        row = min(int(y * self.grid_resolution), self.grid_resolution - 1)
        col = min(int(x * self.grid_resolution), self.grid_resolution - 1)
        return row, col


    def get_state_vector(self, current_node, goal_node, reserve):
        """
        Define NN inputs (positional states, weather, memory, etc..)
        """

        # Positional inputs
        current_pos = np.array(self.graph.nodes[current_node]['pos'])
        goal_pos = np.array(self.graph.nodes[goal_node]['pos'])

        # Normalize position parameters against map bounds
        norm_current_pos = self.normalize(current_pos, reserve.map_bounds)
        norm_goal_pos = self.normalize(goal_pos, reserve.map_bounds)

        #Environment input
        weather_vector = [1.0 if reserve.weather == w else 0.0 for w in WEATHER_STATES]
        night_flag = [1.0 if reserve.night else 0.0]

        # Thread grid & memory
        threat = self.threat_grid if self.use_threat else np.zeros_like(self.threat_grid)
        state_array = threat.flatten()

        state_array = np.concatenate([
            norm_current_pos,
            norm_goal_pos,
            weather_vector,
            night_flag,
            state_array,

        ])
        return torch.tensor(state_array, dtype = torch.float32)

    # Acting and learning

    def candidate_moves(self, node, goal, view):
        """Viable neighborm moves, and one input row per neighbor: situation + that move.

        Edges this poacher found blocked today are skipped, and so is the node just left while the goal
        is unchanged (no immediate return). If that rules out everything, all neighbors are allowed.
        Returns (neighbors, rows) with rows of shape (len(neighbors), input_size).
        """
        neighbors = list(self.graph.neighbors(node))
        back = self.came_from if goal == self.came_from_goal else None
        open_ones = [n for n in neighbors if n != back and tuple(sorted((node, n))) not in self.blocked_today]
        neighbors = open_ones or neighbors

        state = self.get_state_vector(node, goal, view)
        moves = torch.tensor([self.encode_move(node, n, goal) for n in neighbors], dtype=torch.float32)
        # Same situation on every row, then that row's move: (k, situation) + (k, move) -> (k, input_size)
        rows = torch.cat([state.expand(len(neighbors), -1), moves], dim=1)
        return neighbors, rows

    def choose_action(self, rows, epsilon, bonus=None):
        """Epsilon-greedy over the candidate rows. Returns the index of the chosen row.

        `bonus` (one number per row, or None) is added to the scores only when choosing, learning never sees it.
        """
        if self.rng.random() < epsilon:
            return self.rng.randrange(len(rows))
        with torch.no_grad():
            scores = self.main_network(rows).squeeze(1)     # (k, 1) -> (k,): one score per neighbor
        if bonus is not None:
            scores = scores + bonus
        return int(scores.argmax())

    def move_bonus(self, node, neighbors) -> torch.Tensor | None:
        """Extra score per candidate move, used when choosing. Plain DQN agents have none."""
        return None

    # Command to store a log an experience in replay memory
    def remember(self, row, reward, next_rows, done):
        """row = the move taken (situation + move), next_rows = the options it led to."""
        self.replay_memory.append((row, reward, next_rows, done))

    def learn(self, batch_size):
        if len(self.replay_memory) < batch_size:
            return

        mini_batch = self.rng.sample(self.replay_memory, batch_size)
        rows, rewards, next_rows_list, dones = zip(*mini_batch)

        rows = torch.stack(rows)                                    # (B, input_size)
        rewards = torch.tensor(rewards, dtype=torch.float32)        # (B,)
        dones = torch.tensor(dones, dtype=torch.bool)               # (B,)

        predicted_q_values = self.main_network(rows).squeeze(1)    # (B,)

        # Pad every memory's next options to max_degree rows, with a mask marking which rows are real
        padded = torch.zeros(batch_size, self.max_degree, self.input_size)
        mask = torch.zeros(batch_size, self.max_degree, dtype=torch.bool)
        for i, next_rows in enumerate(next_rows_list):
            if next_rows is not None:
                padded[i, :len(next_rows)] = next_rows
                mask[i, :len(next_rows)] = True

        with torch.no_grad():
            online_q = self.main_network(padded).squeeze(-1)   
            online_q = online_q.masked_fill(~mask, float("-inf")) # padding can never be the max
            best = online_q.argmax(dim=1)
            target_q = self.target_network(padded).squeeze(-1) 
            max_next_q = target_q.gather(1, best.unsqueeze(1)).squeeze(1)
            max_next_q = torch.where(dones, torch.zeros_like(max_next_q), max_next_q)
            target_q_values = rewards + self.discount_factor * max_next_q

        loss = self.loss_function(predicted_q_values, target_q_values)

        self.trip_losses.append(loss.item())
        self.trip_qs.append(predicted_q_values.mean().item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.main_network.state_dict())

    def encode_node(self, node):
        """This agent's knowledge of one node as 17 numbers, always in the same order."""
        facts = self.known[node]
        vec = []

        # Visible facts, scaled 0-1
        vec.append(facts["Viewshed"])
        vec.append(facts["dist_to_water"] / DIST_SCALE_M)
        vec.append(facts["centrality"] / self.max_centrality)

        # Type: a known flag, then one slot per type (all zeros while unknown)
        node_type = facts["Type"]
        vec.append(0.0 if node_type is None else 1.0)
        for t in NODE_TYPES:
            vec.append(1.0 if node_type == t else 0.0)

        # Hidden distances: value then known flag, (0, 0) while unknown
        for name in HIDDEN_DISTANCES:
            value = facts[name]
            if value is None:
                vec.extend([0.0, 0.0])
            else:
                vec.extend([value / DIST_SCALE_M, 1.0])

        # Own experience, capped at 5 captures
        vec.append(min(facts["times_caught"], 5) / 5)
        return vec

    def encode_move(self, node, neighbor, goal):
        """One possible move node -> neighbor as 20 numbers: encode_node(neighbor) (17), edge length,
        straight-line progress toward the goal, threat at the neighbor's grid cell."""
        vec = list(self.encode_node(neighbor))
        if not self.use_knowledge:
            vec = [0.0] * len(vec)
        # Edge length, scaled by EDGE_SCALE_M
        neighbor_edge_length = self.graph.edges[node,neighbor]["terrain_dist"] / EDGE_SCALE_M
        vec.append(neighbor_edge_length)

        goal_pos = np.array(self.graph.nodes[goal]["pos"]) 
        node_to_goal = np.linalg.norm(np.array(self.graph.nodes[node]["pos"]) - goal_pos)
        neighbor_to_goal = np.linalg.norm(np.array(self.graph.nodes[neighbor]["pos"]) - goal_pos)

        # Straight-line progress toward the goal, scaled by EDGE_SCALE_M (positive = closer)
        progress = float(node_to_goal - neighbor_to_goal) / EDGE_SCALE_M
        vec.append(progress)

        # Threat where this move leads: the neighbor's cell in the threat grid (0-1)
        row, col = self._reserve_to_grid(self.graph.nodes[neighbor]["pos"], self.map_bounds)
        vec.append(float(self.threat_grid[row, col]) if self.use_threat else 0.0)

        # Progress by the shortest road route (map knowledge).
        # Zeroed when the poacher doesn't know the roads
        road_progress = (self.road_dist[node][goal] - self.road_dist[neighbor][goal]) / EDGE_SCALE_M
        vec.append(road_progress if self.use_road_distance else 0.0)

        # RB Lesson inputs: (weight of each concern) x (how much this move hits it); zeros without a reasoning bank
        vec.extend(self.lesson_features(node, neighbor))
        return vec

    def lesson_features(self, node, neighbor):
        """One number per LESSON_CONCERNS entry. Plain DQN agents have no lessons."""
        return [0.0] * len(LESSON_CONCERNS)
        