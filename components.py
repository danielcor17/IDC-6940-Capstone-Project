from math import sqrt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as fn
import torchvision


class NoisyLinear(nn.Module):
    """ Factorised NoisyLinear layer with bias """

    def __init__(self, in_features, out_features, std_init=0.5):

        super(NoisyLinear, self).__init__()

        self.in_features, self.out_features, self.std_init = in_features, out_features, std_init
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):

        mu_range = 1 / sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.std_init / sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / sqrt(self.out_features))

    def _scale_noise(self, size):

        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):

        epsilon_in, epsilon_out = self._scale_noise(self.in_features), self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, data):

        if self.training:
            noisy_weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            noisy_bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
            return fn.linear(data, noisy_weight, noisy_bias)
        else:
            return fn.linear(data, self.weight_mu, self.bias_mu)


class DQN(nn.Module):

    def __init__(self, args, action_space):
        super(DQN, self).__init__()

        self.action_space = action_space
        self.atoms = args.atoms

        if args.architecture == 'canonical':
            self.convs = nn.Sequential(
                nn.Conv2d(args.history_length, 32, 8, stride=4, padding=0), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU()
            )

            # self.conv_output_size = 1024
            self.conv_output_size = self.convs(torch.ones([1, args.history_length, 64, 64])).flatten().shape[0]

        if args.architecture == 'data-efficient':
            self.convs = nn.Sequential(
                nn.Conv2d(args.history_length, 32, 5, stride=5, padding=0), nn.ReLU(),
                nn.Conv2d(32, 64, 5, stride=5, padding=0), nn.ReLU()
            )

            # self.conv_output_size = 256
            self.conv_output_size = self.convs(torch.ones([1, args.history_length, 64, 64])).flatten().shape[0]

        if args.architecture == 'covnext':

            self.convs = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT)

            for param in self.convs.parameters():
                param.requires_grad = False

            new_base = torch.nn.modules.conv.Conv2d(args.history_length, 96, (4, 4), (4, 4))

            # old_base = self.convs._modules['features'][0][0]
            # for param in new_base.parameters():
            #     param.requires_grad = False
            #
            # for i in range(96):
            #     list(new_base.parameters())[0][i, ...] = \
            #         list(old_base.parameters())[0][i, np.random.choice(3, args.history_length), :, :]
            #
            # list(new_base.parameters())[1] = list(old_base.parameters())[1]
            #
            # for param in new_base.parameters():
            #     param.requires_grad = True

            self.convs._modules['features'][0][0] = new_base
            del self.convs._modules['classifier'][2]

            # self.conv_output_size = 768
            self.conv_output_size = self.convs(torch.ones([1, args.history_length, 128, 128])).flatten().shape[0]

        self.fc_h_v = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
        self.fc_h_a = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
        self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
        self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std)

    def forward(self, x, log=False):
        x = self.convs(x)
        x = x.view(-1, self.conv_output_size)
        # Value stream and Advantage stream
        v, a = self.fc_z_v(fn.relu(self.fc_h_v(x))), self.fc_z_a(fn.relu(self.fc_h_a(x)))
        v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
        q = v + a - a.mean(1, keepdim=True)  # Combine streams
        if log:  # Use log softmax for numerical stability
            q = fn.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
        else:
            q = fn.softmax(q, dim=2)  # Probabilities with action over second dimension
        return q

    def reset_noise(self):
        for name, module in self.named_children():
            if 'fc' in name:
                module.reset_noise()


class SegmentTree:
    """ Segment tree data structure where parent node values are sum/max of children node values """

    def __init__(self, args, size):

        if args.architecture in ('canonical', 'data-efficient'):
            shape = (64, 64)
        else:
            shape = (128, 128)

        trans_dtype = np.dtype(
            [
                ('timestep', np.int32),
                ('state', np.uint8, shape),
                ('action', np.int32),
                ('reward', np.float32),
                ('nonterminal', np.bool_)
            ]
        )

        self._blank_trans = (0, np.zeros(shape, dtype=np.uint8), 0, 0.0, False)
        self.index = 0
        self.size = size
        self.full = False  # Used to track actual capacity
        self.tree_start = 2 ** (size - 1).bit_length() - 1  # Put all used node leaves on last tree level
        self.sum_tree = np.zeros((self.tree_start + self.size,), dtype=np.float32)
        self.data = np.array([self._blank_trans] * size, dtype=trans_dtype)  # Build structured array
        self.max = 1  # Initial max value to return (1 = 1^ω)

    def _update_nodes(self, indices):
        """ Update nodes values from current tree """
        children_indices = indices * 2 + np.expand_dims([1, 2], axis=1)
        self.sum_tree[indices] = np.sum(self.sum_tree[children_indices], axis=0)

    def _propagate(self, indices):
        """ Propagate changes up tree given tree indices """
        parents = (indices - 1) // 2
        unique_parents = np.unique(parents)
        self._update_nodes(unique_parents)
        if parents[0] != 0:
            self._propagate(parents)

    def _propagate_index(self, index):
        """ Propagate single value up tree given a tree index for efficiency """
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate_index(parent)

    def update(self, indices, values):
        """ Update values given tree indices """
        self.sum_tree[indices] = values  # Set new values
        self._propagate(indices)  # Propagate values
        current_max_value = np.max(values)
        self.max = max(current_max_value, self.max)

    def _update_index(self, index, value):
        """ Update single value given a tree index for efficiency """
        self.sum_tree[index] = value  # Set new value
        self._propagate_index(index)  # Propagate value
        self.max = max(value, self.max)

    def append(self, data, value):
        self.data[self.index] = data  # Store data in underlying data structure
        self._update_index(self.index + self.tree_start, value)  # Update tree
        self.index = (self.index + 1) % self.size  # Update index
        self.full = self.full or self.index == 0  # Save when capacity reached
        self.max = max(value, self.max)

    def _retrieve(self, indices, values):
        """ Search for the location of values in the sum tree """
        # If indices correspond to leaf nodes, return them
        # If children indices correspond to leaf nodes, bound rare outliers in case total slightly overshoots
        children_indices = (indices * 2 + np.expand_dims([1, 2], axis=1))  # Make matrix of children indices
        if children_indices[0, 0] >= self.sum_tree.shape[0]:
            return indices
        elif children_indices[0, 0] >= self.tree_start:
            children_indices = np.minimum(children_indices, self.sum_tree.shape[0] - 1)

        # Classify which values are in left or right branches
        # Use classification to index into the indices matrix
        # Subtract the left branch values when searching in the right branch
        left_children_values = self.sum_tree[children_indices[0]]
        successor_choices = np.greater(values, left_children_values).astype(np.int32)
        successor_indices = children_indices[successor_choices, np.arange(indices.size)]
        successor_values = values - successor_choices * left_children_values

        return self._retrieve(successor_indices, successor_values)

    def find(self, values):
        """ Searches for values in sum tree and returns values, data indices and tree indices """
        indices = self._retrieve(np.zeros(values.shape, dtype=np.int32), values)
        data_index = indices - self.tree_start
        return self.sum_tree[indices], data_index, indices

    def get(self, data_index):
        """ Returns data given a data index """
        return self.data[data_index % self.size]

    def total(self):
        return self.sum_tree[0]

    @property
    def blank_trans(self):
        return self._blank_trans


class ReplayMemory:
    def __init__(self, args, capacity):
        self.device = args.device
        self.capacity = capacity
        self.history = args.history_length
        self.discount = args.discount
        self.n = args.multi_step
        # Initial importance sampling weight β, annealed to 1 over course of training
        self.priority_weight = args.priority_weight
        self.priority_exponent = args.priority_exponent
        self.t = 0  # Internal episode timestep counter
        # Discount-scaling vector for n-step returns
        self.n_step_scaling = torch.tensor(
            [self.discount ** i for i in range(self.n)], dtype=torch.float32, device=self.device
        )
        # Store transitions in a wrap-around cyclic buffer within a sum tree for querying priorities
        self.transitions = SegmentTree(args, capacity)
        self._blank_trans = self.transitions.blank_trans

    def append(self, state, action, reward, terminal):
        """ Add state and action at time t, reward and terminal at time t + 1 """
        # Only store last frame and discretise to save memory
        state = state[-1].mul(255).to(dtype=torch.uint8, device=torch.device('cpu'))
        # Store new transition with maximum priority
        self.transitions.append((self.t, state, action, reward, not terminal), self.transitions.max)
        self.t = 0 if terminal else self.t + 1  # Start new episodes with t = 0

    def _get_transitions(self, idxs):
        """ Return the transitions with blank states where appropriate """
        transition_idxs = np.arange(-self.history + 1, self.n + 1) + np.expand_dims(idxs, axis=1)
        transitions = self.transitions.get(transition_idxs)
        transitions_firsts = transitions['timestep'] == 0
        blank_mask = np.zeros_like(transitions_firsts, dtype=np.bool_)
        for t in range(self.history - 2, -1, -1):  # e.g. 2 1 0
            # True if future frame has timestep 0
            blank_mask[:, t] = np.logical_or(blank_mask[:, t + 1], transitions_firsts[:, t + 1])
        for t in range(self.history, self.history + self.n):  # e.g. 4 5 6
            # True if current or past frame has timestep 0
            blank_mask[:, t] = np.logical_or(blank_mask[:, t - 1], transitions_firsts[:, t])
        transitions[blank_mask] = self._blank_trans
        return transitions

    def _get_samples_from_segments(self, batch_size, p_total):
        """ Return a valid sample from each segment """
        segment_length = p_total / batch_size  # Batch size number of segments, based on sum over all probabilities
        segment_starts = np.arange(batch_size) * segment_length
        valid = False
        while not valid:
            # Uniformly sample from within all segments
            samples = np.random.uniform(0.0, segment_length, [batch_size]) + segment_starts
            # Retrieve samples from tree with un-normalised probability
            probs, idxs, tree_idxs = self.transitions.find(samples)
            condition = (
                    np.all((self.transitions.index - idxs) % self.capacity > self.n) and
                    np.all((idxs - self.transitions.index) % self.capacity >= self.history) and
                    np.all(probs != 0)
            )
            if condition:
                # Note that conditions are valid but extra conservative around buffer index 0
                valid = True
        # Retrieve all required transition data (from t - h to t + n)
        transitions = self._get_transitions(idxs)
        # Create un-discretised states and nth next states
        all_states = transitions['state']
        states = torch.tensor(all_states[:, :self.history], device=self.device, dtype=torch.float32).div_(255)
        next_states = torch.tensor(
            all_states[:, self.n:self.n + self.history], device=self.device, dtype=torch.float32
        ).div_(255)
        # Discrete actions to be used as index
        actions = torch.tensor(
            np.copy(transitions['action'][:, self.history - 1]), dtype=torch.int64, device=self.device
        )
        # Calculate truncated n-step discounted returns R^n = Σ_k=0->n-1 (γ^k)R_t+k+1
        # (note that invalid nth next states have reward 0)
        rewards = torch.tensor(
            np.copy(transitions['reward'][:, self.history - 1:-1]), dtype=torch.float32, device=self.device
        )
        R = torch.matmul(rewards, self.n_step_scaling)
        # Mask for non-terminal nth next states
        nonterminals = torch.tensor(
            np.expand_dims(transitions['nonterminal'][:, self.history + self.n - 1], axis=1),
            dtype=torch.float32, device=self.device
        )
        return probs, idxs, tree_idxs, states, actions, R, next_states, nonterminals

    def sample(self, batch_size):
        # Retrieve sum of all priorities (used to create a normalised probability distribution)
        p_total = self.transitions.total()
        # Get batch of valid samples
        probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals = self._get_samples_from_segments(
            batch_size, p_total
        )
        probs = probs / p_total  # Calculate normalised probabilities
        capacity = self.capacity if self.transitions.full else self.transitions.index
        weights = (capacity * probs) ** -self.priority_weight  # Compute importance-sampling weights w
        # Normalise by max importance-sampling weight from batch
        weights = torch.tensor(
            weights / weights.max(), dtype=torch.float32, device=self.device
        )
        return tree_idxs, states, actions, returns, next_states, nonterminals, weights

    def update_priorities(self, idxs, priorities):
        priorities = np.power(priorities, self.priority_exponent)
        self.transitions.update(idxs, priorities)

    def __iter__(self):
        """ Set up internal state for iterator """
        self.current_idx = 0
        return self

    def __next__(self):
        """ Return valid states for validation """
        if self.current_idx == self.capacity:
            raise StopIteration
        transitions = self.transitions.data[np.arange(self.current_idx - self.history + 1, self.current_idx + 1)]
        transitions_firsts = transitions['timestep'] == 0
        blank_mask = np.zeros_like(transitions_firsts, dtype=np.bool_)
        for t in reversed(range(self.history - 1)):
            # If future frame has timestep 0
            blank_mask[t] = np.logical_or(blank_mask[t + 1], transitions_firsts[t + 1])
        transitions[blank_mask] = self._blank_trans
        # Agent will turn into batch
        state = torch.tensor(transitions['state'], dtype=torch.float32, device=self.device).div_(255)
        self.current_idx += 1
        return state

    next = __next__  # Alias __next__ for Python 2 compatibility
