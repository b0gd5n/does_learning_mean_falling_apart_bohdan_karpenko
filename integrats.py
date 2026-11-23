import random
import math
import numpy as np
import pyphi

pyphi.config.PROGRESS_BARS = False
pyphi.config.PARALLEL = False
pyphi.config.SHORTCIRCUIT_SIA = False


class Integrat:
    def __init__(
        self, input_nodes_quantity, core_nodes_quantity, output_nodes_quantity
    ):
        self.input_nodes_quantity = input_nodes_quantity
        self.core_nodes_quantity = core_nodes_quantity
        self.output_nodes_quantity = output_nodes_quantity

        self.total_nodes_quantity = (
            self.input_nodes_quantity
            + self.core_nodes_quantity
            + self.output_nodes_quantity
        )

        self.nodes = []
        self.nodes_probabilities = []
        self.weights = []
        self.biases = []

        self.weights_gradients = []
        self.biases_gradients = []

        self.trainable_weights = []

        self.input_nodes_indexes = []
        self.output_nodes_indexes = []
        self.core_nodes_indexes = []

        self.nodes_history = []

        self.adaptive_moment_estimation_with_weight_decay_timestep = 0

        self.momentum_weights = []
        self.velocity_weights = []
        self.momentum_biases = []
        self.velocity_biases = []

        for node in range(self.total_nodes_quantity):
            self.nodes.append(0)
            self.nodes_probabilities.append(0)
            self.biases.append(0)
            self.biases_gradients.append(0)
            self.momentum_biases.append(0)
            self.velocity_biases.append(0)

            for weight in range(self.total_nodes_quantity):
                self.weights.append(0)
                self.trainable_weights.append(False)
                self.weights_gradients.append(0)
                self.momentum_weights.append(0)
                self.velocity_weights.append(0)

    def initialise(self, method="random", ring_loop_reliability=0.999, seed=None):
        copy_reliability = ring_loop_reliability ** (1 / self.core_nodes_quantity)

        core_gain = math.log(copy_reliability / (1 - copy_reliability))

        if seed is None:
            seed = random.randint(0, 2147483647)
        random.seed(seed)

        shuffled_nodes = list(range(self.total_nodes_quantity))
        random.shuffle(shuffled_nodes)

        self.input_nodes_indexes = shuffled_nodes[: self.input_nodes_quantity]
        self.output_nodes_indexes = shuffled_nodes[
            self.input_nodes_quantity : self.input_nodes_quantity
            + self.output_nodes_quantity
        ]
        self.core_nodes_indexes = shuffled_nodes[
            self.input_nodes_quantity + self.output_nodes_quantity :
        ]

        for weight in range(len(self.weights)):
            self.weights[weight] = 0
            self.trainable_weights[weight] = False
        for bias in range(len(self.biases)):
            self.biases[bias] = 0
        for core_node in range(len(self.core_nodes_indexes)):
            for input_node in range(len(self.input_nodes_indexes)):
                self.trainable_weights[
                    self.core_nodes_indexes[core_node] * self.total_nodes_quantity
                    + self.input_nodes_indexes[input_node]
                ] = True
            for core_node_index in range(len(self.core_nodes_indexes)):
                if core_node_index == core_node:
                    continue
                self.trainable_weights[
                    self.core_nodes_indexes[core_node] * self.total_nodes_quantity
                    + self.core_nodes_indexes[core_node_index]
                ] = True
        for output_node in range(len(self.output_nodes_indexes)):
            for core_node in range(len(self.core_nodes_indexes)):
                self.trainable_weights[
                    self.output_nodes_indexes[output_node] * self.total_nodes_quantity
                    + self.core_nodes_indexes[core_node]
                ] = True
        if method == "random":
            for weight in range(len(self.weights)):
                if self.trainable_weights[weight]:
                    self.weights[weight] = random.uniform(-0.5, 0.5)
            return
        core_matrix = []
        for core_node in range(len(self.core_nodes_indexes)):
            core_matrix.append([])
            for core_node_index in range(len(self.core_nodes_indexes)):
                core_matrix[core_node].append(0)
        for core_node in range(len(self.core_nodes_indexes)):
            core_matrix[core_node][core_node - 1] = 1
        for core_node in range(len(self.core_nodes_indexes)):
            for core_node_index in range(len(self.core_nodes_indexes)):
                if core_node_index == core_node:
                    continue
                self.weights[
                    self.core_nodes_indexes[core_node] * self.total_nodes_quantity
                    + self.core_nodes_indexes[core_node_index]
                ] = (2 * core_gain * core_matrix[core_node][core_node_index])
                self.biases[self.core_nodes_indexes[core_node]] -= (
                    core_gain * core_matrix[core_node][core_node_index]
                )
        for core_node in range(len(self.core_nodes_indexes)):
            if random.random() < 0.5:
                node_sign = -1
            else:
                node_sign = 1
            for input_node in range(len(self.input_nodes_indexes)):
                self.weights[
                    self.core_nodes_indexes[core_node] * self.total_nodes_quantity
                    + self.input_nodes_indexes[input_node]
                ] = node_sign * abs(random.gauss(0, 1))
        for output_node in range(len(self.output_nodes_indexes)):
            for core_node in range(len(self.core_nodes_indexes)):
                self.weights[
                    self.output_nodes_indexes[output_node] * self.total_nodes_quantity
                    + self.core_nodes_indexes[core_node]
                ] = random.gauss(0, 1) / math.sqrt(len(self.core_nodes_indexes))

    def step_forward(self, inputs=None):
        if inputs is not None:
            for input_node in range(len(inputs)):
                self.nodes[self.input_nodes_indexes[input_node]] = inputs[input_node]
        old_state = list(self.nodes)
        new_state = list(self.nodes)

        old_probabilities = list(self.nodes_probabilities)
        new_probabilities = list(self.nodes_probabilities)

        summands = []
        for node in range(self.total_nodes_quantity):
            summands.append(0)
        for core_node in range(len(self.core_nodes_indexes)):
            summand = self.biases[self.core_nodes_indexes[core_node]]
            for input_node in range(len(self.input_nodes_indexes)):
                summand += (
                    old_state[self.input_nodes_indexes[input_node]]
                    * self.weights[
                        self.core_nodes_indexes[core_node] * self.total_nodes_quantity
                        + self.input_nodes_indexes[input_node]
                    ]
                )
            for core_node_index in range(len(self.core_nodes_indexes)):
                if core_node_index == core_node:
                    continue
                summand += (
                    old_state[self.core_nodes_indexes[core_node_index]]
                    * self.weights[
                        self.core_nodes_indexes[core_node] * self.total_nodes_quantity
                        + self.core_nodes_indexes[core_node_index]
                    ]
                )
            summands[self.core_nodes_indexes[core_node]] = summand

            new_state[self.core_nodes_indexes[core_node]] = self.activation_function(
                value=summand
            )
            new_probabilities[self.core_nodes_indexes[core_node]] = (
                self.activation_probability(value=summand)
            )
        for output_node in range(len(self.output_nodes_indexes)):
            summand = self.biases[self.output_nodes_indexes[output_node]]
            for core_node in range(len(self.core_nodes_indexes)):
                summand += (
                    old_probabilities[self.core_nodes_indexes[core_node]]
                    * self.weights[
                        self.output_nodes_indexes[output_node]
                        * self.total_nodes_quantity
                        + self.core_nodes_indexes[core_node]
                    ]
                )
            summands[self.output_nodes_indexes[output_node]] = summand
            new_state[self.output_nodes_indexes[output_node]] = summand
        self.nodes = new_state
        self.nodes_probabilities = new_probabilities

        self.nodes_history.append(
            (old_state, list(self.nodes), summands, old_probabilities)
        )

        output = []
        for output_node in range(len(self.output_nodes_indexes)):
            output.append(self.nodes[self.output_nodes_indexes[output_node]])
        return output

    def step_backward(self, sequence_losses):
        future_nodes_gradients = []
        for node in range(self.total_nodes_quantity):
            future_nodes_gradients.append(0)
        for timestep in reversed(range(len(self.nodes_history))):
            state_before, state_after, summands, probabilities_before = (
                self.nodes_history[timestep]
            )

            current_nodes_gradients = future_nodes_gradients.copy()

            for output_node in range(len(self.output_nodes_indexes)):
                current_nodes_gradients[
                    self.output_nodes_indexes[output_node]
                ] += sequence_losses[timestep][output_node]
            new_future_nodes_gradients = []
            for node in range(self.total_nodes_quantity):
                new_future_nodes_gradients.append(0)
            for output_node in range(len(self.output_nodes_indexes)):
                self.biases_gradients[
                    self.output_nodes_indexes[output_node]
                ] += current_nodes_gradients[self.output_nodes_indexes[output_node]]
                for core_node in range(len(self.core_nodes_indexes)):
                    self.weights_gradients[
                        self.output_nodes_indexes[output_node]
                        * self.total_nodes_quantity
                        + self.core_nodes_indexes[core_node]
                    ] += (
                        current_nodes_gradients[self.output_nodes_indexes[output_node]]
                        * probabilities_before[self.core_nodes_indexes[core_node]]
                    )

                    new_future_nodes_gradients[self.core_nodes_indexes[core_node]] += (
                        current_nodes_gradients[self.output_nodes_indexes[output_node]]
                        * self.weights[
                            self.output_nodes_indexes[output_node]
                            * self.total_nodes_quantity
                            + self.core_nodes_indexes[core_node]
                        ]
                    )
            for core_node in range(len(self.core_nodes_indexes)):
                derivative = self.derivative_activation_function(
                    summands[self.core_nodes_indexes[core_node]]
                )

                self.biases_gradients[self.core_nodes_indexes[core_node]] += (
                    current_nodes_gradients[self.core_nodes_indexes[core_node]]
                    * derivative
                )
                for input_node in range(len(self.input_nodes_indexes)):
                    self.weights_gradients[
                        self.core_nodes_indexes[core_node] * self.total_nodes_quantity
                        + self.input_nodes_indexes[input_node]
                    ] += (
                        current_nodes_gradients[self.core_nodes_indexes[core_node]]
                        * derivative
                        * state_before[self.input_nodes_indexes[input_node]]
                    )
                for core_node_index in range(len(self.core_nodes_indexes)):
                    if core_node_index == core_node:
                        continue
                    self.weights_gradients[
                        self.core_nodes_indexes[core_node] * self.total_nodes_quantity
                        + self.core_nodes_indexes[core_node_index]
                    ] += (
                        current_nodes_gradients[self.core_nodes_indexes[core_node]]
                        * derivative
                        * state_before[self.core_nodes_indexes[core_node_index]]
                    )

                    new_future_nodes_gradients[
                        self.core_nodes_indexes[core_node_index]
                    ] += (
                        current_nodes_gradients[self.core_nodes_indexes[core_node]]
                        * derivative
                        * self.weights[
                            self.core_nodes_indexes[core_node]
                            * self.total_nodes_quantity
                            + self.core_nodes_indexes[core_node_index]
                        ]
                    )
            future_nodes_gradients = new_future_nodes_gradients.copy()

    def activation_function(self, value):
        if random.random() < 1 / (1 + math.exp(-value)):
            return 1
        else:
            return 0

    def activation_probability(self, value):
        return 1 / (1 + math.exp(-value))

    def derivative_activation_function(self, value):
        return 1 / (1 + math.exp(-value)) * (1 - 1 / (1 + math.exp(-value)))

    def adaptive_moment_estimation_with_weight_decay(
        self,
        batch_size,
        learning_rate=1e-4,
        weight_decay=1e-5,
        first_beta=0.9,
        second_beta=0.999,
        epsilon=1e-8,
    ):
        self.adaptive_moment_estimation_with_weight_decay_timestep += 1

        for weight in range(len(self.weights)):
            if not self.trainable_weights[weight]:
                continue
            self.momentum_weights[weight] = first_beta * self.momentum_weights[
                weight
            ] + (1 - first_beta) * (self.weights_gradients[weight] / batch_size)
            self.velocity_weights[weight] = second_beta * self.velocity_weights[
                weight
            ] + (1 - second_beta) * ((self.weights_gradients[weight] / batch_size) ** 2)

            momentum_corrected = self.momentum_weights[weight] / (
                1
                - first_beta**self.adaptive_moment_estimation_with_weight_decay_timestep
            )
            velocity_corrected = self.velocity_weights[weight] / (
                1
                - second_beta
                ** self.adaptive_moment_estimation_with_weight_decay_timestep
            )

            self.weights[weight] -= learning_rate * (
                momentum_corrected / (math.sqrt(velocity_corrected) + epsilon)
                + weight_decay * self.weights[weight]
            )
        for bias in range(len(self.biases)):
            self.momentum_biases[bias] = first_beta * self.momentum_biases[bias] + (
                1 - first_beta
            ) * (self.biases_gradients[bias] / batch_size)
            self.velocity_biases[bias] = second_beta * self.velocity_biases[bias] + (
                1 - second_beta
            ) * ((self.biases_gradients[bias] / batch_size) ** 2)

            momentum_corrected = self.momentum_biases[bias] / (
                1
                - first_beta**self.adaptive_moment_estimation_with_weight_decay_timestep
            )
            velocity_corrected = self.velocity_biases[bias] / (
                1
                - second_beta
                ** self.adaptive_moment_estimation_with_weight_decay_timestep
            )

            self.biases[bias] -= learning_rate * (
                momentum_corrected / (math.sqrt(velocity_corrected) + epsilon)
            )

    def clear_gradients(self):
        for weight in range(len(self.weights_gradients)):
            self.weights_gradients[weight] = 0
        for bias in range(len(self.biases_gradients)):
            self.biases_gradients[bias] = 0

    def clear_nodes_history(self):
        self.nodes_history.clear()

    def reset_network_state(self):
        for node in range(len(self.nodes)):
            self.nodes[node] = 0
            self.nodes_probabilities[node] = 0
        self.clear_nodes_history()

    def analyse_integration(
        self, state=None, system=None, background_inputs=None, compute_big_phi=True
    ):
        if state is None:
            state = []
            for core_node in range(len(self.core_nodes_indexes)):
                state.append(0)
        if background_inputs is None:
            background_inputs = []
            for input_node in range(len(self.input_nodes_indexes)):
                background_inputs.append(0)
        transition_probability_matrix = np.zeros(
            (2 ** len(self.core_nodes_indexes), len(self.core_nodes_indexes)),
            dtype=np.float64,
        )

        for state_index in range(2 ** len(self.core_nodes_indexes)):
            for core_node in range(len(self.core_nodes_indexes)):
                summand = self.biases[self.core_nodes_indexes[core_node]]
                for input_node in range(len(self.input_nodes_indexes)):
                    summand += (
                        background_inputs[input_node]
                        * self.weights[
                            self.core_nodes_indexes[core_node]
                            * self.total_nodes_quantity
                            + self.input_nodes_indexes[input_node]
                        ]
                    )
                for core_node_index in range(len(self.core_nodes_indexes)):
                    if core_node_index == core_node:
                        continue
                    summand += ((state_index >> core_node_index) & 1) * self.weights[
                        self.core_nodes_indexes[core_node] * self.total_nodes_quantity
                        + self.core_nodes_indexes[core_node_index]
                    ]
                transition_probability_matrix[state_index, core_node] = 1 / (
                    1 + math.exp(-summand)
                )
        connectivity_matrix = np.ones(
            (len(self.core_nodes_indexes), len(self.core_nodes_indexes)), dtype=int
        )
        for core_node in range(len(self.core_nodes_indexes)):
            connectivity_matrix[core_node, core_node] = 0
        binary_state = []
        for core_node in range(len(state)):
            if state[core_node] == 1:
                binary_state.append(1)
            else:
                binary_state.append(0)
        try:
            subsystem = pyphi.Subsystem(
                pyphi.Network(transition_probability_matrix, cm=connectivity_matrix),
                tuple(binary_state),
                nodes=system,
            )

            if compute_big_phi:
                integration = pyphi.new_big_phi.phi_structure(subsystem)
                integration.small_phi = integration.sia.phi
            else:
                integration = pyphi.new_big_phi.sia(subsystem)
                integration.small_phi = integration.phi
                integration.big_phi = None
        except pyphi.exceptions.StateUnreachableError:
            return None
        return integration
