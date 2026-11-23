import csv
import math
import os
import random
import sys
from integrats import Integrat

experiment = sys.argv[1]

os.makedirs("data", exist_ok=True)

initialisations = ["random", "ring"]
ring_loop_reliability = 0.999

if experiment == "atlas":
    sizes = [2, 3, 4, 5]
    seeds = [0, 1, 2, 3]

    atlas_file = open(os.path.join("data", "atlas.csv"), "w", newline="")
    atlas_writer = csv.writer(atlas_file)
    atlas_writer.writerow(
        ["size", "initialisation", "seed", "state", "small_phi", "big_phi"]
    )

    for size in range(len(sizes)):
        for initialisation in range(len(initialisations)):
            for seed in range(len(seeds)):
                model = Integrat(1, sizes[size], 1)
                model.initialise(
                    method=initialisations[initialisation],
                    ring_loop_reliability=ring_loop_reliability,
                    seed=seeds[seed],
                )
                for state_index in range(2 ** sizes[size]):
                    state = []
                    for node in range(sizes[size]):
                        state.append((state_index >> node) & 1)
                    integration = model.analyse_integration(
                        state=state, compute_big_phi=True
                    )
                    small_phi = "unreachable"
                    big_phi = ""
                    if integration is not None:
                        small_phi = max(integration.small_phi, 0)
                        big_phi = 0
                        if small_phi > 0:
                            big_phi = integration.big_phi
                    atlas_writer.writerow(
                        [
                            sizes[size],
                            initialisations[initialisation],
                            seeds[seed],
                            state_index,
                            small_phi,
                            big_phi,
                        ]
                    )
                atlas_file.flush()
    atlas_file.close()
    sys.exit()
task = experiment
sizes = [2, 3, 4]
seeds = [0, 1, 2, 3]
delays = [0, 1, 2, 4, 8]
epochs = 16384
measure_every = 1024
shuffles = 8
batch_size = 8
learning_rate = 1e-3
test_sequences = 128
stream_length = 16

accuracy_file = open(
    os.path.join("data", "training_accuracy_" + task + ".csv"), "w", newline=""
)
accuracy_writer = csv.writer(accuracy_file)
accuracy_writer.writerow(
    ["size", "initialisation", "task", "delay", "seed", "epoch", "shuffle", "accuracy"]
)

integration_file = open(
    os.path.join("data", "training_integration_" + task + ".csv"), "w", newline=""
)
integration_writer = csv.writer(integration_file)
integration_writer.writerow(
    [
        "size",
        "initialisation",
        "task",
        "delay",
        "seed",
        "epoch",
        "shuffle",
        "state",
        "small_phi",
        "big_phi",
    ]
)

weights_file = open(
    os.path.join("data", "training_weights_" + task + ".csv"), "w", newline=""
)
weights_writer = csv.writer(weights_file)
weights_writer.writerow(
    [
        "size",
        "initialisation",
        "task",
        "delay",
        "seed",
        "input_nodes_indexes",
        "core_nodes_indexes",
        "output_nodes_indexes",
        "weights",
        "biases",
    ]
)

for size in range(len(sizes)):
    for initialisation in range(len(initialisations)):
        for delay in range(len(delays)):
            for seed in range(len(seeds)):
                if task == "and" or task == "or" or task == "xor":
                    model = Integrat(2, sizes[size], 1)
                else:
                    model = Integrat(1, sizes[size], 1)
                model.initialise(
                    method=initialisations[initialisation],
                    ring_loop_reliability=ring_loop_reliability,
                    seed=seeds[seed],
                )

                if initialisations[initialisation] == "ring":
                    for core_node in range(len(model.core_nodes_indexes)):
                        for input_node in range(len(model.input_nodes_indexes)):
                            model.weights[
                                model.core_nodes_indexes[core_node]
                                * model.total_nodes_quantity
                                + model.input_nodes_indexes[input_node]
                            ] *= 2 * math.log(
                                ring_loop_reliability ** (1 / sizes[size])
                                / (1 - ring_loop_reliability ** (1 / sizes[size]))
                            )
                random.seed(seeds[seed] + 1024)

                for epoch in range(epochs + 1 + shuffles):
                    if epoch == epochs + 1:
                        random.seed(seeds[seed] + 2048)
                    if epoch > epochs:
                        shuffled_values = []
                        for core_node in range(len(model.core_nodes_indexes)):
                            for core_node_index in range(len(model.core_nodes_indexes)):
                                if core_node_index == core_node:
                                    continue
                                shuffled_values.append(
                                    original_weights[
                                        model.core_nodes_indexes[core_node]
                                        * model.total_nodes_quantity
                                        + model.core_nodes_indexes[core_node_index]
                                    ]
                                )
                        random.shuffle(shuffled_values)

                        for core_node in range(len(model.core_nodes_indexes)):
                            for core_node_index in range(len(model.core_nodes_indexes)):
                                if core_node_index == core_node:
                                    continue
                                model.weights[
                                    model.core_nodes_indexes[core_node]
                                    * model.total_nodes_quantity
                                    + model.core_nodes_indexes[core_node_index]
                                ] = shuffled_values.pop(0)
                    model.clear_gradients()
                    correct = 0
                    tested_ticks = 0

                    for sequence in range(batch_size + test_sequences):
                        if sequence < batch_size and (epoch == 0 or epoch > epochs):
                            continue
                        if (
                            sequence >= batch_size
                            and epoch % measure_every != 0
                            and epoch < epochs
                        ):
                            break
                        if sequence == batch_size:
                            training_random_state = random.getstate()
                            random.seed(12345)
                        inputs = []
                        targets = []

                        if task == "and" or task == "or" or task == "xor":
                            first_bits = []
                            second_bits = []
                            for timestep in range(stream_length):
                                first_bits.append(random.randint(0, 1))
                                second_bits.append(random.randint(0, 1))
                                inputs.append(
                                    [
                                        2 * first_bits[timestep] - 1,
                                        2 * second_bits[timestep] - 1,
                                    ]
                                )
                                if timestep < 1 + delays[delay]:
                                    targets.append(None)
                                elif task == "and":
                                    targets.append(
                                        first_bits[timestep - 1 - delays[delay]]
                                        & second_bits[timestep - 1 - delays[delay]]
                                    )
                                elif task == "or":
                                    targets.append(
                                        first_bits[timestep - 1 - delays[delay]]
                                        | second_bits[timestep - 1 - delays[delay]]
                                    )
                                else:
                                    targets.append(
                                        first_bits[timestep - 1 - delays[delay]]
                                        ^ second_bits[timestep - 1 - delays[delay]]
                                    )
                        elif task == "one_back_xor":
                            bits = []
                            for timestep in range(stream_length):
                                bits.append(random.randint(0, 1))
                                inputs.append([2 * bits[timestep] - 1])
                                if timestep < 2 + delays[delay]:
                                    targets.append(None)
                                else:
                                    targets.append(
                                        bits[timestep - 1 - delays[delay]]
                                        ^ bits[timestep - 2 - delays[delay]]
                                    )
                        elif task == "noise":
                            for timestep in range(stream_length):
                                inputs.append([2 * random.randint(0, 1) - 1])
                                if timestep < 1 + delays[delay]:
                                    targets.append(None)
                                else:
                                    targets.append(random.randint(0, 1))
                        else:
                            bits = [1, 1, 0, 0]
                            random.shuffle(bits)
                            for timestep in range(len(bits)):
                                inputs.append([2 * bits[timestep] - 1])
                                targets.append(None)
                            silent_ticks = 8
                            if task == "replay":
                                silent_ticks = 4
                            for timestep in range(delays[delay] + silent_ticks):
                                inputs.append([0])
                                if timestep < delays[delay]:
                                    targets.append(None)
                                else:
                                    targets.append(
                                        bits[(timestep - delays[delay]) % len(bits)]
                                    )
                        model.reset_network_state()
                        outputs = []
                        for timestep in range(len(inputs)):
                            outputs.append(model.step_forward(inputs[timestep])[0])
                        if sequence < batch_size:
                            answered_ticks = 0
                            for timestep in range(len(targets)):
                                if targets[timestep] is not None:
                                    answered_ticks += 1
                            losses = []
                            for timestep in range(len(targets)):
                                if targets[timestep] is None:
                                    losses.append([0])
                                else:
                                    losses.append(
                                        [
                                            2
                                            * (outputs[timestep] - targets[timestep])
                                            / answered_ticks
                                        ]
                                    )
                            model.step_backward(losses)
                        else:
                            for timestep in range(len(targets)):
                                if targets[timestep] is None:
                                    continue
                                if (outputs[timestep] > 0.5) == (
                                    targets[timestep] == 1
                                ):
                                    correct += 1
                                tested_ticks += 1
                    if 0 < epoch <= epochs:
                        model.adaptive_moment_estimation_with_weight_decay(
                            batch_size=batch_size, learning_rate=learning_rate
                        )
                    if epoch % measure_every != 0 and epoch < epochs:
                        continue
                    random.setstate(training_random_state)

                    shuffle = "none"
                    if epoch > epochs:
                        shuffle = epoch - epochs - 1
                    accuracy_writer.writerow(
                        [
                            sizes[size],
                            initialisations[initialisation],
                            task,
                            delays[delay],
                            seeds[seed],
                            min(epoch, epochs),
                            shuffle,
                            correct / tested_ticks,
                        ]
                    )
                    for state_index in range(2 ** sizes[size]):
                        state = []
                        for node in range(sizes[size]):
                            state.append((state_index >> node) & 1)
                        integration = model.analyse_integration(
                            state=state, compute_big_phi=True
                        )
                        small_phi = "unreachable"
                        big_phi = ""
                        if integration is not None:
                            small_phi = max(integration.small_phi, 0)
                            big_phi = 0
                            if small_phi > 0:
                                big_phi = integration.big_phi
                        integration_writer.writerow(
                            [
                                sizes[size],
                                initialisations[initialisation],
                                task,
                                delays[delay],
                                seeds[seed],
                                min(epoch, epochs),
                                shuffle,
                                state_index,
                                small_phi,
                                big_phi,
                            ]
                        )
                    if epoch == epochs:
                        original_weights = list(model.weights)
                        weights_writer.writerow(
                            [
                                sizes[size],
                                initialisations[initialisation],
                                task,
                                delays[delay],
                                seeds[seed],
                                model.input_nodes_indexes,
                                model.core_nodes_indexes,
                                model.output_nodes_indexes,
                                model.weights,
                                model.biases,
                            ]
                        )
                accuracy_file.flush()
                integration_file.flush()
                weights_file.flush()
accuracy_file.close()
integration_file.close()
weights_file.close()
