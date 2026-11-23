import csv
import glob
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def apply_adaptive_scale(axis, values, horizontal=False):
    positive_values = []
    for value in range(len(values)):
        if values[value] == values[value] and values[value] > 0:
            positive_values.append(values[value])
    if len(positive_values) == 0:
        return
    if max(positive_values) / min(positive_values) <= 100:
        return
    linear_threshold = max(min(positive_values), max(positive_values) / 10000)
    if horizontal:
        axis.set_xscale("symlog", linthresh=linear_threshold)
    else:
        axis.set_yscale("symlog", linthresh=linear_threshold)


epochs = 16384
shuffles = 8
tasks = ["and", "or", "xor", "one_back_xor", "replay", "replay_twice", "noise"]
initialisations = ["random", "ring"]
languages = ["en", "de"]
metrics = ["small_phi", "big_phi", "accuracy"]
bar_columns = ["before", "after", "shuffled"]

texts = {
    "en": {
        "small_phi": "φₛ",
        "big_phi": "Φ",
        "accuracy": "Test accuracy",
        "epoch": "Epoch",
        "size": "Core size",
        "delay": "Delay",
        "nodes": "nodes",
        "random": "random initialisation",
        "ring": "ring initialisation",
        "before": "before training",
        "after": "after training",
        "shuffled": "shuffled weights",
        "atlas_title": "Integration of untrained cores",
        "median_note": "median, 25–75%",
        "training_title": "Dynamics during training",
        "controls_title": "Before and after training, shuffle control",
        "delay_title": "Effect of delay",
        "scatter_title": "Accuracy and integration after training",
        "and": "AND",
        "or": "OR",
        "xor": "XOR",
        "one_back_xor": "one-back XOR",
        "replay": "replay",
        "replay_twice": "replay twice",
        "noise": "noise",
    },
    "de": {
        "small_phi": "φₛ",
        "big_phi": "Φ",
        "accuracy": "Testgenauigkeit",
        "epoch": "Epoche",
        "size": "Kerngröße",
        "delay": "Verzögerung",
        "nodes": "Knoten",
        "random": "zufällige Initialisierung",
        "ring": "Ring-Initialisierung",
        "before": "vor dem Training",
        "after": "nach dem Training",
        "shuffled": "gemischte Gewichte",
        "atlas_title": "Integration untrainierter Kerne",
        "median_note": "Median, 25–75 %",
        "training_title": "Verlauf während des Trainings",
        "controls_title": "Vor und nach dem Training, Mischkontrolle",
        "delay_title": "Einfluss der Verzögerung",
        "scatter_title": "Genauigkeit und Integration nach dem Training",
        "and": "UND",
        "or": "ODER",
        "xor": "XOR",
        "one_back_xor": "XOR benachbarter Bits",
        "replay": "Wiederholung",
        "replay_twice": "doppelte Wiederholung",
        "noise": "Rauschen",
    },
}

for language in range(len(languages)):
    os.makedirs(os.path.join("plots", languages[language]), exist_ok=True)
atlas_values = {}
if os.path.exists(os.path.join("data", "atlas.csv")):
    atlas_file = open(os.path.join("data", "atlas.csv"), newline="")
    atlas_rows = list(csv.DictReader(atlas_file))
    atlas_file.close()
    for row in range(len(atlas_rows)):
        if atlas_rows[row]["small_phi"] == "unreachable":
            continue
        for metric in range(2):
            if atlas_rows[row][metrics[metric]] == "":
                continue
            atlas_values.setdefault(
                (
                    int(atlas_rows[row]["size"]),
                    atlas_rows[row]["initialisation"],
                    metrics[metric],
                ),
                [],
            ).append(float(atlas_rows[row][metrics[metric]]))
runs = {}
integration_files = sorted(glob.glob(os.path.join("data", "training_integration*.csv")))
for integration_file_index in range(len(integration_files)):
    integration_file = open(integration_files[integration_file_index], newline="")
    integration_rows = list(csv.DictReader(integration_file))
    integration_file.close()
    sums = {}
    for row in range(len(integration_rows)):
        if integration_rows[row]["small_phi"] == "unreachable":
            continue
        measure_key = (
            int(integration_rows[row]["size"]),
            integration_rows[row]["initialisation"],
            integration_rows[row]["task"],
            int(integration_rows[row]["delay"]),
            int(integration_rows[row]["seed"]),
            int(integration_rows[row]["epoch"]),
            integration_rows[row]["shuffle"],
        )
        sums.setdefault(measure_key, [0, 0, 0, 0])
        sums[measure_key][0] += float(integration_rows[row]["small_phi"])
        sums[measure_key][1] += 1
        if integration_rows[row]["big_phi"] != "":
            sums[measure_key][2] += float(integration_rows[row]["big_phi"])
            sums[measure_key][3] += 1
    measure_keys = list(sums)
    for measure in range(len(measure_keys)):
        runs.setdefault(measure_keys[measure][:5], {}).setdefault(
            measure_keys[measure][5:], {}
        )
        runs[measure_keys[measure][:5]][measure_keys[measure][5:]]["small_phi"] = (
            sums[measure_keys[measure]][0] / sums[measure_keys[measure]][1]
        )
        if sums[measure_keys[measure]][3] > 0:
            runs[measure_keys[measure][:5]][measure_keys[measure][5:]]["big_phi"] = (
                sums[measure_keys[measure]][2] / sums[measure_keys[measure]][3]
            )
accuracy_files = sorted(glob.glob(os.path.join("data", "training_accuracy*.csv")))
for accuracy_file_index in range(len(accuracy_files)):
    accuracy_file = open(accuracy_files[accuracy_file_index], newline="")
    accuracy_rows = list(csv.DictReader(accuracy_file))
    accuracy_file.close()
    for row in range(len(accuracy_rows)):
        runs.setdefault(
            (
                int(accuracy_rows[row]["size"]),
                accuracy_rows[row]["initialisation"],
                accuracy_rows[row]["task"],
                int(accuracy_rows[row]["delay"]),
                int(accuracy_rows[row]["seed"]),
            ),
            {},
        ).setdefault(
            (int(accuracy_rows[row]["epoch"]), accuracy_rows[row]["shuffle"]), {}
        )[
            "accuracy"
        ] = float(
            accuracy_rows[row]["accuracy"]
        )
runs_keys = list(runs)
sizes = []
present_tasks = []
delays = []
for run in range(len(runs_keys)):
    if runs_keys[run][0] not in sizes:
        sizes.append(runs_keys[run][0])
    if runs_keys[run][2] not in present_tasks:
        present_tasks.append(runs_keys[run][2])
    if runs_keys[run][3] not in delays:
        delays.append(runs_keys[run][3])
sizes.sort()
delays.sort()
ordered_tasks = []
for task in range(len(tasks)):
    if tasks[task] in present_tasks:
        ordered_tasks.append(tasks[task])
finished_runs = []
for run in range(len(runs_keys)):
    shuffles_quantity = 0
    measures_keys = list(runs[runs_keys[run]])
    for measure in range(len(measures_keys)):
        if measures_keys[measure][1] != "none":
            shuffles_quantity += 1
    if (epochs, "none") in runs[runs_keys[run]] and shuffles_quantity == shuffles:
        finished_runs.append(runs_keys[run])
controls = {}
controls_counts = {}
for run in range(len(finished_runs)):
    for metric in range(len(metrics)):
        control_key = (
            finished_runs[run][0],
            finished_runs[run][1],
            finished_runs[run][2],
            metrics[metric],
        )
        controls.setdefault(control_key, [0, 0, 0])
        controls_counts.setdefault(control_key, [0, 0, 0])
        if metrics[metric] in runs[finished_runs[run]].get((0, "none"), {}):
            controls[control_key][0] += runs[finished_runs[run]][(0, "none")][
                metrics[metric]
            ]
            controls_counts[control_key][0] += 1
        if metrics[metric] in runs[finished_runs[run]][(epochs, "none")]:
            controls[control_key][1] += runs[finished_runs[run]][(epochs, "none")][
                metrics[metric]
            ]
            controls_counts[control_key][1] += 1
        measures_keys = list(runs[finished_runs[run]])
        for measure in range(len(measures_keys)):
            if (
                measures_keys[measure][1] != "none"
                and metrics[metric] in runs[finished_runs[run]][measures_keys[measure]]
            ):
                controls[control_key][2] += runs[finished_runs[run]][
                    measures_keys[measure]
                ][metrics[metric]]
                controls_counts[control_key][2] += 1
controls_keys = list(controls)
for control in range(len(controls_keys)):
    for column in range(len(bar_columns)):
        if controls_counts[controls_keys[control]][column] > 0:
            controls[controls_keys[control]][column] /= controls_counts[
                controls_keys[control]
            ][column]
        else:
            controls[controls_keys[control]][column] = float("nan")
summary_file = open(os.path.join("plots", "summary.csv"), "w", newline="")
summary_writer = csv.writer(summary_file)
summary_writer.writerow(
    [
        "size",
        "initialisation",
        "task",
        "metric",
        "before",
        "after",
        "shuffled",
        "finished_runs",
    ]
)
for control in range(len(controls_keys)):
    summary_writer.writerow(
        list(controls_keys[control])
        + controls[controls_keys[control]]
        + [controls_counts[controls_keys[control]][1]]
    )
summary_file.close()

for language in range(len(languages)):
    text = texts[languages[language]]

    if len(atlas_values) > 0:
        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        for metric in range(2):
            axis_values = []
            for initialisation in range(len(initialisations)):
                atlas_sizes = []
                atlas_medians = []
                atlas_lower_errors = []
                atlas_upper_errors = []
                for size in range(2, 6):
                    if (
                        size,
                        initialisations[initialisation],
                        metrics[metric],
                    ) not in atlas_values:
                        continue
                    values = sorted(
                        atlas_values[
                            (size, initialisations[initialisation], metrics[metric])
                        ]
                    )
                    atlas_sizes.append(size)
                    atlas_medians.append(values[len(values) // 2])
                    atlas_lower_errors.append(
                        values[len(values) // 2] - values[len(values) // 4]
                    )
                    atlas_upper_errors.append(
                        values[(3 * len(values)) // 4] - values[len(values) // 2]
                    )
                    for value in range(len(values)):
                        axis_values.append(values[value])
                axes[metric].errorbar(
                    atlas_sizes,
                    atlas_medians,
                    yerr=[atlas_lower_errors, atlas_upper_errors],
                    marker="o",
                    capsize=4,
                    label=text[initialisations[initialisation]],
                )
            apply_adaptive_scale(axes[metric], axis_values)
            axes[metric].set_xlabel(text["size"])
            axes[metric].set_ylabel(text[metrics[metric]])
            axes[metric].grid(alpha=0.3)
            axes[metric].legend()
        figure.suptitle(f"{text['atlas_title']} ({text['median_note']})")
        figure.tight_layout()
        figure.savefig(
            os.path.join(
                "plots", languages[language], "1_atlas_integration_by_size.png"
            ),
            dpi=150,
        )
        plt.close(figure)
    if len(runs_keys) == 0:
        continue
    for metric in range(len(metrics)):
        figure, axes = plt.subplots(
            len(sizes), 2, figsize=(12, 3.3 * len(sizes)), squeeze=False
        )
        for size in range(len(sizes)):
            for initialisation in range(len(initialisations)):
                axis_values = []
                for task in range(len(ordered_tasks)):
                    epochs_sums = {}
                    for run in range(len(runs_keys)):
                        if (
                            runs_keys[run][0] != sizes[size]
                            or runs_keys[run][1] != initialisations[initialisation]
                            or runs_keys[run][2] != ordered_tasks[task]
                        ):
                            continue
                        measures_keys = list(runs[runs_keys[run]])
                        for measure in range(len(measures_keys)):
                            if (
                                measures_keys[measure][1] != "none"
                                or metrics[metric]
                                not in runs[runs_keys[run]][measures_keys[measure]]
                            ):
                                continue
                            epochs_sums.setdefault(measures_keys[measure][0], [0, 0])
                            epochs_sums[measures_keys[measure][0]][0] += runs[
                                runs_keys[run]
                            ][measures_keys[measure]][metrics[metric]]
                            epochs_sums[measures_keys[measure][0]][1] += 1
                    curve_epochs = sorted(epochs_sums)
                    curve_values = []
                    for curve_epoch in range(len(curve_epochs)):
                        curve_values.append(
                            epochs_sums[curve_epochs[curve_epoch]][0]
                            / epochs_sums[curve_epochs[curve_epoch]][1]
                        )
                        axis_values.append(curve_values[curve_epoch])
                    if len(curve_epochs) > 0:
                        axes[size][initialisation].plot(
                            curve_epochs,
                            curve_values,
                            marker=".",
                            label=text[ordered_tasks[task]],
                        )
                axes[size][initialisation].set_title(
                    f"{sizes[size]} {text['nodes']}, {text[initialisations[initialisation]]}",
                    fontsize=10,
                )
                apply_adaptive_scale(axes[size][initialisation], axis_values)
                axes[size][initialisation].set_xlabel(text["epoch"])
                axes[size][initialisation].set_ylabel(text[metrics[metric]])
                axes[size][initialisation].grid(alpha=0.3)
        axes[0][1].legend(fontsize=8, loc="best")
        figure.suptitle(f"{text['training_title']}: {text[metrics[metric]]}")
        figure.tight_layout()
        figure.savefig(
            os.path.join(
                "plots", languages[language], f"2_training_{metrics[metric]}.png"
            ),
            dpi=150,
        )
        plt.close(figure)

        figure, axes = plt.subplots(
            len(sizes), 2, figsize=(12, 3.3 * len(sizes)), squeeze=False
        )
        for size in range(len(sizes)):
            for initialisation in range(len(initialisations)):
                axis_values = []
                for column in range(len(bar_columns)):
                    bars_positions = []
                    bars_heights = []
                    for task in range(len(ordered_tasks)):
                        bars_positions.append(task + (column - 1) * 0.27)
                        if (
                            sizes[size],
                            initialisations[initialisation],
                            ordered_tasks[task],
                            metrics[metric],
                        ) in controls:
                            bars_heights.append(
                                controls[
                                    (
                                        sizes[size],
                                        initialisations[initialisation],
                                        ordered_tasks[task],
                                        metrics[metric],
                                    )
                                ][column]
                            )
                        else:
                            bars_heights.append(float("nan"))
                        axis_values.append(bars_heights[task])
                    axes[size][initialisation].bar(
                        bars_positions,
                        bars_heights,
                        width=0.27,
                        label=text[bar_columns[column]],
                    )
                tasks_labels = []
                for task in range(len(ordered_tasks)):
                    tasks_labels.append(text[ordered_tasks[task]])
                axes[size][initialisation].set_xticks(range(len(ordered_tasks)))
                axes[size][initialisation].set_xticklabels(
                    tasks_labels, rotation=30, ha="right", fontsize=8
                )
                axes[size][initialisation].set_title(
                    f"{sizes[size]} {text['nodes']}, {text[initialisations[initialisation]]}",
                    fontsize=10,
                )
                axes[size][initialisation].set_ylabel(text[metrics[metric]])
                axes[size][initialisation].grid(alpha=0.3, axis="y")
                apply_adaptive_scale(axes[size][initialisation], axis_values)
        axes[0][1].legend(fontsize=8)
        figure.suptitle(f"{text['controls_title']}: {text[metrics[metric]]}")
        figure.tight_layout()
        figure.savefig(
            os.path.join(
                "plots", languages[language], f"3_controls_{metrics[metric]}.png"
            ),
            dpi=150,
        )
        plt.close(figure)

        figure, axes = plt.subplots(
            len(sizes), 2, figsize=(12, 3.3 * len(sizes)), squeeze=False
        )
        for size in range(len(sizes)):
            for initialisation in range(len(initialisations)):
                axis_values = []
                for task in range(len(ordered_tasks)):
                    curve_delays = []
                    curve_values = []
                    for delay in range(len(delays)):
                        value_sum = 0
                        value_count = 0
                        for run in range(len(finished_runs)):
                            if (
                                finished_runs[run][0] == sizes[size]
                                and finished_runs[run][1]
                                == initialisations[initialisation]
                                and finished_runs[run][2] == ordered_tasks[task]
                                and finished_runs[run][3] == delays[delay]
                                and metrics[metric]
                                in runs[finished_runs[run]][(epochs, "none")]
                            ):
                                value_sum += runs[finished_runs[run]][(epochs, "none")][
                                    metrics[metric]
                                ]
                                value_count += 1
                        if value_count > 0:
                            curve_delays.append(delays[delay])
                            curve_values.append(value_sum / value_count)
                            axis_values.append(value_sum / value_count)
                    if len(curve_delays) > 0:
                        axes[size][initialisation].plot(
                            curve_delays,
                            curve_values,
                            marker="o",
                            label=text[ordered_tasks[task]],
                        )
                axes[size][initialisation].set_title(
                    f"{sizes[size]} {text['nodes']}, {text[initialisations[initialisation]]}",
                    fontsize=10,
                )
                apply_adaptive_scale(axes[size][initialisation], axis_values)
                axes[size][initialisation].set_xlabel(text["delay"])
                axes[size][initialisation].set_ylabel(text[metrics[metric]])
                axes[size][initialisation].grid(alpha=0.3)
        axes[0][1].legend(fontsize=8)
        figure.suptitle(f"{text['delay_title']}: {text[metrics[metric]]}")
        figure.tight_layout()
        figure.savefig(
            os.path.join(
                "plots", languages[language], f"4_delay_{metrics[metric]}.png"
            ),
            dpi=150,
        )
        plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for metric in range(2):
        axis_values = []
        for initialisation in range(len(initialisations)):
            for size in range(len(sizes)):
                scatter_x = []
                scatter_y = []
                for run in range(len(finished_runs)):
                    if (
                        finished_runs[run][0] == sizes[size]
                        and finished_runs[run][1] == initialisations[initialisation]
                        and metrics[metric]
                        in runs[finished_runs[run]][(epochs, "none")]
                        and "accuracy" in runs[finished_runs[run]][(epochs, "none")]
                    ):
                        scatter_x.append(
                            runs[finished_runs[run]][(epochs, "none")][metrics[metric]]
                        )
                        scatter_y.append(
                            runs[finished_runs[run]][(epochs, "none")]["accuracy"]
                        )
                        axis_values.append(
                            runs[finished_runs[run]][(epochs, "none")][metrics[metric]]
                        )
                if len(scatter_x) > 0:
                    axes[metric].scatter(
                        scatter_x,
                        scatter_y,
                        s=18,
                        alpha=0.6,
                        marker=["o", "s", "^", "D"][size % 4],
                        color=["tab:blue", "tab:orange"][initialisation],
                        label=f"{text[initialisations[initialisation]]}, {sizes[size]} {text['nodes']}",
                    )
        apply_adaptive_scale(axes[metric], axis_values, horizontal=True)
        axes[metric].set_xlabel(text[metrics[metric]])
        axes[metric].set_ylabel(text["accuracy"])
        axes[metric].grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    figure.suptitle(text["scatter_title"])
    figure.tight_layout()
    figure.savefig(
        os.path.join("plots", languages[language], "5_accuracy_versus_integration.png"),
        dpi=150,
    )
    plt.close(figure)
