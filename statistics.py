import csv
import glob
import os
from scipy import stats

epochs = 16384
memory_tasks = ["replay", "replay_twice"]
metrics = ["small_phi", "big_phi"]
initialisations = ["ring", "random"]

networks = {}

integration_files = sorted(
    glob.glob(os.path.join("data", "training_integration_*.csv"))
)
for integration_file_index in range(len(integration_files)):
    integration_file = open(integration_files[integration_file_index], newline="")
    integration_rows = list(csv.DictReader(integration_file))
    integration_file.close()
    for row in range(len(integration_rows)):
        if integration_rows[row]["small_phi"] == "unreachable":
            continue
        moment = "shuffled"
        if integration_rows[row]["shuffle"] == "none":
            moment = "before"
            if int(integration_rows[row]["epoch"]) == epochs:
                moment = "after"
            elif int(integration_rows[row]["epoch"]) != 0:
                continue
        network_key = (
            int(integration_rows[row]["size"]),
            integration_rows[row]["initialisation"],
            integration_rows[row]["task"],
            int(integration_rows[row]["delay"]),
            int(integration_rows[row]["seed"]),
        )
        networks.setdefault(network_key, {})
        for metric in range(len(metrics)):
            if integration_rows[row][metrics[metric]] == "":
                continue
            networks[network_key].setdefault((moment, metrics[metric]), [0, 0])
            networks[network_key][(moment, metrics[metric])][0] += float(
                integration_rows[row][metrics[metric]]
            )
            networks[network_key][(moment, metrics[metric])][1] += 1
accuracy_files = sorted(glob.glob(os.path.join("data", "training_accuracy_*.csv")))
for accuracy_file_index in range(len(accuracy_files)):
    accuracy_file = open(accuracy_files[accuracy_file_index], newline="")
    accuracy_rows = list(csv.DictReader(accuracy_file))
    accuracy_file.close()
    for row in range(len(accuracy_rows)):
        moment = "shuffled"
        if accuracy_rows[row]["shuffle"] == "none":
            moment = "before"
            if int(accuracy_rows[row]["epoch"]) == epochs:
                moment = "after"
            elif int(accuracy_rows[row]["epoch"]) != 0:
                continue
        network_key = (
            int(accuracy_rows[row]["size"]),
            accuracy_rows[row]["initialisation"],
            accuracy_rows[row]["task"],
            int(accuracy_rows[row]["delay"]),
            int(accuracy_rows[row]["seed"]),
        )
        networks.setdefault(network_key, {})
        networks[network_key].setdefault((moment, "accuracy"), [0, 0])
        networks[network_key][(moment, "accuracy")][0] += float(
            accuracy_rows[row]["accuracy"]
        )
        networks[network_key][(moment, "accuracy")][1] += 1
networks_keys = sorted(networks)
for network in range(len(networks_keys)):
    values_keys = list(networks[networks_keys[network]])
    for value in range(len(values_keys)):
        networks[networks_keys[network]][values_keys[value]] = (
            networks[networks_keys[network]][values_keys[value]][0]
            / networks[networks_keys[network]][values_keys[value]][1]
        )
print("Networks found", len(networks_keys))
print()

print("1. Training, before versus after (paired Wilcoxon signed-rank test)")
for initialisation in range(len(initialisations)):
    for metric in range(len(metrics)):
        before = []
        after = []
        for network in range(len(networks_keys)):
            if networks_keys[network][1] != initialisations[initialisation]:
                continue
            before.append(networks[networks_keys[network]][("before", metrics[metric])])
            after.append(networks[networks_keys[network]][("after", metrics[metric])])
        decreased = 0
        for network in range(len(before)):
            if after[network] < before[network]:
                decreased += 1
        test = stats.wilcoxon(before, after)
        print(
            f"   {initialisations[initialisation]:6} {metrics[metric]:9} "
            f"decreased in {decreased} of {len(before)} "
            f"({100 * decreased / len(before):.1f}%), p = {test.pvalue:.1e}"
        )
before = []
after = []
for network in range(len(networks_keys)):
    if networks_keys[network][1] != "random":
        continue
    if networks_keys[network][2] not in memory_tasks:
        continue
    if networks_keys[network][3] > 1:
        continue
    before.append(networks[networks_keys[network]][("before", "small_phi")])
    after.append(networks[networks_keys[network]][("after", "small_phi")])
increased = 0
for network in range(len(before)):
    if after[network] > before[network]:
        increased += 1
test = stats.wilcoxon(before, after)
print(
    f"   random small_phi in memory tasks at delays 0 and 1 "
    f"increased in {increased} of {len(before)} "
    f"({100 * increased / len(before):.1f}%), p = {test.pvalue:.1e}"
)
print()

print("2. After training versus after shuffling (paired Wilcoxon signed-rank test)")
for initialisation in range(len(initialisations)):
    for metric in range(len(metrics) + 1):
        metric_name = "accuracy"
        if metric < len(metrics):
            metric_name = metrics[metric]
        after = []
        shuffled = []
        for network in range(len(networks_keys)):
            if networks_keys[network][1] != initialisations[initialisation]:
                continue
            after.append(networks[networks_keys[network]][("after", metric_name)])
            shuffled.append(networks[networks_keys[network]][("shuffled", metric_name)])
        lower = 0
        for network in range(len(after)):
            if shuffled[network] < after[network]:
                lower += 1
        test = stats.wilcoxon(after, shuffled)
        print(
            f"   {initialisations[initialisation]:6} {metric_name:9} "
            f"lower after shuffling in {lower} of {len(after)} "
            f"({100 * lower / len(after):.1f}%), p = {test.pvalue:.1e}"
        )
print()

print("3. Accuracy versus integration at the end of training")
for initialisation in range(len(initialisations)):
    for metric in range(len(metrics)):
        accuracy = []
        integration = []
        for network in range(len(networks_keys)):
            if networks_keys[network][1] != initialisations[initialisation]:
                continue
            accuracy.append(networks[networks_keys[network]][("after", "accuracy")])
            integration.append(
                networks[networks_keys[network]][("after", metrics[metric])]
            )
        pearson = stats.pearsonr(accuracy, integration)
        spearman = stats.spearmanr(accuracy, integration)
        print(
            f"   {initialisations[initialisation]:6} {metrics[metric]:9} "
            f"{len(accuracy)} networks, "
            f"r = {pearson.statistic:.2f} (p = {pearson.pvalue:.1e}), "
            f"rho = {spearman.statistic:.2f} (p = {spearman.pvalue:.1e})"
        )
for metric in range(len(metrics)):
    accuracy = []
    integration = []
    for network in range(len(networks_keys)):
        if networks_keys[network][1] != "random":
            continue
        if networks_keys[network][2] not in memory_tasks:
            continue
        accuracy.append(networks[networks_keys[network]][("after", "accuracy")])
        integration.append(networks[networks_keys[network]][("after", metrics[metric])])
    spearman = stats.spearmanr(accuracy, integration)
    print(
        f"   random {metrics[metric]:9} memory tasks only, {len(accuracy)} networks, "
        f"rho = {spearman.statistic:.2f} (p = {spearman.pvalue:.1e})"
    )
sizes = []
for network in range(len(networks_keys)):
    if networks_keys[network][0] not in sizes:
        sizes.append(networks_keys[network][0])
sizes.sort()
for size in range(len(sizes)):
    accuracy = []
    integration = []
    for network in range(len(networks_keys)):
        if networks_keys[network][1] != "ring":
            continue
        if networks_keys[network][0] != sizes[size]:
            continue
        accuracy.append(networks[networks_keys[network]][("after", "accuracy")])
        integration.append(networks[networks_keys[network]][("after", "small_phi")])
    spearman = stats.spearmanr(accuracy, integration)
    print(
        f"   ring   small_phi {sizes[size]} nodes, {len(accuracy)} networks, "
        f"rho = {spearman.statistic:.2f} (p = {spearman.pvalue:.1e})"
    )
print()

print(
    "4. Gain in accuracy versus change of small phi during training, without the control task"
)
for initialisation in range(len(initialisations)):
    accuracy_gain = []
    integration_change = []
    for network in range(len(networks_keys)):
        if networks_keys[network][1] != initialisations[initialisation]:
            continue
        if networks_keys[network][2] == "noise":
            continue
        accuracy_gain.append(
            networks[networks_keys[network]][("after", "accuracy")]
            - networks[networks_keys[network]][("before", "accuracy")]
        )
        integration_change.append(
            networks[networks_keys[network]][("after", "small_phi")]
            - networks[networks_keys[network]][("before", "small_phi")]
        )
    spearman = stats.spearmanr(accuracy_gain, integration_change)
    print(
        f"   {initialisations[initialisation]:6} {len(accuracy_gain)} networks, "
        f"rho = {spearman.statistic:.2f} (p = {spearman.pvalue:.1e})"
    )
