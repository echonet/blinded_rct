import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas
import scipy.stats


def main():
    # Loading data
    data = pandas.read_csv("data.csv")
    sono = (data["ActualArm"] == "Sonographer")
    ai = (data["ActualArm"] == "AI")
    both = (sono | ai)
    assert both.all()
    arms = [both, ai, sono]

    # Table 1
    def count(f, stat, values=None):
        f.write(stat + ",\n")
        if values is None:
            values = list(zip(*sorted(collections.Counter(data[stat]).items(), key=lambda x: x[1])))[0][::-1]
        for x in values:
            f.write("    {},{},{},{}\n".format(
                x, *["{} ({:.0f}%)".format((data[a][stat] == x).sum(), 100 * (data[a][stat] == x).sum() / a.sum()) for a in arms]
            ))

    with open("characteristics.csv", "w") as f:
        f.write("Variable,Total (n = {}),AI (n = {}),Sonographer (n = {})\n".format(
            *[a.sum() for a in arms]
        ))

        f.write("Prior Clinical EF,{},{},{}\n".format(
            *["{:.1f} ± {:.1f}".format(data[a]["PriorClinicalEF"].mean(), data[a]["PriorClinicalEF"].std()) for a in arms]
        ))
        count(f, "MethodOfLVEFEvaluation")
        count(f, "StudyQuality", ["Poor", "Adequate", "Good", "Not Specified"])
        count(f, "Location")

    # Table 2
    arms = [ai, sono]
    with open("outcomes.csv", "w") as f:
        f.write(",AI (n = {}),Sonographer (n = {}),Difference (95% CI),p-value\n".format(*[a.sum() for a in arms]))

        f.write("Primary Outcomes,\n")
        x = [greater_than(data[data["ActualArm"] == arm][["InitialEF", "FinalEF"]].values) for arm in ["AI", "Sonographer"]]
        arm_v = [(i, 100 * j) for i in [0, 1] for j in x[i]]
        ci = scipy.stats.bootstrap((arm_v,), group_diff, method="percentile")
        f.write("    Substantial Change,{} ({:.1f}%),{} ({:.1f}%),{:.1f} ({:.1f} - {:.1f}),{}\n".format(
            x[0].sum(), 100 * x[0].mean(), x[1].sum(), 100 * x[1].mean(),
            group_diff(np.expand_dims(np.array(arm_v).transpose(), 1), None)[0],
            ci.confidence_interval.low,
            ci.confidence_interval.high,
            scipy.stats.fisher_exact([[x[0].sum(), x[1].sum()], [(~x[0]).sum(), (~x[1]).sum()]])[1]))

        x = [mean_absolute_difference(data[data["ActualArm"] == arm][["InitialEF", "FinalEF"]].values) for arm in ["AI", "Sonographer"]]
        arm_v = [(i, j) for i in [0, 1] for j in x[i]]
        ci = scipy.stats.bootstrap((arm_v,), group_diff, method="percentile")
        f.write("    Mean Absolute Difference,{:.2f} ± {:.2f},{:.2f} ± {:.2f},{:.2f} ({:.2f} - {:.2f}),{}\n".format(
            x[0].mean(), x[0].std(), x[1].mean(), x[1].std(),
            group_diff(np.expand_dims(np.array(arm_v).transpose(), 1), None)[0],
            ci.confidence_interval.low,
            ci.confidence_interval.high,
            scipy.stats.ttest_ind(x[0], x[1], equal_var=False).pvalue))

        f.write("Secondary Outcomes,\n")
        x = [greater_than(data[data["ActualArm"] == arm][["FinalEF", "PriorClinicalEF"]].values) for arm in ["AI", "Sonographer"]]
        arm_v = [(i, 100 * j) for i in [0, 1] for j in x[i]]
        ci = scipy.stats.bootstrap((arm_v,), group_diff, method="percentile")
        f.write("    Substantial Change,{} ({:.1f}%),{} ({:.1f}%),{:.1f} ({:.1f} - {:.1f}),{}\n".format(
            x[0].sum(), 100 * x[0].mean(), x[1].sum(), 100 * x[1].mean(),
            group_diff(np.expand_dims(np.array(arm_v).transpose(), 1), None)[0],
            ci.confidence_interval.low,
            ci.confidence_interval.high,
            scipy.stats.fisher_exact([[x[0].sum(), x[1].sum()], [(~x[0]).sum(), (~x[1]).sum()]])[1]))

        x = [mean_absolute_difference(data[data["ActualArm"] == arm][["FinalEF", "PriorClinicalEF"]].values) for arm in ["AI", "Sonographer"]]
        arm_v = [(i, j) for i in [0, 1] for j in x[i]]
        ci = scipy.stats.bootstrap((arm_v,), group_diff, method="percentile")
        f.write("    Mean Absolute Difference,{:.2f} ± {:.2f},{:.2f} ± {:.2f},{:.2f} ({:.2f} - {:.2f}),{}\n".format(
            x[0].mean(), x[0].std(), x[1].mean(), x[1].std(),
            group_diff(np.expand_dims(np.array(arm_v).transpose(), 1), None)[0],
            ci.confidence_interval.low,
            ci.confidence_interval.high,
            scipy.stats.ttest_ind(x[0], x[1], equal_var=False).pvalue))

        x = [changed(data[data["ActualArm"] == arm][["InitialEF", "FinalEF"]].values) for arm in ["AI", "Sonographer"]]
        arm_v = [(i, 100 * j) for i in [0, 1] for j in x[i]]
        ci = scipy.stats.bootstrap((arm_v,), group_diff, method="percentile")
        f.write("Any Change,{} ({:.1f}%),{} ({:.1f}%),{:.1f} ({:.1f} - {:.1f}),{}\n".format(
            x[0].sum(), 100 * x[0].mean(), x[1].sum(), 100 * x[1].mean(),
            group_diff(np.expand_dims(np.array(arm_v).transpose(), 1), None)[0],
            ci.confidence_interval.low,
            ci.confidence_interval.high,
            scipy.stats.fisher_exact([[x[0].sum(), x[1].sum()], [(~x[0]).sum(), (~x[1]).sum()]])[1]))

        x = [across(data[data["ActualArm"] == arm][["InitialEF", "FinalEF"]].values) for arm in ["AI", "Sonographer"]]
        f.write("Changed across 35,{} ({:.1f}%),{} ({:.1f}%),{}\n".format(x[0].sum(), 100 * x[0].mean(), x[1].sum(), 100 * x[1].mean(), scipy.stats.fisher_exact([[x[0].sum(), x[1].sum()], [(~x[0]).sum(), (~x[1]).sum()]])[1]))

    # Table 3
    arms = [ai, sono]

    def compare(f, stat, values=None):
        if values is None:
            values = list(zip(*sorted(collections.Counter(data[stat]).items(), key=lambda x: x[1])))[0][::-1]
        for x in values:
            f.write(x + ",")

            for a in arms:
                ef = data[a & (data[stat] == x)][["InitialEF", "FinalEF"]].values
                error = np.abs(ef[:, 0] - ef[:, 1])
                error = np.array(sorted(error))
                f.write("{},{:.2f} ± {:.2f},".format(len(error), np.mean(error), np.std(error)))

            t = data[data[stat] == x]
            arm_error = list(zip(t["ActualArm"] == "Sonographer", np.abs((t["InitialEF"] - t["FinalEF"]).values)))
            ci = scipy.stats.bootstrap((arm_error,), group_diff, method="percentile")
            f.write("{:.2f} ({:.2f} - {:.2f})".format(
                group_diff(np.expand_dims(np.array(arm_error).transpose(), 1), None)[0],
                ci.confidence_interval.low,
                ci.confidence_interval.high,
            ))
            f.write("\n")

    with open("subgroup.csv", "w") as f:
        f.write("Subgroup,AI (n), AI (MAD), Sonographer (n), Sonographer (MAD), Difference (95% confidence interval)\n")
        compare(f, "MethodOfLVEFEvaluation")
        compare(f, "StudyQuality", ["Poor", "Adequate", "Good", "Not Specified"])
        compare(f, "Location")
        compare(f, "CardiologistPredictedArm", ["AI", "Sonographer", "Uncertain"])

        f.write("Correct,")
        for a in arms:
            ef = data[a & (data["ActualArm"] == data["CardiologistPredictedArm"]) & (data["CardiologistPredictedArm"] != "Uncertain")][["InitialEF", "FinalEF"]].values
            error = np.abs(ef[:, 0] - ef[:, 1])
            error = np.array(sorted(error))
            f.write("{},{:.2f} ± {:.2f},".format(len(error), np.mean(error), np.std(error)))
        t = data[(data["ActualArm"] == data["CardiologistPredictedArm"]) & (data["CardiologistPredictedArm"] != "Uncertain")]
        arm_error = list(zip(t["ActualArm"] == "Sonographer", np.abs((t["InitialEF"] - t["FinalEF"]).values)))
        ci = scipy.stats.bootstrap((arm_error,), group_diff, method="percentile")
        f.write("{:.2f} ({:.2f} - {:.2f})".format(
            group_diff(np.expand_dims(np.array(arm_error).transpose(), 1), None)[0],
            ci.confidence_interval.low,
            ci.confidence_interval.high,
        ))
        f.write("\n")

        f.write("Incorrect,")
        for a in arms:
            ef = data[a & (data["ActualArm"] != data["CardiologistPredictedArm"]) & (data["CardiologistPredictedArm"] != "Uncertain")][["InitialEF", "FinalEF"]].values
            error = np.abs(ef[:, 0] - ef[:, 1])
            error = np.array(sorted(error))
            f.write("{},{:.2f} ± {:.2f},".format(len(error), np.mean(error), np.std(error)))
        t = data[(data["ActualArm"] != data["CardiologistPredictedArm"]) & (data["CardiologistPredictedArm"] != "Uncertain")]
        arm_error = list(zip(t["ActualArm"] == "Sonographer", np.abs((t["InitialEF"] - t["FinalEF"]).values)))
        ci = scipy.stats.bootstrap((arm_error,), group_diff, method="percentile")
        f.write("{:.2f} ({:.2f} - {:.2f})".format(
            group_diff(np.expand_dims(np.array(arm_error).transpose(), 1), None)[0],
            ci.confidence_interval.low,
            ci.confidence_interval.high,
        ))
        f.write("\n")

    # Fig. 2
    fig, ax = plt.subplots(2, 2, figsize=(5, 5))
    for (i, j, x, y, xlabel, ylabel, color) in [
        (0, 0, data[ai]["InitialEF"], data[ai]["FinalEF"], "AI initial assessment (%)", "Final cardiologist assessment (%)", "red"),
        (0, 1, data[sono]["InitialEF"], data[sono]["FinalEF"], "Sonographer initial assessment (%)", "Final cardiologist assessment (%)", "blue"),
        (1, 0, data[ai]["FinalEF"], data[ai]["PriorClinicalEF"], "Cardiologist with AI guidance (%)", "Historical cardiologist assessment (%)", "red"),
        (1, 1, data[sono]["FinalEF"], data[sono]["PriorClinicalEF"], "Cardiologist with sonographer guidance (%)", "Historical cardiologist assessment (%)", "blue"),
    ]:
        ax[i, j].scatter(x, y, s=0.5, color=color, alpha=0.3)
        ax[i, j].set_xlim([-10, 110])
        ax[i, j].set_ylim([-10, 110])
        ax[i, j].set_xticks([0, 25, 50, 75, 100])
        ax[i, j].set_yticks([0, 25, 50, 75, 100])
        ax[i, j].set_xlabel(xlabel)
        ax[i, j].set_ylabel(ylabel)
        a, b = np.polyfit(x, y, 1)
        ax[i, j].plot([0, 100], [b, 100 * a + b], linewidth=1, color="black")
        ax[i, j].text(70, 15, "MAD {:.2f}%".format(np.mean(np.abs(x - y))))
    fig.savefig("comparison.pdf")

    # Bang's blinding index
    count = collections.Counter(map(tuple, data[["ActualArm", "CardiologistPredictedArm"]].values))
    n = np.array([[count[real, pred] for pred in ["AI", "Sonographer", "Uncertain"]] for real in ["AI", "Sonographer"]])

    def bootstrap(n, metric, samples=1000):
        ans = []
        ij = []
        for i in range(n.shape[0]):
            for j in range(n.shape[1]):
                ij.extend([(i, j)] * n[i, j])
        ij = np.array(ij)
        for _ in range(samples):
            n_hat = np.zeros_like(n)
            for (i, j) in ij[np.random.randint(n.sum(), size=n.sum())]:
                n_hat[i, j] += 1
            ans.append(metric(n_hat))
        ans = list(map(sorted, zip(*ans)))
        return list(map(lambda x: (x[round(0.025 * len(x))], x[round(0.975 * len(x))]), ans))

    def bang_blinding_index(n):
        ans = []
        for i in [0, 1]:
            rhatii = n[i, i] / (n[i, 0] + n[i, 1])
            ans.append((2 * rhatii - 1) * (n[i, 0] + n[i, 1]) / (n[i, 0] + n[i, 1] + n[i, 2]))
        return ans

    (bbi_sono, bbi_ai) = bang_blinding_index(n)
    [(bbi_sono_lower, bbi_sono_upper), (bbi_ai_lower, bbi_ai_upper)] = bootstrap(n, bang_blinding_index)
    print("Bang's blinding index: {:.3f} ({:.3f} - {:.3f}), {:.3f} ({:.3f} - {:.3f})".format(
        bbi_sono, bbi_sono_lower, bbi_sono_upper, bbi_ai, bbi_ai_lower, bbi_ai_upper))


def group_diff(x, axis):
    ai_mad = (x[0, :, :] * x[1, :, :]).sum(1) / x[0, :, :].sum(1)
    sono_mad = ((1 - x[0, :, :]) * x[1, :, :]).sum(1) / (1 - x[0, :, :]).sum(1)
    return sono_mad - ai_mad


def greater_than(x, thresh=5):
    return np.abs(x[:, 0] - x[:, 1]) > thresh


def mean_absolute_difference(x):
    return np.abs(x[:, 0] - x[:, 1])


def changed(x):
    return x[:, 0] != x[:, 1]


def across(x, thresh=35):
    return (x[:, 0] > thresh) != (x[:, 1] > thresh)


if __name__ == "__main__":
    main()
