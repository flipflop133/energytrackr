(() => {
    const data_all = full.data;
    const raw_all = raw.data;
    const data_box = box.data;
    const sc1 = scatter1.data;
    const sc2 = scatter2.data;
    const line_data = line.data;
    const commits = data_all["commit"];
    const c1 = sel1.value;
    const c2 = sel2.value;
    const sin = inlier_scatter.data;
    const sout = outlier_scatter.data;
    const i1 = commits.indexOf(c1);
    const i2 = commits.indexOf(c2);

    // --- update box stats (use "median" instead of "q2") ---
    for (const f of ["commit", "q1", "median", "q3", "lower", "upper", "n_low", "n_high"]) {
        data_box[f] = [data_all[f][i1], data_all[f][i2]];
    }
    // (colors live in data_box["color"] already)
    box.change.emit();

    // --- update scatter points (use "values" / "value" fields) ---
    const v1 = raw_all["values"][i1];
    const v2 = raw_all["values"][i2];
    sc1["commit"] = Array(v1.length).fill(c1);
    sc1["value"] = v1;
    sc2["commit"] = Array(v2.length).fill(c2);
    sc2["value"] = v2;
    scatter1.change.emit();
    scatter2.change.emit();

    // --- split into inliers vs outliers by whisker bounds ---
    const l1 = data_box["lower"][0], u1 = data_box["upper"][0];
    const l2 = data_box["lower"][1], u2 = data_box["upper"][1];
    const norm1 = v1.filter(v => v >= l1 && v <= u1);
    const out1 = v1.filter(v => v < l1 || v > u1);
    const norm2 = v2.filter(v => v >= l2 && v <= u2);
    const out2 = v2.filter(v => v < l2 || v > u2);

    // update inlier‐scatter source (small, grey points)
    sin["commit"] = [
        ...Array(norm1.length).fill(c1),
        ...Array(norm2.length).fill(c2),
    ];
    sin["value"] = [...norm1, ...norm2];
    inlier_scatter.change.emit();

    // update outlier‐scatter source (big, bold points)
    sout["commit"] = [
        ...Array(out1.length).fill(c1),
        ...Array(out2.length).fill(c2),
    ];
    sout["value"] = [...out1, ...out2];
    outlier_scatter.change.emit();
    // --- update connector line (use median) ---
    line_data["x"] = [c1, c2];
    line_data["y"] = [data_box["median"][0], data_box["median"][1]];
    line.change.emit();

    // --- update x-axis factors ---
    plot.x_range.factors = [c1, c2];

    // --- recompute y-padding based on updated whiskers ---
    const low0 = data_box["lower"][0], low1 = data_box["lower"][1];
    const up0 = data_box["upper"][0], up1 = data_box["upper"][1];
    const gmin = Math.min(low0, low1);
    const gmax = Math.max(up0, up1);
    const margin = (gmax - gmin) * 0.05;
    plot.y_range.start = gmin - margin;
    plot.y_range.end = gmax + margin;
})();
