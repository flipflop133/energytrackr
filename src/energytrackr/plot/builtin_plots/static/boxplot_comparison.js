(() => {
    const data_all = full.data;
    const raw_all = raw.data;
    const data_box = box.data;
    const sc1 = scatter1.data;
    const sc2 = scatter2.data;
    const tick_data = tick.data;
    const line_data = line.data;
    const commits = data_all["commit"];
    const c1 = sel1.value;
    const c2 = sel2.value;
    const i1 = commits.indexOf(c1);
    const i2 = commits.indexOf(c2);

    // --- update box stats ---
    for (const f of ["commit", "q1", "q2", "q3", "lower", "upper", "n_low", "n_high"]) {
        data_box[f] = [data_all[f][i1], data_all[f][i2]];
    }
    data_box["color"] = [color_list[0], color_list[1]];
    box.change.emit();

    // --- update scatter points ---
    const a1 = raw_all["raw"][i1];
    const a2 = raw_all["raw"][i2];
    sc1["x"] = Array(a1.length).fill(c1);
    sc1["y"] = a1;
    sc2["x"] = Array(a2.length).fill(c2);
    sc2["y"] = a2;
    scatter1.change.emit();
    scatter2.change.emit();

    // --- update median ticks ---
    tick_data["commit"] = [c1, c2];
    tick_data["med"] = [data_box["q2"][0], data_box["q2"][1]];
    tick.change.emit();

    // --- update connector line ---
    line_data["x"] = [c1, c2];
    line_data["y"] = [data_box["q2"][0], data_box["q2"][1]];
    line.change.emit();

    // --- update x-axis ---
    plot.x_range.factors = [c1, c2];

    // --- recompute y-padding based on new whiskers ---
    const low0 = data_box["lower"][0], low1 = data_box["lower"][1];
    const up0 = data_box["upper"][0], up1 = data_box["upper"][1];
    const gmin = Math.min(low0, low1);
    const gmax = Math.max(up0, up1);
    const m = (gmax - gmin) * 0.05;
    plot.y_range.start = gmin - m;
    plot.y_range.end = gmax + m;
})();