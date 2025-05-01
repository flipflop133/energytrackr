// static/evolution_select.js
// args: full, plot, sel1, sel2, [OPTIONAL: any other data sources you want to update]

(() => {
    // 1) pull everything out of the callback args
    const dataAll = full.data;
    const commits = dataAll["commit"];
    const medians = dataAll["median"];
    const c1 = sel1.value;
    const c2 = sel2.value;

    // 2) find their indices
    const i1 = commits.indexOf(c1);
    const i2 = commits.indexOf(c2);
    if (i1 < 0 || i2 < 0) {
        // safety check: if something’s wrong, bail
        return;
    }

    // 3) update the x-axis to only show these two commits
    plot.x_range.factors = [c1, c2];

    // 4) update the median line (if you rendered it as a glyph backed by a CDSrc)
    //    assume you passed it in as `median_src`
    if (typeof median_src !== "undefined") {
        const mdata = median_src.data;
        mdata["commit"] = [c1, c2];
        mdata["y"] = [medians[i1], medians[i2]];
        median_src.change.emit();
    }

    // 5) if you have a scatter of raw points, update its source too:
    //    assume `raw_src.data` has an array-of-arrays in raw_src.data["values"]
    if (typeof raw_src !== "undefined") {
        const raw = raw_src.data;
        const v1 = raw["values"][i1];
        const v2 = raw["values"][i2];
        // flatten into commit/value lists:
        raw["commit"] = [...Array(v1.length).fill(c1), ...Array(v2.length).fill(c2)];
        raw["value"] = [...v1, ...v2];
        raw_src.change.emit();
    }

    // 6) if you have any other glyphs (CUSUM, EWMA, etc), repeat the same idea:
    //    read full.data for those arrays, pick [i1,i2], write back to
    //    their CDS and `change.emit()`.

    // 7) finally, recompute y-range so both points are nicely visible:
    //    find min/max y among your updated sources, then:
    const allY = median_src ? median_src.data["y"] : [];
    if (allY.length) {
        const mn = Math.min(...allY);
        const mx = Math.max(...allY);
        const pad = (mx - mn) * 0.1;
        plot.y_range.start = mn - pad;
        plot.y_range.end = mx + pad;
    }
})();
