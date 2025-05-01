// bootstrap_comparison.js
// callback args: raw, hist, ci, sel1, sel2, low_span, high_span, low_label, high_label, plot
(() => {
    try {
        // 1) unpack
        const commits = raw.data["commit"];
        const arrays = raw.data["raw"];
        const c1 = sel1.value;
        const c2 = sel2.value;
        const i1 = commits.indexOf(c1);
        const i2 = commits.indexOf(c2);
        const a1 = arrays[i1];
        const a2 = arrays[i2];

        // 2) bootstrap
        const B = 1000;
        let diffs = [];
        for (let b = 0; b < B; b++) {
            let s1 = [], s2 = [];
            for (let i = 0; i < a1.length; i++) {
                s1.push(a1[Math.floor(Math.random() * a1.length)]);
                s2.push(a2[Math.floor(Math.random() * a2.length)]);
            }
            s1.sort((x, y) => x - y);
            s2.sort((x, y) => x - y);
            let m1 = s1[Math.floor(s1.length / 2)];
            let m2 = s2[Math.floor(s2.length / 2)];
            diffs.push((m2 / m1 - 1) * 100);
        }

        // 3) rebuild histogram
        const nb = Math.ceil(Math.sqrt(diffs.length));
        const dmin = Math.min(...diffs);
        const dmax = Math.max(...diffs);
        const bw = (dmax - dmin) / nb;
        let edges = Array.from({ length: nb + 1 }, (_, k) => dmin + k * bw);
        let counts = new Array(nb).fill(0);
        for (let d of diffs) {
            let idx = Math.floor((d - dmin) / bw);
            idx = Math.min(Math.max(idx, 0), nb - 1);
            counts[idx]++;
        }

        // 4) update the ColumnDataSource for the bars
        hist.data["top"] = counts;
        hist.data["left"] = edges.slice(0, nb);
        hist.data["right"] = edges.slice(1);
        hist.change.emit();

        // 5) recompute 95% CI
        diffs.sort((a, b) => a - b);
        const low_idx = Math.floor(0.025 * diffs.length);
        const high_idx = Math.floor(0.975 * diffs.length);
        const low_val = diffs[low_idx];
        const high_val = diffs[high_idx];

        ci.data["low"] = [low_val];
        ci.data["high"] = [high_val];
        ci.change.emit();

        // 6) move the spans and labels
        low_span.location = low_val;
        high_span.location = high_val;

        const { start, end } = plot.y_range;
        const tops = hist.data["top"];
        const y_max = Math.max(...tops) * 0.9;

        low_label.x = low_val;
        low_label.y = y_max;
        low_label.text = `2.5%: ${low_val.toFixed(2)}%`;

        high_label.x = high_val;
        high_label.y = y_max;
        high_label.text = `97.5%: ${high_val.toFixed(2)}%`;

    } catch (err) {
        // if anything goes wrong, you’ll see it in the browser console
        console.error("bootstrap_comparison JS error:", err);
    }
})();
