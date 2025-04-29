// bootstrap_comparison.js

// args: raw, sel1, sel2, hist_source, ci_source, low_span, high_span, low_label, high_label, plot
(() => {
    const data = raw.data;
    const commits = data['commit'];
    const arrays = data['raw'];
    const c1 = sel1.value;
    const c2 = sel2.value;
    const i1 = commits.indexOf(c1);
    const i2 = commits.indexOf(c2);
    const a1 = arrays[i1];
    const a2 = arrays[i2];

    // bootstrap B samples
    const B = 1000;
    let diffs = [];
    for (let b = 0; b < B; b++) {
        let s1 = [], s2 = [];
        for (let i = 0; i < a1.length; i++)
            s1.push(a1[Math.floor(Math.random() * a1.length)]);
        for (let i = 0; i < a2.length; i++)
            s2.push(a2[Math.floor(Math.random() * a2.length)]);
        s1.sort((x, y) => x - y);
        s2.sort((x, y) => x - y);
        let m1 = s1[Math.floor(s1.length / 2)];
        let m2 = s2[Math.floor(s2.length / 2)];
        diffs.push((m2 / m1 - 1) * 100);
    }

    // histogram
    const nb = Math.ceil(Math.sqrt(diffs.length));
    const dmin = Math.min(...diffs);
    const dmax = Math.max(...diffs);
    const bw = (dmax - dmin) / nb;
    let edges = [];
    for (let k = 0; k <= nb; k++)
        edges.push(dmin + k * bw);
    let counts = new Array(nb).fill(0);
    for (let d of diffs) {
        let idx = Math.floor((d - dmin) / bw);
        if (idx < 0) idx = 0;
        if (idx >= nb) idx = nb - 1;
        counts[idx]++;
    }
    hist.data['top'] = counts;
    hist.data['left'] = edges.slice(0, nb);
    hist.data['right'] = edges.slice(1);
    hist.change.emit();

    // compute 95% CI
    diffs.sort((a, b) => a - b);
    const low_idx = Math.floor(0.025 * diffs.length);
    const high_idx = Math.floor(0.975 * diffs.length);
    const low_val = diffs[low_idx];
    const high_val = diffs[high_idx];
    ci_source.data['low'] = [low_val];
    ci_source.data['high'] = [high_val];
    ci_source.change.emit();

    // update spans
    low_span.location = low_val;
    high_span.location = high_val;

    // update labels
    const y = plot.y_range.end * 0.9;
    low_label.x = low_val;
    low_label.y = y;
    low_label.text = `2.5%: ${low_val.toFixed(2)}%`;
    high_label.x = high_val;
    high_label.y = y;
    high_label.text = `97.5%: ${high_val.toFixed(2)}%`;
})();
