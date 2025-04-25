(() => {
    // full_quant, commits, sel1, sel2, qq_src, idl_src, plot are all injected via cb_args

    // 1) pick the new quantile arrays
    const allq = full_quant.quant;
    const idxA = commits.indexOf(sel1.value);
    const idxB = commits.indexOf(sel2.value);
    const qA = allq[idxA];
    const qB = allq[idxB];

    // 2) update the QQ scatter
    qq_src.data.x = qA;
    qq_src.data.y = qB;
    qq_src.change.emit();

    // 3) recompute bounds & margin
    const minv = Math.min(...qA, ...qB);
    const maxv = Math.max(...qA, ...qB);
    const margin = (maxv - minv) * 0.02;

    // 4) update the identity line
    idl_src.data.x = [minv, maxv];
    idl_src.data.y = [minv, maxv];
    idl_src.change.emit();

    // 5) adjust both axes
    plot.x_range.start = minv - margin;
    plot.x_range.end = maxv + margin;
    plot.y_range.start = minv - margin;
    plot.y_range.end = maxv + margin;
})();
