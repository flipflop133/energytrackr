(() => {
    const x_all = full_ecdf.data['ecdf_x'];
    const y_all = full_ecdf.data['ecdf_y'];
    const c1 = sel1.value, c2 = sel2.value;
    const i1 = commits.indexOf(c1);
    const i2 = commits.indexOf(c2);

    // update ECDF1
    ecdf1.data['x'] = x_all[i1];
    ecdf1.data['y'] = y_all[i1];
    ecdf1.change.emit();

    // update ECDF2
    ecdf2.data['x'] = x_all[i2];
    ecdf2.data['y'] = y_all[i2];
    ecdf2.change.emit();

    // recompute axis domain to encompass both distributions
    const data1 = x_all[i1];
    const data2 = x_all[i2];
    const combined = data1.concat(data2);
    const min = Math.min.apply(null, combined);
    const max = Math.max.apply(null, combined);
    plot.x_range.start = min;
    plot.x_range.end = max;
})();