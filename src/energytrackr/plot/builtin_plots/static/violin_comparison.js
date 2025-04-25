(() => {
    const data = full_kde.data;
    const i1 = labels.indexOf(sel1.value);
    const i2 = labels.indexOf(sel2.value);

    // Rebuild violin A
    let xs = [], ys = [];
    for (let j = 0; j < data['kde_x'][i1].length; j++) {
        xs.push(0 - data['kde_x'][i1][j] * width);
        ys.push(data['kde_y'][i1][j]);
    }
    for (let j = data['kde_x'][i1].length - 1; j >= 0; j--) {
        xs.push(0 + data['kde_x'][i1][j] * width);
        ys.push(data['kde_y'][i1][j]);
    }
    violin1_ds.data = {
        x: xs,
        y: ys,
        commit: Array(xs.length).fill(sel1.value)
    };
    violin1_ds.change.emit();

    // Rebuild violin B
    xs = []; ys = [];
    for (let j = 0; j < data['kde_x'][i2].length; j++) {
        xs.push(1 - data['kde_x'][i2][j] * width);
        ys.push(data['kde_y'][i2][j]);
    }
    for (let j = data['kde_x'][i2].length - 1; j >= 0; j--) {
        xs.push(1 + data['kde_x'][i2][j] * width);
        ys.push(data['kde_y'][i2][j]);
    }
    violin2_ds.data = {
        x: xs,
        y: ys,
        commit: Array(xs.length).fill(sel2.value)
    };
    violin2_ds.change.emit();

    // Update medians & spans
    const med1 = data['median'][i1];
    const med2 = data['median'][i2];
    span1.location = med1; span1.change.emit();
    span2.location = med2; span2.line_color = (med2 > med1 ? 'red' : 'green'); span2.change.emit();

    // Update Δ label
    const d = med2 - med1;
    label_diff.text = `Δ median = ${d.toFixed(2)} J`;
    label_diff.text_color = (d > 0 ? 'red' : 'green');
    label_diff.change.emit();

    // Update x-axis labels
    ticker.ticks = [0, 1];
    ticker.change.emit();
    const ax = cb_obj.x_range;  // trigger re-draw
})();