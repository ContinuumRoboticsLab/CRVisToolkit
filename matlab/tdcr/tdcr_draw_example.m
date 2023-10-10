load tdcr_curve_examples.mat

fig1 = draw_tdcr(onesegtdcr,10);
fig2 = draw_tdcr(threesegtdcr,[10 20 30],projections=1,baseplate=0);
fig3 = draw_tdcr(foursectdcr,[15 30 45 60],segframe=1,baseframe=1);