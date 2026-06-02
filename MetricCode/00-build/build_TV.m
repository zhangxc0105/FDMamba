function I_TV = build_TV(data, mode, opts)
switch opts.sensor
    %this is not QB and GF2's mtf, but use it as default sensor's mtf
    case {'IKONOS','QB', 'GF2'} 
        w=[0.1091    0.2127    0.2928    0.3854];
        c = 8;
        alpha=1.064;
        maxiter=10;
        lambda = 0.47106;
    case {'GeoEye1','WV4'}
        w=[0.1552, 0.3959, 0.2902, 0.1587];
        c = 8;
        alpha=0.75;
        maxiter=50;
        lambda = 157.8954;
    case {'WV3','WV2'}
        w=[0.0657    0.1012    0.1537    0.1473    0.1245    0.1545    0.1338    0.1192];
        c = 8;
        alpha=0.75;
        maxiter=50;
        lambda = 1.0000e-03;
end
cd TV
t2 = tic;
I_TV = TV_pansharpen(data.I_MS_LR,data.I_PAN,alpha,lambda,c,maxiter,w);
time_TV = toc(t2);
fprintf('Elaboration time TV: %.2f [sec]\n',time_TV);
cd ..
end