function I_MTF_GLP_HPM_R = build_MTF_GLP_HPM_R(data, mode, opts)
cd GLP
t2=tic;
I_MTF_GLP_HPM_R = MTF_GLP_HPM_R(data.I_MS,data.I_PAN,opts.sensor,opts.ratio);
time_MTF_GLP_HPM_R = toc(t2);
fprintf('Elaboration time MTF-GLP: %.2f [sec]\n',time_MTF_GLP_HPM_R);
cd ..
end