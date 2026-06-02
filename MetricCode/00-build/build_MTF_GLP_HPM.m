function I_MTF_GLP_HPM_R = build_MTF_GLP_HPM(data, mode, opts)
addpath GLP
t2=tic;
% I_MTF_GLP_HPM_R = MTF_GLP_HPM(data.I_MS,data.I_PAN,opts.sensor,opts.ratio);
I_MTF_GLP_HPM_R = MTF_GLP_HPM_PP(data.I_PAN,data.I_MS_LR,opts.sensor,opts.ratio);
time_MTF_GLP_HPM = toc(t2);
fprintf('Elaboration time MTF-GLP-HPM: %.2f [sec]\n',time_MTF_GLP_HPM);
rmpath GLP
end