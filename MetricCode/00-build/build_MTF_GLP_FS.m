function I_MTF_GLP_FS = build_MTF_GLP_HPM_R(data, mode, opts)
cd GLP
t2=tic;
I_MTF_GLP_FS = MTF_GLP_FS(data.I_MS,data.I_PAN,opts.sensor,opts.ratio);
time_MTF_GLP_FS = toc(t2);
fprintf('Elaboration time MTF-GLP-FS: %.2f [sec]\n',time_MTF_GLP_FS);
cd ..   
end