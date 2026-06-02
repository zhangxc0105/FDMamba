function I_BDSD_PC = build_BDSD_PC(data, mode,opts)
cd BDSD
t2=tic;
I_BDSD_PC = BDSD_PC(data.I_MS,data.I_PAN,opts.ratio,opts.sensor);
time_BDSD_PC = toc(t2);
fprintf('Elaboration time BDSD-PC: %.2f [sec]\n',time_BDSD_PC);
cd ..
end