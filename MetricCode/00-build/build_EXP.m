function I_EXP = build_EXP(data, mode, opts)
t2=tic;
I_EXP = data.I_MS;
time_I_EXP = toc(t2);
fprintf('Elaboration time EXP: %.2f [sec]\n',time_I_EXP);

end