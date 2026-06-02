function I_BT_H=build_BT_H(data, mode, opts)
cd BT-H
t2=tic;
I_BT_H = BroveyRegHazeMin(data.I_MS,data.I_PAN,opts.ratio);
time_BT_H = toc(t2);
fprintf('Elaboration time BT-H: %.2f [sec]\n',time_BT_H);
cd ..
end