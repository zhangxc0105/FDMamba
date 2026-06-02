function Avg_MatrixResults = avg_fr_metrics(D_lambda_multiexm, D_S_multiexm, QNRI_multiexm)
%average Q_avg
avg_D_lambda_multiexm = mean(D_lambda_multiexm, 2);
std_D_lambda_multiexm = std(D_lambda_multiexm, 0, 2);

avg_D_S_multiexm = mean(D_S_multiexm, 2);
std_D_S_multiexm = std(D_S_multiexm, 0, 2);

avg_QNRI_multiexm = mean(QNRI_multiexm, 2);
std_QNRI_multiexm = std(QNRI_multiexm, 0, 2);

Avg_MatrixResults = [avg_D_lambda_multiexm, std_D_lambda_multiexm, ...
    avg_D_S_multiexm, std_D_S_multiexm, avg_QNRI_multiexm, std_QNRI_multiexm
    ];
end