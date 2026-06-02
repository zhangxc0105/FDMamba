function Avg_MatrixResults = avg_rr_metrics(varargin)
% %average Q_avg
% avg_Q_multiexm = mean(Q_multiexm, 2);
% std_Q_multiexm = std(Q_multiexm, 0, 2);
% 
% avg_Q_avg_multiexm = mean(Q_avg_multiexm, 2);
% std_Q_avg_multiexm = std(Q_avg_multiexm, 0, 2);
% 
% avg_SAM_multiexm = mean(SAM_multiexm, 2);
% std_SAM_multiexm = std(SAM_multiexm, 0, 2);
% 
% avg_ERGAS_multiexm = mean(ERGAS_multiexm, 2);
% std_ERGAS_multiexm = std(ERGAS_multiexm, 0, 2);
% 
% avg_SCC_multiexm = mean(SCC_multiexm, 2);
% std_SCC_multiexm = std(SCC_multiexm, 0, 2);
% 
% % avg_psnr_multiexm = mean(psnr_multiexm, 2);
% % std_psnr_multiexm = std(psnr_multiexm, 0, 2);
% % 
% % avg_ssim_multiexm = mean(ssim_multiexm, 2);
% % std_ssim_multiexm = std(ssim_multiexm, 0, 2);
% 
% Avg_MatrixResults = [avg_Q_multiexm, std_Q_multiexm, avg_Q_avg_multiexm, std_Q_avg_multiexm, ...
%     avg_SAM_multiexm, std_SAM_multiexm, avg_ERGAS_multiexm, std_ERGAS_multiexm,...
%     avg_SCC_multiexm, std_SCC_multiexm];%, avg_psnr_multiexm, std_psnr_multiexm, avg_ssim_multiexm, std_ssim_multiexm];
num_alg = size(varargin{1}, 1);
Avg_MatrixResults = zeros(num_alg, 2*nargin);
j = 1;
for i = 1:2:2*nargin
    Avg_MatrixResults(:, i:i+1) = [mean(varargin{j}, 2), std(varargin{j}, 0, 2)];
    j = j +1;
end