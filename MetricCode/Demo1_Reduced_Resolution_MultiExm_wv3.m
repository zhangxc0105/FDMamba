clear, clc
%This is a demo to run fusion algorithms on Reduced Resolution
% Xiao Wu (UESTC)
% last modified in 2023.
% LJ Deng(UESTC)
clear; close all;
warning('off','all');
% warning('query','last')
%% =======load directors========
% Tools
addpath('../Tools', '../Tools/export_fig', 'Tools', '00-build', 'Quality_Indices');
% Select algorithms to run
Algorithms = struct("EXP", @build_EXP, "BT_H", @build_BT_H, "BDSD_PC", @build_BDSD_PC, ...
    "MTF_GLP_HPM_R", @build_MTF_GLP_HPM_R, "MTF_GLP_HPM", @build_MTF_GLP_HPM, "MTF_GLP_FS", @build_MTF_GLP_FS, ...
    "TV", @build_TV, "CVPR19", @build_CVPR19);
% DL_lists = ["BDPN" "BDSD_PC" "BT_H" "CANNet" "CVPR19" "DCFNet" "DiCNN" "EXP" "FusionNet" "GT" "GTP-PNet" "HMPNet" "LAGConv" "LRTCFPan" "MMNet" "MSDCNN" "MTF_GLP_FS" "MTF_GLP_HPM_R" "PanNet" "PMAC" "PNN" "RRNet" "TV"];  % used for load mat ,"PMAC", "MMNet" (only wv3)
% Alg_names = ["EXP", "BDSD_PC", "MTF_GLP_FS", "MTF_GLP_HPM", "CVPR19", "BT_H", "PNN", "DiCNN", "PanNet", "FusionNet", "LAGConv", "DCFNet"]; 
% Alg_names = [fieldnames(Algorithms)', DL_lists(2:end)];
%'MTF_GLP_HPM_R','MTF_GLP_FS','TV','PNN','PanNet','DiCNN','MSDCNN','BDPN','FusionNet','LAGConv'};
% Alg_names = ["BT_H", "PMAC", "MMNet"]
% Alg_names = fieldnames(Algorithms);
DL_lists = ["fdmamba"]
Alg_names = DL_lists;
opts.copy_list = [];
opts.expc = "PanCollection";
opts.mode = "debug";  %test_v2: print once, test: print to logs for each alg, search: gridsearch for optimization
opts.sensor = 'WV3'; % 'WV3', 'WV2', 'QB', 'GF2', you don't set 'none' (modified sensor cases of MTF_PAN, genMTF, build_TV).
opts.task = 'pansharpening';
opts.file = 'test_wv3_multiExm1'; %test_WV3_multiExm1
%%
location1                = [];  % Location of zoom in
location2                = [];
range_bar = [0, 1];
%% Initialization of the Matrix of Results
NumIndexes = 9;
MatrixResults = zeros(numel(Alg_names),NumIndexes);
%%
flag_show = 0; % don't show and save figures.
flagvisible = 'off'; %used for saving figures but don't show figures.
flag_zoomin = 1;
flag_savemat = 0;
printEPS = 0;% Print Eps and pdf

%% load Indexes for WV3_RR
flag_cut_bounds = 1;% Cut Final Image
Qblocks_size = 32;
bicubic = 0;% Interpolator
dim_cut = 30;% Cut Final Image, PanCollection: 30, DLPan: 21
thvalues = 0;% Threshold values out of dynamic range
opts.ratio = 4;% Resize Factor
L = 11;% Radiometric Resolution
% WV3,WV2,WV4,QB and q2n should be 2047.0; GF-2 is 1023.0; Other indexes
% should be calculated within 0-1. maxvalue is used for q2n to restore data range of input data;
maxvalue = 2047; 

%% =======read Multiple TestData_wv3.h5 (four 512x512 WV3 simulated data)========
%file_test = strcat('D:/datasets/Pansharpening/test_data/GF2/', opts.file,'.h5');
file_test = strcat('D:/数据集/pan_dataset/', opts.file,'.h5');
% file_test = strcat('D:/Datasets/pansharpening/DLPan/test_data/wv3/', opts.file,'.h5');
gt_multiExm_tmp = h5read(file_test,'/gt');  % WxHxCxN=1x2x3x4
gt_multiExm = permute(gt_multiExm_tmp, [4 2 1 3]); % NxHxWxC=4x2x1x3

ms_multiExm_tmp = h5read(file_test,'/ms');  % WxHxCxN=1x2x3x4
ms_multiExm = permute(ms_multiExm_tmp, [4 2 1 3]); % NxHxWxC=4x2x1x3

lms_multiExm_tmp = h5read(file_test,'/lms');  % WxHxCxN=1x2x3x4
lms_multiExm = permute(lms_multiExm_tmp, [4 2 1 3]); % NxHxWxC=4x2x1x3

pan_multiExm_tmp = h5read(file_test,'/pan');  % WxHxCxN=1x2x3x4
pan_multiExm = permute(pan_multiExm_tmp, [4 2 1 3]); % NxHxWxC=4x2x1x3


data_name = strcat('3_EPS/PanCollection/', opts.sensor, '/multi/');  % director to save EPS figures
mkdir(dirpath(data_name));

alg_name = char(Alg_names(1));
save_dir = fullfile('3_EPS/PanCollection', opts.sensor, 'multi', 'reduced', alg_name);
mkdir(save_dir);
%% ==========Read each Data====================
exm_num = size(ms_multiExm, 1);
for num = 1:exm_num
    alg = 0;
    %% read each data
    HRMS_tmp = gt_multiExm(num, :, :, :); % I_GT
    I_GT     = squeeze(HRMS_tmp);
    LRMS_tmp = ms_multiExm(num, :, :, :); % I_MS_LR
    data.I_MS_LR     = squeeze(LRMS_tmp);
    LMS_tmp  = lms_multiExm(num, :, :, :); % I_MS
    data.I_MS     = squeeze(LMS_tmp);
    PAN_tmp  = pan_multiExm(num, :, :, :); % I_PAN
    data.I_PAN      = squeeze(PAN_tmp);
    
    %% show I_MS_LR, I_GT, PAN Imgs:
    if flag_show
        mkdir(strcat(data_name,'reduced'));
        showImage_zoomin(data.I_MS,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, ...
            location1, location2, "MS", flagvisible, flag_zoomin, strcat(data_name, 'reduced/', opts.file, '_', num2str(num-1), '_ms'));
        showImage_zoomin(I_GT,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, ...
            location1, location2, "GT", flagvisible, flag_zoomin, strcat(data_name, 'reduced/', opts.file, '_',num2str(num-1), '_gt'));
        showPan_zoomin(data.I_PAN,printEPS,2,flag_cut_bounds,dim_cut, ...
            location1, location2, "PAN", flagvisible, flag_zoomin, strcat(data_name, 'reduced/', opts.file, '_',num2str(num-1), '_pan'));
        % showImage(data.I_MS,0,1,flag_cut_bounds,dim_cut,thvalues,L, "MS", flagvisible, strcat(data_name, num2str(i-1), '_ms'));
        % showImage(I_GT,0,1,flag_cut_bounds,dim_cut,thvalues,L, "GT", flagvisible, strcat(data_name, num2str(i-1), '_gt'));
        % showPan(data.I_PAN,printEPS,2,flag_cut_bounds,dim_cut, "PAN", flagvisible, strcat(data_name, num2str(i-1), '_pan'));
    end
    %%

    run_reduced();
    % I=double(sr);
    % showImageErrnew(I, I_GT, opts.task);
    % print('-dpng', fullfile(save_dir, [num2str(num-1), '_error.png']))
    % showImage8LR(I,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, opts.ratio);
    % print('-dpng', fullfile(save_dir, [num2str(num-1), '_ms.png']))

    % showImage8LR(I_GT,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, opts.ratio);
    % print('-dpng', strcat(data_name, 'reduced/gt/', num2str(num-1), '_gt.png'))
end
close all;
%Print in LATEX
PSNR_multiexm,SSIM_multiexm,Q_multiexm, Q_avg_multiexm, SAM_multiexm, ERGAS_multiexm, CC_multiexm, SCC_multiexm, RMSE_multiexm
matrix2latex(MatrixResults(:,[1,3,4]),'RR_Assessment.tex', 'rowLabels',Alg_names,'columnLabels',[{'Q2n'},{'SAM'},{'ERGAS'}],'alignment','c','format', '%.4f');
matrix2latex(MatrixResults(:, [1,2,4,5,6]), strcat(opts.file, '_RR_Assessment.tex'), 'rowLabels', Alg_names,'columnLabels',...
[{'PSNR'},{'SSIM'},{'Q2n'},{'SAM'},{'ERGAS'}],'alignment','c','format', '%.4f');
%% View All

if size(I_GT,3) == 4
    vect_index_RGB = [3,2,1];
else
    vect_index_RGB = [5,3,2];
end

titleImages = strrep(Alg_names,"_", "-");
figure, showImagesAll(MatrixImage,titleImages,vect_index_RGB,flag_cut_bounds,dim_cut,0);

%% PSNR_multiexm,SSIM_multiexm,Q_multiexm, Q_avg_multiexm, SAM_multiexm, ERGAS_multiexm, CC_multiexm, SCC_multiexm, RMSE_multiexm
matrix2latex(Avg_MatrixResults(:,[9,10,11,12,5,6,1,2,3,4,15,16]),strcat(opts.file, '_Avg_RR_Assessment.tex'), ...
'rowLabels',titleImages,'columnLabels',[{'SAM'}, {'SAM-std'}, {'ERGAS'}, {'ERGAS-std'}, {'Q2n'}, {'Q2n-std'}, {'PSNR'}, {'PSNR-std'}, {'SSIM'}, {'SSIM-std'}, {'SCC'}, {'SCC-std'}],'alignment','c','format', '%.3f');
%%
fprintf('\n')
disp('#######################################################')
disp(['Display the performance for:', num2str(1:num)])
disp('#######################################################')
disp(' |====PSNR(Inf)====|====SSIM(1)====|====Q(1)====|===Q_avg(1)===|=====SAM(0)=====|======ERGAS(0)=======|=======CC(1)=======|=======SCC(1)=======|=======RMSE(0)=======')
for i=1:length(titleImages)
    fprintf("%s ", titleImages{i});
    fprintf([ repmat('%.4f ',1,numel(Avg_MatrixResults(i, :))) '\n'], Avg_MatrixResults(i, :));
end
close;
diary off;

