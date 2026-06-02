% Xiao Wu (UESTC)
% last modified in 2022.
% LJ Deng (UESTC)
% 2020-06-02

clear; close all;
warning('off','all');
% warning('query','last')
%% =======load directors========
% Tools
addpath([pwd,'/Tools']);
addpath([pwd,'/00-build']);
addpath([pwd,'/Quality_Indices']);
% % Select algorithms to run
% algorithms = {'BT-H','BDSD-PC',...
%     'MTF-GLP-HPM-R','MTF-GLP-FS','TV','PNN','PanNet','DiCNN','MSDCNN','BDPN','FusionNet','LAGConv'};%'PNN'
% algorithms = {'GT', 'BT-H', 'PanNet', 'FusionNet'};
Algorithms = struct("EXP", @build_EXP, "BT_H", @build_BT_H, "BDSD_PC", @build_BDSD_PC, ...
    "MTF_GLP_HPM_R", @build_MTF_GLP_HPM_R, "MTF_GLP_FS", @build_MTF_GLP_FS, ...
    "TV", @build_TV,"CVPR19", @build_CVPR19, "MTF_GLP_HPM", @build_MTF_GLP_HPM);
% DL_lists = ["BDPN" "BDSD_PC" "CANNet" "CVPR19" "DCFNet" "DiCNN" "EXP" "FusionNet" "HMPNet" "LAGConv" "LRTCFPan" "MMNet" "MSDCNN" "MTF_GLP_FS" "PanNet" "PMAC" "PNN" "RRNet" "TV"];  % used for load mat
% Alg_names = {'BT_H', 'PanNet', 'FusionNet'};
% Alg_names = ["BDSD_PC", "MTF_GLP_FS", "MTF_GLP_HPM", "CVPR19", "PNN", "DiCNN", "PanNet", "FusionNet", "LAGConv", "DCFNet"]; 
% Alg_names = ["RRNet"];
DL_lists = ["fdmamba"]
Alg_names = DL_lists;
opts.copy_list = [];
opts.expc = "PanCollection";
opts.mode = "debug"; %test_v2: print once, test: print to logs for each alg, search: gridsearch for optimization
opts.sensor = 'WV3'; % 'WV3', 'WV2', 'QB', 'GF2', you don't set 'none' (modified sensor cases of MTF_PAN, genMTF, build_TV).
opts.task = 'pansharpening';
opts.file = 'test_wv3_OrigScale_multiExm1';
%%
data_name = join(['3_EPS/PanCollection/',opts.sensor,strcat(opts.sensor, '_os_')], '/');  % director to save EPS figures
mkdir(dirpath(data_name));

alg_name = char(Alg_names(1));
save_dir = fullfile('3_EPS/PanCollection', opts.sensor, 'multi', 'full', alg_name);
mkdir(save_dir);
%%
NumIndexes = 3;
MatrixResults = zeros(numel(Alg_names),NumIndexes);
flag_show = 0; % don't show and save figures.
flagvisible = 'off'; %used for saving figures but don't show figures.
flagQNR = 0; %% Flag QNR/HQNR, 1: QNR otherwise HQNR
% for img show
location1                = [];  %default: data6: [10 50 1 60]; data7:[140 180 5 60]
location2                = [];  %default: data6: [190 240 5 60]; data7:[190 235 120 150]

%% load Indexes for FR
Qblocks_size = 32;
bicubic = 0;% Interpolator
flag_cut_bounds = 1;% Cut Final Image
dim_cut = 21;% Cut Final Image
thvalues = 0;% Threshold values out of dynamic range
printEPS = 1;% Print Eps
opts.ratio = 4;% Resize Factor
L = 11;% Radiometric Resolution
clear print

%% ==========Read each Data====================
%% read each data
file_test = strcat('D:/数据集/pan_dataset/', opts.file,'.h5');
%load(file_test)   % get I_MS_LR, I_MS, I_PAN and sensors' info.
ms_multiExm_tmp = h5read(file_test,'/ms');  % WxHxCxN=1x2x3x4
ms_multiExm = permute(ms_multiExm_tmp, [4 2 1 3]); % NxHxWxC=4x2x1x3

lms_multiExm_tmp = h5read(file_test,'/lms');  % WxHxCxN=1x2x3x4
lms_multiExm = permute(lms_multiExm_tmp, [4 2 1 3]); % NxHxWxC=4x2x1x3

pan_multiExm_tmp = h5read(file_test,'/pan');  % WxHxCxN=1x2x3x4
pan_multiExm = permute(pan_multiExm_tmp, [4 2 1 3]); % NxHxWxC=4x2x1x3


%% ==========Read each Data====================
exm_num = size(ms_multiExm, 1);
% ms_shape = size(ms_multiExm_tmp);  % 获取数组的形状

for num = 1:exm_num
    alg = 0;
    %% read each data
    LRMS_tmp = ms_multiExm(num, :, :, :); % I_MS_LR
    data.I_MS_LR     = squeeze(LRMS_tmp);
    LMS_tmp  = lms_multiExm(num, :, :, :); % I_MS
    data.I_MS     = squeeze(LMS_tmp);
    PAN_tmp  = pan_multiExm(num, :, :, :); % I_PAN
    data.I_PAN      = squeeze(PAN_tmp);
    
    
    
    %% show I_MS_LR, I_GT, PAN Imgs:
    if flag_show
        showImage_zoomin(data.I_MS,0,1,flag_cut_bounds,dim_cut,thvalues,L, ...
            location1, location2, "MS", flagvisible, strcat(data_name, num2str(num-1), '_ms'));
        
        
        %showPan(I_PAN,printEPS,2,flag_cut_bounds,dim_cut);
        showPan_zoomin(data.I_PAN,printEPS,2,flag_cut_bounds,dim_cut, ...
            location1, location2, "PAN", flagvisible, strcat(data_name, num2str(num-1), '_pan'));
        %    pause(2);print('-deps', strcat(data_name, num2str(i-1), '_pan', '.eps'))
    end
    %%
    run_full();

    % I  = double(sr);
    % showImage8LR(I,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, opts.ratio);
    % print('-dpng', fullfile(save_dir, [num2str(num-1), '_ms.png']))

    % for alg = 1:numel(Alg_names)
    %     outdir = strcat("4_Export/", opts.sensor, "_Full/", Alg_names(alg), "/results/");
    %     mkdir(outdir);
    %     outfile = strcat(outdir, "output_mulExm_", num2str(num - 1), ".mat");
    %     sr = MatrixImage(:, :, :, alg);
    %     save(outfile, "sr");
    % end
end
%% Print in LATEX
if flagQNR == 1
    matrix2latex(MatrixResults,strcat(opts.file, '_FR_Assessment.tex'), 'rowLabels',Alg_names,'columnLabels',[{'DL'},{'DS'},{'QNR'}],'alignment','c','format', '%.4f');
else
    matrix2latex(MatrixResults,strcat(opts.file, '_FR_Assessment.tex'), 'rowLabels',Alg_names,'columnLabels',[{'DL'},{'DS'},{'HQNR'}],'alignment','c','format', '%.4f');
end

%% View All

if size(data.I_MS,3) == 4
    vect_index_RGB = [3,2,1];
else
    vect_index_RGB = [5,3,2];
end

titleImages = strrep(Alg_names,"_", "-");
figure, showImagesAll(MatrixImage,titleImages,vect_index_RGB,flag_cut_bounds,dim_cut,0);

%% D_lambda  D_s QNR
matrix2latex(Avg_MatrixResults,strcat(opts.file, '_Avg_FR_Assessment.tex'), 'rowLabels',titleImages,'columnLabels',[{'D_lambda'}, {'D_l-std'}, {'D_S'}, {'D_S-std'}, {'QNR'}, {'QNR-std'}],'alignment','c','format', '%.4f');
%%
fprintf('\n')
disp('#######################################################')
disp(['Display the performance for:', num2str(1:num)])
disp('#######################################################')
disp(' |===D_lambda_avg (0)===|=====D_s_avg (0)=====|======QNR (1)=======')
% Avg_MatrixResults
for i=1:length(titleImages)
    fprintf("%s ", titleImages{i});
    fprintf([ repmat('%.4f ',1,numel(Avg_MatrixResults(i, :))) '\n'], Avg_MatrixResults(i, :));
end
close all;