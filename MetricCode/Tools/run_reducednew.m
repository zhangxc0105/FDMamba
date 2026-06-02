mode = opts.mode;

if strcmp(mode, "test_v2") && num == 1
    alg_save_dir = join(['results', opts.task, opts.file, opts.mode], '/');

    if ~strcmp(opts.mode, 'debug')
        mkdir(alg_save_dir);
    end

    cur_datetime = join(string(int16(datevec(now))), "_");

    if ~strcmp(opts.mode, 'debug')
        diary(strcat(join([alg_save_dir, cur_datetime], '/'), '.txt'));
        diary on;

        if exist(alg_save_dir, 'dir')
            alg_save_dir = join([alg_save_dir, cur_datetime], '/');
            mkdir(alg_save_dir);
        end

        disp(strcat("save results in ", alg_save_dir));

        copy_list = opts.copy_list;
        for c = 1:length(copy_list)
            copyfile(copy_list(c), alg_save_dir);
        end
    end
end

for j = 1:length(Alg_names)
    name = Alg_names{j};

    if strcmp(name, 'GT')
        I_HRMS = I_GT;
        alg = alg + 1;

    elseif ~contains(name, DL_lists)
        [I_HRMS, alg] = runner(data, Algorithms, name, alg, opts);

    else
        if strcmp(opts.expc, 'PanCollection')
            load(strcat('2_DL_Result/PanCollection/', opts.sensor, '_Reduced/', name, '/results/output_mulExm_', num2str(num - 1), '.mat'));
        elseif strcmp(opts.expc, 'oldPan')
            load(strcat('2_DL_Result/oldPan/', opts.sensor, '/', opts.file, '/', name, '/output_mulExm_', num2str(num - 1), '.mat'));
        elseif strcmp(opts.expc, 'oldPan_single')
            load(strcat('2_DL_Result/oldPan/', opts.sensor, '/Single/', name, '/', lower(opts.sensor), '_', opts.file, '_', name, '.mat'));
        elseif strcmp(opts.expc, 'DLPan')
            load(strcat('2_DL_Result/DLPan', opts.sensor, '_Reduced/', name, '/results/output_mulExm_', num2str(num - 1), '.mat'));
        end

        I_HRMS = double(sr);
        alg = alg + 1;
    end

    [PSNR, SSIM, Q_avg, SAM, ERGAS, CC, SCC, Q, RMSE] = indexes_evaluation(...
        I_HRMS, I_GT, opts.ratio, L, Qblocks_size, flag_cut_bounds, dim_cut, thvalues, maxvalue);

    MatrixResults(alg, :) = [PSNR, SSIM, Q, Q_avg, SAM, ERGAS, CC, SCC, RMSE];
    MatrixImage(:, :, :, alg) = I_HRMS;

    PSNR_multiexm(alg, num)  = PSNR;
    SSIM_multiexm(alg, num)  = SSIM;
    Q_avg_multiexm(alg, num) = Q_avg;
    SAM_multiexm(alg, num)   = SAM;
    ERGAS_multiexm(alg, num) = ERGAS;
    CC_multiexm(alg, num)    = CC;
    SCC_multiexm(alg, num)   = SCC;
    Q_multiexm(alg, num)     = Q;
    RMSE_multiexm(alg, num)  = RMSE;

    if num == exm_num
        if exm_num == 1258
            drop_ids = [220, 231, 236, 469, 766, 914];
            PSNR_multiexm(:, drop_ids)  = [];
            SSIM_multiexm(:, drop_ids)  = [];
            Q_avg_multiexm(:, drop_ids) = [];
            SAM_multiexm(:, drop_ids)   = [];
            ERGAS_multiexm(:, drop_ids) = [];
            CC_multiexm(:, drop_ids)    = [];
            SCC_multiexm(:, drop_ids)   = [];
            Q_multiexm(:, drop_ids)     = [];
            RMSE_multiexm(:, drop_ids)  = [];
        end

        Avg_MatrixResults = avg_rr_metrics(...
            PSNR_multiexm, SSIM_multiexm, Q_multiexm, Q_avg_multiexm, ...
            SAM_multiexm, ERGAS_multiexm, CC_multiexm, SCC_multiexm, RMSE_multiexm);
    end

    if flag_show
        showImage_zoomin(I_HRMS, printEPS, 1, flag_cut_bounds, dim_cut, thvalues, L, ...
            location1, location2, name, flagvisible, flag_zoomin, ...
            strcat(data_name, num2str(num - 1), '_', name));

        showImageErrnew( I_HRMS, I_GT, location1, location2, range_bar, printEPS, flagvisible, flag_zoomin, strcat( data_name, num2str( num- 1), '_', name, '_err'), opts. task);
    end

    if flag_savemat == 1 && contains(name, fieldnames(Algorithms))
        save(strcat(data_name, name, '.mat'), 'sr');
    end
end

if mod(num, 100)
    fprintf('\n');
    disp('#######################################################');
    disp(['Display the performance for:', num2str(num)]);
    disp('#######################################################');
    disp(' |====PSNR(Inf)====|====SSIM(1)====|====Q(1)====|===Q_avg(1)===|=====SAM(0)=====|======ERGAS(0)=======|=======CC(1)=======|=======SCC(1)=======|=======RMSE(0)=======|');

    for k = 1:length(Alg_names)
        fprintf("%s ", Alg_names{k});
        fprintf([repmat('%.4f ', 1, numel(MatrixResults(k, :))), '\n'], MatrixResults(k, :));
    end
end

if num == exm_num
    if strcmp(mode, "test_v2")
        disp("(ver.0.2) last modified in 2022 by Xiao Wu (UESTC).");
    end
end
