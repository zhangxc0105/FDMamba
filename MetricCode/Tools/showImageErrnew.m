function  showImageErrnew( Ori_Imag, Output, location3, location4, range_bar, print_fig, flagvisible, flag_zoomin, filename, task)

    % 获取图像的尺寸和维度
    n_size = size(Ori_Imag);
    dims = ndims(Ori_Imag);

    % 根据维度设定通道和帧数
    if dims == 2
        channel = 1;
        frame = 1;
    elseif dims == 3
        num_channels = size(Ori_Imag, 3);
        if num_channels == 8
            channel = [1, 3, 5];
        elseif num_channels == 4
            channel = [1, 2, 3];
        else
            channel = 1;  % 默认通道
        end
        frame = 1;
    elseif dims == 4
        channel = [1, 2, 3];
        frame = 1;
    end

    % 计算多通道误差图
    Multi_Err = abs(Ori_Imag(:,:,channel,frame) - Output(:,:,channel,frame));

    % 取三个通道误差图的均值作为误差热图
    ErrMap = mean(Multi_Err, 3);

    % 归一化误差图并映射到颜色图
    ent = uint8((ErrMap - min(ErrMap(:))) / (max(ErrMap(:)) - min(ErrMap(:))) * 256);
    ent = ind2rgb(ent, parula(256));
    imshow(ent);
end
