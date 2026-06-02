function I_CVPR19 = build_CVPR2019(data, mode, opts)
cd CVPR19
LRMS = data.I_MS_LR;
t = tic;
for j=1:size(LRMS,3) % preprocessing to LRMS
    bandCoeffs(j)=max(max(LRMS(:,:,j)));
    ImageMS(:,:,j)=LRMS(:,:,j)/bandCoeffs(j);
end
ImageP = data.I_PAN/max(max(data.I_PAN));

divK = 4; % main function
lambda =0.2;
[I_guide,P] = guide( ImageMS,ImageP,divK,lambda,100);

for j=1:size(I_guide, 3)% post-processing
    I_CVPR19(:,:,j)=I_guide(:,:,j)*bandCoeffs(j);
end
time_CVPR2019 = toc(t);
fprintf('Elaboration time CVPR2019: %.2f [sec]\n',time_CVPR2019);
cd ..
end