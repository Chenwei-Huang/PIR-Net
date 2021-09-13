clear all
close all
clc

source_path         = '.\data\source\test_';
opticalflow_lable   = '.\data\lables\';
opticalflow_predict = '.\demo\test_';
targetpath          = '.\data\target\';

index = 1006;
rgb = imread([source_path num2str(index),'.jpg']);    
load([opticalflow_lable num2str(index),'.mat']);
load([opticalflow_predict num2str(index),'_flow.mat']);
target = imread([targetpath num2str(index),'_t.jpg']);

[Xa,Ya] = meshgrid(1:620,1:460);
% Xb = round(Xa + fmap(:,:,1));
% Yb = round(Ya + fmap(:,:,2));
Xb = round(Xa + predict_fmap(:,:,1));
Yb = round(Ya + predict_fmap(:,:,2));
flag = fmap(:,:,3);

for i = 1:3
    Ca_image = rgb(:,:,i);
    Cb_image = zeros(460,620);
    for u = 1:460
        for v = 1:620
            if ( (flag(u,v) == 1) && (Yb(u,v)>0 && Xb(u,v)>0) )
                Cb_image(Yb(u,v),Xb(u,v)) = Ca_image(u,v);
            else
                Cb_image(1,1) = Ca_image(u,v);
            end
        end
    end
    Cb(:,:,i) = Cb_image;
end

Cb = uint8(Cb);
se = strel('disk',2);
I = imdilate(Cb,se);

subplot(2,2,1),imshow(rgb),title('source');
subplot(2,2,2),imshow(target),title('target');
subplot(2,2,3),imshow(Cb),title('source+opticalflow_lable');
subplot(2,2,4),imshow(I),title('dilate');
figure,imshow(uint8(I/2+target/2)),title('Cb/2+target/2')

% imwrite(I,'transformed_image.jpg');