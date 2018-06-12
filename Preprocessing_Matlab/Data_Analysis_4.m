%% Analysis of hyperspectral images

clc;	% Clear command window.
clear;	% Delete all variables.
close all;	% Close all figure windows except those created by imtool.
imtool close all;	% Close all figure windows created by imtool.


%% Check how long array will be:
file = 'C:\Users\dario\Documents\SemThes\Images_RAD\1.mat'; % Define your working folder
% matFiles = dir(fullfile(myFolder, '*.mat'));
fprintf(1, 'Now reading %s\n', file);
load (file);

%% Reshape the images rad: 
rad = permute(rad,[2,1,3]);

figure;
rad_new = zeros(size(rad,1),size(rad,2), size(rad,3));
for i = 1:size(rad_new,1)
        rad_new(size(rad_new,1)+1-i,:,:) = rad(i,:,:);
end
[row, column, spectrum] = size(rad_new);
n_features = row*column;
rad_reshaped = permute(reshape(rad_new,[1,n_features,spectrum]),[3,2,1]);

%% Compute first image
q = 1;
for i = 400:10:701
    CSS_calc(1,q) = 0.58*exp(-(i-615)^2/(2*32.59^2));
    CSS_calc(2,q) = 0.5*exp(-(i-530)^2/(2*34.54^2));
    CSS_calc(3,q) = 0.57*exp(-(i-465)^2/(2*33.2^2));
    q = q+1;
end
% I = [3 x m] || CSS_new = [3 x 33] || rad_reshaped = [33 x m]
I = CSS_calc*rad_reshaped/4095;
I_image_1 = permute(reshape(I,[3,row,column]),[2,3,1]); 
subplot(2,3,1);
imshow(I_image_1);
subplot(2,3,2);
imshow(I_image_1(:, :, 1));

%% Compute second image
q = 1;
for i = 400:10:701
    CSS_calc(1,q) = 0.50*exp(-(i-615)^2/(2*32.59^2));
%     CSS_calc(2,q) = 0.5*exp(-(i-530)^2/(2*34.54^2));
%     CSS_calc(3,q) = 0.57*exp(-(i-465)^2/(2*33.2^2));
    q = q+1;
end
% I = [3 x m] || CSS_new = [3 x 33] || rad_reshaped = [33 x m]
I = CSS_calc*rad_reshaped/4095;
I_image_2 = permute(reshape(I,[3,row,column]),[2,3,1]); 
subplot(2,3,4);
imshow(I_image_2);
subplot(2,3,5);
imshow(I_image_2(:, :, 1));

%% Display Difference from first to second image
I_diff = abs(I_image_2(:,:,1)-I_image_1(:,:,1));
subplot(2,3,[3,6]);
imshow(I_diff);

disp('Job terminated!')


