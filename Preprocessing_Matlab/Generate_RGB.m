%% This code generates different RGB images

clc;	% Clear command window.
clear;	% Delete all variables.
close all;	% Close all figure windows except those created by imtool.
imtool close all;	% Close all figure windows created by imtool.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Select directory
% Storage_path = 'D:\Data_224_224';

%% Batch window:
X_shape = 224; 
Y_shape = 224; 

%% Shape of output image, downsampled:
X_shape_output = 28;
Y_shape_output = 28;

%% Starting and ending of image:
% bin:      |  1  |  2  |  3  |  4  |
% starting: |  1  | 51  | 101 | 151 |
% ending:   |  50 | 100 | 150 | 201 |
bin = 4; 
starting = 151;
ending = 201;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('_CSS_Raw_Param.mat');
load('_nr_images.mat'); % for each image in dataset, it creates "nr_images"


%% Check how long array will be:
myFolder = 'D:\Images\'; % Define your working folder
% matFiles = dir(fullfile(myFolder, '*.mat'));

array_size = 0;
for k = starting:ending % length(matFiles)
    %% Loading single .mat-files in this script
    file = fullfile([myFolder, num2str(k) '.mat']);
    load(file);
        
    %% Reshape the images rad
    rad = permute(rad,[2,1,3]);
    X_window = floor(size(rad,1)/X_shape);
    Y_window = floor(size(rad,2)/Y_shape);
    array_size = array_size + X_window * Y_window * nr_images;
end
disp(array_size);

%% Add library to create NPY outputs
addpath(genpath('CreateNPY'));

%% Import all images:
matFiles = dir(fullfile(myFolder, '*.mat'));

CSS_calc = zeros(3,31);
CSS = zeros(array_size-1,3,3);

if bin == 1
    batch_counter_RGB = 1; % For image storage!
else
    load(['_batch_counter_RGB_' num2str(bin-1) '.mat']);
end
batch_counter_CSS = 1;

for k = starting:ending
    %% Loading single .mat-files in this script:
    file = fullfile([myFolder, num2str(k) '.mat']);
    fprintf(1, 'Now reading %s\n', file);
    load (file);
        
    %% Reshape the images rad: 
    rad = permute(rad,[2,1,3]);
    X_window = floor(size(rad,1)/X_shape);
    Y_window = floor(size(rad,2)/Y_shape);
    rad = rad(1:X_window*X_shape,1:Y_window*Y_shape,:);
    
    rad_new = zeros(size(rad,1),size(rad,2), size(rad,3));

    for i = 1:size(rad_new,1)
            rad_new(size(rad_new,1)+1-i,:,:) = rad(i,:,:);
    end
    
    [row, column, spectrum] = size(rad_new);
    n_features = row*column;
    rad_reshaped = permute(reshape(rad_new,[1,n_features,spectrum]),[3,2,1]); % size(rad_reshaped = [31 x m]

    %% Calculate RGB image from Power Spectrum and CSS
    
    for counter = 1:nr_images
        q = 1;
        for i = 401:10:710
            CSS_calc(1,q) = r_alpha(1,counter)*exp(-(i-r_mean(1,counter))^2/(2*r_sigma(1,counter)^2));
            CSS_calc(2,q) = g_alpha(1,counter)*exp(-(i-g_mean(1,counter))^2/(2*g_sigma(1,counter)^2));
            CSS_calc(3,q) = b_alpha(1,counter)*exp(-(i-b_mean(1,counter))^2/(2*b_sigma(1,counter)^2));
            q = q+1;
        end
        % I = [3 x m] || CSS_new = [3 x 33] || rad_reshaped = [33 x m]
        I = CSS_calc*rad_reshaped/4095;
        I_image = permute(reshape(I,[3,row,column]),[2,3,1]); 
        
        % Now store batches of Image!
        for i = 0:(X_window-1)
            for j = 0:(Y_window-1)
                I_image_batch = I_image((1 + i*X_shape):(i*X_shape + X_shape),(1 + j*Y_shape):(j*Y_shape + Y_shape),:);
%                 I_image_batch = imresize(I_image_batch,[X_shape_output Y_shape_output], 'bicubic');
                imwrite(I_image_batch,[fullfile(Storage_path, '\Image') num2str(batch_counter_RGB) '.jpeg']);

                % Fill CSS Array
                CSS(batch_counter_CSS,:,1) = [r_alpha(1,counter), r_mean(1,counter), r_sigma(1,counter)];
                CSS(batch_counter_CSS,:,2) = [g_alpha(1,counter), g_mean(1,counter), g_sigma(1,counter)];
                CSS(batch_counter_CSS,:,3) = [b_alpha(1,counter), b_mean(1,counter), b_sigma(1,counter)];
                batch_counter_RGB = batch_counter_RGB + 1;
                batch_counter_CSS = batch_counter_CSS + 1;
            end
        end
    end
end
save(['_batch_counter_RGB_' num2str(bin) '.mat'],'batch_counter_RGB');
writeNPY(CSS, fullfile([Storage_path, '\CSS' num2str(bin) '.npy']));
disp('Job terminated!')



