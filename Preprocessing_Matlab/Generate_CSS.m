%% This code generates different CSS parameters

clc;	% Clear command window.
clear;	% Delete all variables.
close all;	% Close all figure windows except those created by imtool.
imtool close all;	% Close all figure windows created by imtool.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
%% Set Parameters for CSS:
Red_boundaries = [0.5, 600, 28; ... % min
                  0.6, 615, 35];      % max
Green_boundaries = [0.5, 525, 28; ...
                    0.6, 540, 35];
Blue_boundaries = [0.5, 460, 28; ...
                  0.6, 475, 35];

%% Calculate parameters for 3 channels:
nr_images = 200; % for each image in dataset, it creates "nr_images"

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

save('_nr_images.mat','nr_images');
% red channel
r_alpha = zeros(1,nr_images);
r_mean = zeros(1,nr_images);
r_sigma = zeros(1,nr_images);
% green channel
g_alpha = zeros(1,nr_images);
g_mean = zeros(1,nr_images);
g_sigma = zeros(1,nr_images);
% blue channel
b_alpha = zeros(1,nr_images);
b_mean = zeros(1,nr_images);
b_sigma = zeros(1,nr_images);

for counter = 1:nr_images
    % r = a + (b-a).*rand(N,1)
    % Red channel
    r_alpha(1,counter) = roundn(Red_boundaries(1,1) + (Red_boundaries(2,1) - Red_boundaries(1,1)).*rand(1),-3);
    r_mean(1,counter) = roundn(Red_boundaries(1,2) + (Red_boundaries(2,2) - Red_boundaries(1,2)).*rand(1),-1);
    r_sigma(1,counter) = roundn(Red_boundaries(1,3) + (Red_boundaries(2,3) - Red_boundaries(1,3)).*rand(1),-2);

    % Green channel
    g_alpha(1,counter) = roundn(Green_boundaries(1,1) + (Green_boundaries(2,1) - Green_boundaries(1,1)).*rand(1),-3);
    g_mean(1,counter) = roundn(Green_boundaries(1,2) + (Green_boundaries(2,2) - Green_boundaries(1,2)).*rand(1),-1);
    g_sigma(1,counter) = roundn(Green_boundaries(1,3) + (Green_boundaries(2,3) - Green_boundaries(1,3)).*rand(1),-2);

    % Blue channel
    b_alpha(1,counter) = roundn(Blue_boundaries(1,1) + (Blue_boundaries(2,1) - Blue_boundaries(1,1)).*rand(1),-3);
    b_mean(1,counter) = roundn(Blue_boundaries(1,2) + (Blue_boundaries(2,2) - Blue_boundaries(1,2)).*rand(1),-1);
    b_sigma(1,counter) = roundn(Blue_boundaries(1,3) + (Blue_boundaries(2,3) - Blue_boundaries(1,3)).*rand(1),-2);
end


save('_CSS_Raw_Param.mat','r_alpha','r_mean', 'r_sigma', 'g_alpha','g_mean', 'g_sigma', 'b_alpha','b_mean', 'b_sigma');

disp('Job terminated!');


