%% Analysis of hyperspectral images

clc;	% Clear command window.
clear;	% Delete all variables.
close all;	% Close all figure windows except those created by imtool.
imtool close all;	% Close all figure windows created by imtool.


%% Import all images:
myFolder = 'C:\Users\dario\Documents\SemThes\Images_RAD\';
matFiles = dir(fullfile(myFolder, '*.mat'));

for k = 1:1%20:length(matFiles)
    %% Loading single .mat-files in this script:
    file = fullfile([myFolder, num2str(k) '.mat']);
    fprintf(1, 'Now reading %s\n', file);
    load (file);
        
    %% Reshape the images rad: 
    rad = permute(rad,[2,1,3]);
    rad_new = zeros(size(rad,1),size(rad,2), size(rad,3));

    for i = 1:size(rad_new,1)
            rad_new(size(rad_new,1)+1-i,:,:) = rad(i,:,:);
    end
    
    [row, column, spectrum] = size(rad_new);
    n_features = row*column;
    rad_reshaped = permute(reshape(rad_new,[1,n_features,spectrum]),[3,2,1]); % size(rad_reshaped = [31 x m]

    %% Compute and plot histograms of red channel
    x_range = [0, 1.2];
    y_range = [0, 6*10^4];
    figure(k)
    % g_alpha
    r_alpha = [0.2, 0.3, 0.5, 0.6];
    r_mean = [600, 600, 600, 600];
    r_sigma = [30, 30, 30, 30, 30];
    for counter = 1:4
        q = 1;
        for i = 401:10:710
            CSS_calc(1,q) = r_alpha(1,counter)*exp(-(i-r_mean(1,counter))^2/(2*r_sigma(1,counter)^2));
            q = q+1;
        end
        I_image = permute(reshape(CSS_calc*rad_reshaped/4095,[1,row,column]),[2,3,1]); 
%         imwrite(I_image,'Image','jpeg'); 
%         I_image = imread('Image');
        
        subplot(4,3,1+(counter-1)*3);
        histogram(I_image(:,:,1));
        xlabel(['alpha_r: ' num2str(r_alpha(counter))])
        axis([x_range(1) x_range(2) y_range(1) y_range(2)])
    end
    %______________________________________________________________________
    % r_mean
    r_alpha = [0.4, 0.4, 0.4, 0.4];
    r_mean = [600, 605, 610, 615];
    r_sigma = [30, 30, 30, 30, 30];
    for counter = 1:4
        q = 1;
        for i = 401:10:710
            CSS_calc(1,q) = r_alpha(1,counter)*exp(-(i-r_mean(1,counter))^2/(2*r_sigma(1,counter)^2));
            q = q+1;
        end
        I_image = permute(reshape(CSS_calc*rad_reshaped/4095,[1,row,column]),[2,3,1]); 
%         imwrite(I_image,'Image','jpeg'); 
%         I_image = imread('Image');
        
        subplot(4,3,2+(counter-1)*3);
        histogram(I_image(:,:,1));
        xlabel(['mean_r: ' num2str(r_mean(counter))])
        axis([x_range(1) x_range(2) y_range(1) y_range(2)])
    end
    %______________________________________________________________________
    % r_sigma
    r_alpha = [0.4, 0.4, 0.4, 0.4];
    r_mean = [600, 600, 600, 600];
    r_sigma = [25, 30, 35, 40];
    for counter = 1:4
        q = 1;
        for i = 401:10:710
            CSS_calc(1,q) = r_alpha(1,counter)*exp(-(i-r_mean(1,counter))^2/(2*r_sigma(1,counter)^2));
            q = q+1;
        end
        I_image = permute(reshape(CSS_calc*rad_reshaped/4095,[1,row,column]),[2,3,1]); 
%         imwrite(I_image,'Image','jpeg'); 
%         I_image = imread('Image');
        
        subplot(4,3,3+(counter-1)*3);
        histogram(I_image(:,:,1));
        xlabel(['sigma_r: ' num2str(r_sigma(counter))])
        axis([x_range(1) x_range(2) y_range(1) y_range(2)])
    end
end


