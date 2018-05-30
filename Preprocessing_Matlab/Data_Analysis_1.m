%% Analysis of hyperspectral images

clc;	% Clear command window.
clear;	% Delete all variables.
close all;	% Close all figure windows except those created by imtool.
imtool close all;	% Close all figure windows created by imtool.

%% Import all images:
myFolder = 'C:\Users\dario\Documents\SemThes\Images'; % Define your working folder

if ~isdir(myFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end

filePattern = fullfile(myFolder, '*.mat');
matFiles = dir(filePattern);

for k = 150:151%length(matFiles)
    baseFileName = matFiles(k).name;
    fullFileName = fullfile(myFolder, baseFileName);

    %% Loading single .mat-files in this script:
    fprintf(1, 'Now reading %s\n', fullFileName);
    load (fullFileName);
    
    %% Print Image
    figure
    subplot(2,2,1)
    imshow(rgb)
    title('Original Image')
    
    %% Reshape the images rgb & rad: 
    % resizing parameters for image:
    X_shape = 1300;
    Y_shape = 1300;

    % rad resizing:
    rad = permute(rad,[2,1,3]);
    rad = imresize(rad,[X_shape Y_shape], 'bicubic');
    rad_new = zeros(size(rad,1),size(rad,2), size(rad,3));

    for i = 1:size(rad_new,1)
            rad_new(size(rad_new,1)+1-i,:,:) = rad(i,:,:);
    end
    
    [row, columns, spectrum] = size(rad_new);
    n_features = row*columns;
    rad_reshaped = permute(reshape(rad_new,[1,n_features,spectrum]),[3,2,1]); % size(rad_reshaped = [33 x m]

    %% Calculate RGB image from Power Spectrum and CSS
    a = 1;
    b = 1;
    for i = 401:10:710
        alpha_red = 0.5680;
        mean_red = 606.3374;
        sigma_red = 32.5;
        CSS_calc(1,a) = alpha_red*exp(-(i-mean_red)^2/(2*sigma_red^2));
        alpha_green = 0.601;
        mean_green = 532.8549;
        sigma_green = 33.5;
        CSS_calc(2,a) = alpha_green*exp(-(i-mean_green)^2/(2*sigma_green^2));
        alpha_blue = 0.56;
        mean_blue = 466.5481;
        sigma_blue = 32.5;
        CSS_calc(3,a) = alpha_blue*exp(-(i-mean_blue)^2/(2*sigma_blue^2));
        a = a+1;
    end
    
    I = CSS_calc*rad_reshaped/4095;
    I_image = permute(reshape(I,[3,X_shape,Y_shape]),[2,3,1]);
    
    subplot(2,2,2)
    imshow(I_image)
    title('Reconstructed Image from estimated CSS')

    subplot(2,2,[3,4]);
    hold on
    x = 401:10:710;
    plot(x,CSS_calc(1,1:31),'red');
    plot(x,CSS_calc(2,1:31),'green');
    plot(x,CSS_calc(3,1:31),'blue');
    hold off
    title('Estimated CSS')
    
end

