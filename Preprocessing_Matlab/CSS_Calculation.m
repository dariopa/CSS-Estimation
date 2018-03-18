%% Calculate CSS functions and visualise it, given RGB and RAD data

clc;	% Clear command window.
clear;	% Delete all variables.
close all;	% Close all figure windows except those created by imtool.
imtool close all;	% Close all figure windows created by imtool.

%% Import all images in loop:
myFolder = 'D:\Images'; % Define your working folder

if ~isdir(myFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end

filePattern = fullfile(myFolder, '*.mat');
matFiles = dir(filePattern);

for k = 1:1%length(matFiles)
    baseFileName = matFiles(k).name;
    fullFileName = fullfile(myFolder, baseFileName);

    %% Loading single .mat-files in this script:
    fprintf(1, 'Now reading %s\n', fullFileName);
    load (fullFileName);
    
    %% Plot RGB image
    figure
    subplot(2,2,1)
    imshow(rgb)
    title('Original Image')
    
    %% Reshape the images rgb & rad: 
    % reshape parameteres for rgd & rad:
    X_shape = 1392;
    Y_shape = 1300;
    
    % rgb resizing:
    rgb_new = rgb(1:X_shape,1:Y_shape,:);
    [row, columns, intensities] = size(rgb_new);
    n_features = row*columns;
    rgb_reshaped = permute(reshape(rgb_new,[1,n_features,intensities]),[3,2,1]);

    % rad resizing:
    rad = permute(rad,[2,1,3]);
    rad_new = zeros(size(rad,1),size(rad,2), size(rad,3));

    for i = 1:size(rad_new,1)
            rad_new(size(rad_new,1)+1-i,:,:) = rad(0+i,:,:);
    end
    
    rad_new = rad_new(1:X_shape,1:Y_shape,:);
    [row, columns, spectrum] = size(rad_new);
    n_features = row*columns;
    rad_reshaped = permute(reshape(rad_new,[1,n_features,spectrum]),[3,2,1]);
    
    %% Calculate CSS
    for i = 1:3
        CSS(i,:) = rgb_reshaped(i,:) * pinv(rad_reshaped/4095);
    end
    
    % Plot CSS
    subplot(2,2,[3,4])
    hold on
    plot(CSS(1,:), 'red')
    plot(CSS(2,:), 'green')
    plot(CSS(3,:), 'blue')
    hold off
    title('Calculated CSS')


    %% Reconstruct RGB from CSS and Power Spectrum
    I = CSS*rad_reshaped/4095;
    I_image = permute(reshape(I,[3,X_shape,Y_shape]),[2,3,1]);
    
    subplot(2,2,2)
    imshow(I_image)
    title('Reconstructed Image from calculated CSS')
    

end

