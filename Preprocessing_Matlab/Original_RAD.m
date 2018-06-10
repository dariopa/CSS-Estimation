%% This code renumbers the hyperspectral images from Arad et al.

clc;	% Clear command window.
clear;	% Delete all variables.
close all;	% Close all figure windows except those created by imtool.
imtool close all;	% Close all figure windows created by imtool.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Select directory
Storage_path = '/scratch_net/biwidl102/dariopa/Images_RAD';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Check how long array will be:
myFolder = '/scratch_net/biwidl102/dariopa/Images/'; % Define your working folder
matFiles = dir(fullfile(myFolder, '*.mat'));

%% Import all images:
matFiles = dir(fullfile(myFolder, '*.mat'));

for k = 1:length(matFiles)
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
    
    save(fullfile(Storage_path, ['RAD_' num2str(k) '.mat']),'rad_new');
end

disp('Job terminated!')



