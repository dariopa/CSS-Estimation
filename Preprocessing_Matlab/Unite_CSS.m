%% Unites the CSS bins

clc;	% Clear command window.
clear;	% Delete all variables.
close all;	% Close all figure windows except those created by imtool.
imtool close all;	% Close all figure windows created by imtool.

%% Add library to create NPY outputs
addpath(genpath('CreateNPY'));

%% Check how long array will be:
% myFolder = 'D:\Data_224_224'; 
matFiles = dir(fullfile(myFolder, '\*.npy'));

for k = 1:length(matFiles)
    %% Loading single .mat-files in this script
    file = fullfile([myFolder, '\CSS' num2str(k) '.npy']);
    input_batch = readNPY(fullfile([myFolder, '\CSS' num2str(k) '.npy']));
    if k == 1
        CSS = input_batch;
    else
        CSS = [CSS;input_batch];
    end
end

disp(size(CSS));
writeNPY(CSS, fullfile([myFolder, '\CSS.npy']));
disp('Job terminated!');






