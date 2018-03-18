clc;	% Clear command window.
clear;	% Delete all variables.
close all;	% Close all figure windows except those created by imtool.
imtool close all;	% Close all figure windows created by imtool.

%% Select directory
% Storage_path = 'D:\Data_224_224';

%% Delete old images before generating new ones
if ~isempty(dir(fullfile(Storage_path, '/*.jpeg')))
    which_dir = Storage_path;
    dinfo = dir(which_dir);
    dinfo([dinfo.isdir]) = [];   %skip directories
    filenames = fullfile(which_dir, {dinfo.name});
    delete( filenames{:} )
end
disp('Job terminated!');