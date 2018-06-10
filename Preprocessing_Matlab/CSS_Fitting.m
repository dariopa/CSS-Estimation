%% Fit parameters to Database form Jun Jiang:

clc;	% Clear command window.
clear;	% Delete all variables.
close all;	% Close all figure windows except those created by imtool.


%% Import all images in loop:
myFolder = 'Sensitivity_Functions'; % Define your working folder
if ~isdir(myFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end
filePattern = fullfile(myFolder, '*.mat');
matFiles = dir(filePattern);

% Where fitted parameters will be fitted: 
Red_CSS = zeros(length(matFiles), 3);
Green_CSS = zeros(length(matFiles), 3);
Blue_CSS = zeros(length(matFiles), 3);


for k = 1:length(matFiles)
    baseFileName = matFiles(k).name;
    fullFileName = fullfile(myFolder, baseFileName);
  
    %% Loading single .mat-files in this script:
    fprintf(1, 'Now reading %s\n', fullFileName);
    load (fullFileName);
  

    %% Fit gaussian functions for red, green and blue channel:    
    my_option = optimset('display','iter','TolFun', 1e-18, 'TolX', 1e-18, 'MaxIter',200, 'MaxFunEvals',1000);

    % Red channel
    x = 401:10:710;
    disp(size(x));
    y = r(1,1:31);
    amp0 = 0.5;
    mean0 = 600;
    sigma0 = 30;
    params_red = [amp0, mean0, sigma0];

    fun = @(params_red,x)params_red(1,1).*exp(-(x-params_red(1,2)).^2./(2*params_red(1,3).^2)); 
    params_red =lsqcurvefit(fun,params_red,x,y,[],[],my_option);

    % Green channel
    x = 401:10:710;
    y = g(1,1:31);
    amp0 = 0.5; 
    mean0 = 530;
    sigma0 = 35;
    params_green = [amp0, mean0, sigma0];

    fun = @(params_green,x)params_green(1,1).*exp(-(x-params_green(1,2)).^2./(2*params_green(1,3).^2)); 
    params_green =lsqcurvefit(fun,params_green,x,y,[],[],my_option);

    % Blue channel
    x = 401:10:710;
    y = b(1,1:31);
    amp0 = 0.5;
    mean0 = 450;
    sigma0 = 30;
    params_blue = [amp0, mean0, sigma0];

    fun = @( params_blue,x) params_blue(1,1).*exp(-(x- params_blue(1,2)).^2./(2* params_blue(1,3).^2)); 
    params_blue =lsqcurvefit(fun,params_blue,x,y,[],[],my_option);

    %% Plot functions:
        
    figure
    hold on
    x = 401:10:710;
    plot(x,r(1,1:31),'red');
    y_red = params_red(1,1) * gaussmf(x, [params_red(1,3) params_red(1,2)]);
    plot(x,y_red, 'red')
    plot(x,g(1,1:31),'green');
    y_green = params_green(1,1) * gaussmf(x, [params_green(1,3) params_green(1,2)]);
    plot(x,y_green,'green')
    plot(x,b(1,1:31),'blue');
    y_blue = params_blue(1,1) * gaussmf(x, [params_blue(1,3) params_blue(1,2)]);
    plot(x,y_blue,'blue')
    hold off
    
    %% Save parameters:
    Red_CSS(k,:) = params_red;
    Green_CSS(k,:) = params_green;
    Blue_CSS(k,:) = params_blue;
    
    r_alpha(1,k) = params_red(1,1);
    r_mean(1,k) = params_red(1,2);
    r_sigma(1,k) = params_red(1,3);
    % Green channel
    g_alpha(1,k) = params_green(1,1);
    g_mean(1,k) = params_green(1,3);
    g_sigma(1,k) = params_green(1,3);

    % Blue channel
    b_alpha(1,k) = params_blue(1,1);
    b_mean(1,k) = params_blue(1,2);
    b_sigma(1,k) = params_blue(1,3);
end


disp('Job terminated');
