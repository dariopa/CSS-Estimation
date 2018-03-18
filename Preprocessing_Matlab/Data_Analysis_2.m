%% Analysis of hyperspectral images

clc;	% Clear command window.
clear;	% Delete all variables.
close all;	% Close all figure windows except those created by imtool.
imtool close all;	% Close all figure windows created by imtool.


%% Read the image
load D:/Images/1.mat;
imshow(rgb);

%% Plot the RGB values
r = rgb(:,:,1);
g = rgb(:,:,2);
b = rgb(:,:,3);

figure
hold on
plot(r(500,:),'red');
plot(g(500,:),'green');
plot(b(500,:),'blue');
hold off


z = 80; % rotation around z
y = 20; % rotation around y
figure
subplot(2,2,1)
surf(rgb(:,:,1));
view(z,y)
shading interp

subplot(2,2,2)
surf(rgb(:,:,2))
view(z,y)
shading interp

subplot(2,2,3)
surf(rgb(:,:,3))
view(z,y)
shading interp

colormap(jet)


%% Plot Rad: Spectrum

rad = permute(rad,[2,1,3]);
rad_new = zeros(size(rad,1),size(rad,2), size(rad,3));

for i = 1:size(rad_new,1)
        rad_new(size(rad_new,1)+1-i,:,:) = rad(0+i,:,:);
end

x_start = 1390;
y_start = 1;
x_split = 0;
y_split = 40;

a = 1; %just legendinfo
figure 
hold on
for i = x_start:(x_start+x_split)
    for j = (y_start):(y_start+y_split)
        x = 401:10:710;
        y = reshape(rad_new(i,j,:),[],31);
        for n = 1:31
            plot(x,y,'-O')            
        end
        legendinfo{a} = ['pixel (' num2str(i) ', ' num2str(j) ')'];
        a = a + 1;
    end
end

legend(legendinfo)
hold off
