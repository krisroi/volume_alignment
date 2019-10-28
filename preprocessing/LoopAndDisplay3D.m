path = '/Users/kristofferroise/project/patient_data/';

%load the dataset
data = HdfImport (strcat(path, 'gr5_STolav5to8/p7_3d/J249J70M_proc.h5'));

%number of loops (how many times the movie repeats)
loopNo = 3;

%number of frames in the recording
frameNo = length(fieldnames(data.CartesianVolumes));

for l = 1:loopNo
    for f = 1:frameNo
        %create the ImageType frame
        volName = sprintf('vol%02d', f);
        image = data.CartesianVolumes.(volName);
        sz = size(image);

        %sax slice
        sliceSax = squeeze(image(:,round(sz(2)/2),:));
        %lax1
        sliceLax1 = squeeze(image(round(sz(1)/2),:,:));
        %lax2
        sliceLax2 = squeeze(image(:,:,round(sz(3)/2)));

        %visual debug
        figure(11), hold on
        set(gcf, 'Position',  [100, 100, 1000, 600])
        subplot(1,3,1), imshow(squeeze(sliceSax), [0 255]); title('y-coordinate');
        subplot(1,3,2), imshow(squeeze(sliceLax1), [0 255]); title('x-coordinate');
        subplot(1,3,3), imshow(squeeze(sliceLax2), [0 255]); title('z-coordinate');
        
        drawnow

    end
end


