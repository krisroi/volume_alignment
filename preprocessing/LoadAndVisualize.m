path = '/Users/kristofferroise/project/patient_data_raw/';
group = "DataStOlavs19to28/";
patient = "p21_3115006/";

%load the dataset
hdfdata1 = HdfImport(strcat(path, group, patient, 'J65BP1I4_ecg.h5'));
hdfdata2 = HdfImport(strcat(path, group, patient, 'J65BP1I4_proc.h5'));


%sax slice
for i = 1:3
    slice1 = squeeze(hdfdata1.CartesianVolumes.vol01(:,round(end/2),:));
    slice2 = squeeze(hdfdata2.CartesianVolumes.vol01(:,round(end/2),:));
    figure(1); 
    subplot(2,3,1); imshow(slice1, [0 255])
    subplot(2,3,4); imshow(slice2, [0 255])
end


%lax1
slice1 = squeeze(hdfdata1.CartesianVolumes.vol01(round(end/2),:,:));
slice2 = squeeze(hdfdata2.CartesianVolumes.vol01(round(end/2),:,:));
figure(1), subplot(2,3,2), imshow(slice1, [0 255]);
figure(1), subplot(2,3,5), imshow(slice2, [0 255]);

%lax2
slice1 = squeeze(hdfdata1.CartesianVolumes.vol01(:,:,round(end/2)));
slice2 = squeeze(hdfdata2.CartesianVolumes.vol01(:,:,round(end/2)));
figure(1), subplot(2,3,3), imshow(slice1, [0 255]);
figure(1), subplot(2,3,6), imshow(slice2, [0 255]);

%plot ECG data and timestamps
%ecgdata = HdfImport(strcat(path, group, patient, ''));
%figure, plot(ecgdata.ecg_times, ecgdata.ecg_data)

%hold on
%plot (ecgdata.ecg_times,10+zeros(size(ecgdata.ecg_times)), 'xr');

