path = '/Users/kristofferroise/project/patient_data_raw/';
group = "DataStOlavs19to28/";
patient = "p28_3115013/";

dataset = ["J65BP3A4_ecg",...
           "J65BP3A8_ecg"];

for data_idx = 1:length(dataset)
    
    hdf_file = HdfImport(strcat(path, group, patient, dataset(data_idx), '.h5'));
    hdf_file_out = strcat(path, group, patient, erase(dataset(data_idx), '_ecg'), '_proc.h5');

    hdf_data = hdf_file;

    aa = imread('Palette.bmp');
    lookup = squeeze(aa(1,:,1));

    fn = fieldnames(hdf_data.CartesianVolumes);

    for i = 1:numel(fn)
    
        if( isnumeric(hdf_data.CartesianVolumes.(fn{i})) )
    
            data = hdf_data.CartesianVolumes.(fn{i});
            sz = size(data);

            %figure(1), imshow (squeeze(data(:,:,round(sz(3)/2)))')

            data_smooth = medfilt3(data,[5 5 5]);
            %figure(2), imshow (squeeze(data_smooth(:,:,round(sz(3)/2)))')

            data_gauss = imgaussfilt3(data_smooth, 1.5);
            %figure (3), imshow (squeeze(data_gauss(:,:,round(sz(3)/2)))')

            data_lookup = lookup(data_gauss(:)+1);
            data_lookup = reshape (data_lookup, size(data_smooth));
            %figure(4), imshow (squeeze(data_lookup(:,:,round(sz(3)/2)))')

            close all hidden;

            hdf_data.CartesianVolumes.(fn{i}) = data_lookup;
    
        end

    end

    HdfExport(hdf_file_out, hdf_data);
    
end
