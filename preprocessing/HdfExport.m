function HdfExport (filename, data)
% Writes datastructure to HDF
% WARNING: Does not support cells!
% by Fredrik Orderud, 2008

% Write data to file
writeparams(data, '/', filename, false);

function file_created = writeparams (outparams, location, filename, file_created)
f = fieldnames(outparams);
for t=1:length(f)
    % extract field name and value
    name  = f{t};
    value = outparams.(name);
    
    if isstruct(value)
        % recursive parsing of structs
        file_created = writeparams(value, [location name], filename, file_created);
    else
        % write field to file
        details.Location = location;
        details.Name     = name;
        
        %disp(details.Location);
        %disp(details.Name);
        %disp([details.Location '/' details.Name])
        h5create(filename, [details.Location '/' details.Name], value);
        if ~file_created
            h5write(filename, [details.Location '/' details.Name], value);
            file_created = true;
        else
            h5write(filename, [details.Location '/' details.Name], value);
        end
    end
end
