function path = dirpath(path)
    path = strrep(path, '\\', '/');
    offset = strfind(path, '/');
    path = path(1:offset(end));
end