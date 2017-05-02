function [ matr ] = plot_per_group( obj_ell )
max_iter = 0;
for i=1:length(obj_cell)
    if length(obj_cell{i}) > max_iter
        max_iter = length(obj_cell{i});
    end
end
% it is possible that not all the iterations are the same length so we
% pad them into this matrix so we can plot together
matr = zeros(length(obj_cell), max_iter);
for i=1:length(obj_cell)
    matr(i, 1:length(obj_cell{i})) = obj_cell{i};
end
plot(matr');
end

