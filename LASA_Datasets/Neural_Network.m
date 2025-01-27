% Get Training Data
%function [Data, Data_sh, att, x0_all, data, dt] = load_LASA_dataset_shape_DS(number_model, sub_sample, nb_trajectories)
Data = load_LASA_dataset_shape_DS(4,1,[1:6]);
save("Dbended_train.mat")
clear
Data = load_LASA_dataset_shape_DS(4,1,[7:7]);
save("Dbended_test.mat")
clear
% Data = load_LASA_dataset_shape_DS(24,1,[7:7]);
% save("G_shape_val.mat")