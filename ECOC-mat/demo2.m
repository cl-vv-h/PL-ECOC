load('sample data.mat'); % Loading the file containing the necessary inputs for calling the SSPL function
% Set the corresponding coefficients
k = 10;
r = 0.7;
alpha = 0.7;
beta = 0.25;
unlabeled_data = zeros(10,38);
model = Propagation(train_data, train_p_target.', unlabeled_data, k, alpha, beta);

sets_Data = [model.PartialData;model.UnlabelData];
sets_label= [model.Fp;model.Fu];

ECOC_model = ECOC(sets_Data,sets_label,17,100);
%disp(ECOC_model.length)

pred_res = predict_(test_data,ECOC_model);
test_num = size(test_data,1);
count = 0.0;
disp(size(pred_res));
test = test_target.';
for i=1:test_num
    if(pred_res(i,:)==test(i,:))
        count = count + 1;
    end
end

acc = count/test_num;
disp(acc);