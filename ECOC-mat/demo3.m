load('d.mat'); % Loading the file containing the necessary inputs for calling the SSPL function
% Set the corresponding coefficients
k = 3;
r = 0.7;
alpha = 0.7;
beta = 0.25;

train_data = double(train_data);
partial_data = train_data(1:230,:);
partial_label = train_label(1:230,:);
unlabeled_data = train_data(231:250,:);
model = Propagation(partial_data, partial_label, unlabeled_data, k, alpha, beta);

sets_Data = [model.PartialData;model.UnlabelData];
sets_label= [model.Fp;model.Fu];

ECOC_model = ECOC(sets_Data,sets_label,15,50);
%disp(ECOC_model.length)

pred_res = predict_(double(test_data),ECOC_model,0);
test_num = size(test_data,1);
count = 0.0;
disp(size(pred_res));
% test = test_target.';
for i=1:test_num
    if(pred_res(i,:)==test_label(i,:))
        count = count + 1;
    end
end

acc = count/test_num;
disp(acc);