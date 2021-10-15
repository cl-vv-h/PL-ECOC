function results = predict_(X,model,way4dis)
    
    pred_code = ECOC_predcode(X,model);
    %disp(size(pred_code));
    pred_code = (pred_code-0.5)*2;
    codes = (model.code_matrix-0.5)*2;
    matrix_H = model.matrix_H;
    [q,l] = size(codes);
    train_num = size(pred_code,1);
    if(way4dis)
        %disp(train_num);
        pred_matrix = zeros(train_num,q);
        results=[];
        for k=1:train_num
            res = zeros(q,1);
            for i=1:q
                for j=1:l
                    pred_matrix(i,j) = matrix_H(i,j) * exp(-pred_code(k,l)*codes(i,j));
                end
            end
            sum_pred = sum(pred_matrix,2);
            [maxVal maxInd] = max(sum_pred);
            res(maxInd) = 1;
            results = [results,res];
        end
    else
        results=[];
        for k=1:train_num
            res = zeros(q,1);
            tmp = codes - pred_code(k,:);
            distance = sum(abs(tmp).^2,2).^(1/2);
            [maxVal maxInd] = max(distance);
            res(maxInd)= 1;
            results = [results,res];
        end
    end
    results = results.';
end
    function pred_code = ECOC_predcode(X,model)
    pred_y = [];
    
    l = model.length;
    models = model.models;
    for i=1:l
        pred_i = predict(models{i,1},X);
        %disp(size(pred_i));
        pred_y = [pred_y,pred_i];
    end
    pred_code = pred_y;
    end