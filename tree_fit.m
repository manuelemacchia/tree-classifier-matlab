function tree = tree_fit(X_train, X_type, y_train)
    % Build a decision tree classifier from the training set.
    %
    % X_train is a matrix that contains data to train the classifier. Each
    % row is an instance and each column corresponds to a certain
    % categorical or numerical feature.
    %
    % X_type is a vector of length equal to the number of columns in
    % X_train. The i-th element of X_type specifies the type of feature of
    % the i-th column of the training matrix. Type 1 features are
    % categorical, while type 2 features are numerical. This information is
    % needed when building the decision tree.
    %
    % y_train is a vector of length equal to the number of rows in X_train.
    % The i-th element of y_train specifies the class of the i-th instance
    % in X_train (0 or 1).
    %
    % Return a tree in the form of a cell array.
    %
    % Example:
    %                      HEADACHE?
    %                      /      \
    %                 yes /        \ no
    %                    /          \
    %               TEMPERATURE?    class = 0
    %              /     |      \
    %   very high /      | high  \ normal
    %            /       |        \
    %     class = 1  class = 1  class = 0
    %
    % Equivalent cell array:
    % tree = {HEADACHE, {yes, {TEMPERATURE, {very high, class = 1},
    %                                       {high,      class = 1},
    %                                       {low,       class = 0}
    %                         },
    %                   {no, class = 0}
    %        }

    igr = []; % Vector of IGRs used for finding the best feature
    t = []; % Vector of thresholds for numerical features
    for i = 1:size(X_train, 2) % Iterate over all features
        % We compute the joint probability density matrix J.
        % J has as many rows as there are classes, in our case two, and as
        % many columns as there are splits of the feature that's being
        % considered (i.e. the number of values the feature can have).
        % An example of a J matrix follows.
        % J = [
        %    very_high  high  low
        %    1/7        2/7   0;    class=yes
        %    0          1/7   3/7   class=no
        % ];

        if X_type(i) == -1
            % Skip over features that have already been used.
            continue

        elseif X_type(i) == 1 % Categorical feature
            values = unique(X_train(:,i));
            n_values = size(values, 1); % Compute the number of possible values
            J = zeros(2, n_values); % J always has two rows (two classes)

            n_vectors = size(X_train, 1);
            for j = 1:n_vectors
                % We use y_train(j)+1 to find the row (first row for class 0,
                % second row for class 1) of the joint probability matrix.
                % We use the mask values==X_train(j,i) to find the column of
                % the joint probability matrix. This expression returns the
                % index of the feature value.
                J(y_train(j)+1, values==X_train(j,i)) = J(y_train(j)+1, values==X_train(j,i)) + 1;
            end

            J = J/n_vectors;

            igr(i) = information_gain_ratio(J);

        elseif X_type(i) == 2 % Numerical feature
            % In this case, the split should be made at a certain threshold
            % value. To find this value, all possible splits should be
            % evaluated to find the one that maximizes the IGR.

            for t_i = unique(X_train(:,i))' % Iterate over candidate thresholds
                % We compute the joint probability density matrix J.
                % J will be a two by two matrix, since we handle two
                % classes and numerical features always have binary splits.
                J = [
                    sum(X_train(:,i)<=t_i & (y_train==0)')    sum(X_train(:,i)>t_i & (y_train==0)');
                    sum(X_train(:,i)<=t_i & (y_train==1)')    sum(X_train(:,i)>t_i & (y_train==1)')
                ];

                % We compute the IGR for the current threshold and check
                % whether it's better than previous ones. If so, store it.
                igr_split = information_gain_ratio(J);

                % The following condition handles first iterations of this
                % loop (when igr(i) does not exist yet) and checks if the
                % current split is better than the current best one.
                if numel(igr) < i || isnan(igr(i)) || igr_split > igr(i)
                    igr(i) = igr_split;
                    t(i) = t_i;
                end
            end
        end
    end

    % We choose the feature with the maximum IGR value, and we store its
    % index in the tree.
    [~, idx] = max(igr);
    tree = {idx};

    % Compute the subsets in which X_train should be split.
    if X_type(idx) == 1 % Categorical features are split over all possible categories.
        splits = unique(X_train(:,idx));
    elseif X_type(idx) == 2 % Numerical features are split over a certain threshold (binary split).
        splits = [t(idx) t(idx)];
        % NOTE: This introduces redundant information in the tree
        % structure, because the threshold is stored twice every time a
        % numerical feature is chosen. We accept this in order to
        % conveniently store in the same fashion numerical and categorical
        % feature nodes in the tree.
    end

    % Iterate over splits (subsets).
    for i = 1:numel(splits)
        % Compute the classes vector for the current split using a mask.
        if X_type(idx) == 1
            y_split = y_train(X_train(:,idx)==splits(i),:);
        elseif X_type(idx) == 2
            if i == 1 % First split: less than or equal to threshold
                y_split = y_train(X_train(:,idx)<=t(idx));
            elseif i == 2 % Second split: more than threshold
                y_split = y_train(X_train(:,idx)>t(idx));
            end
        end

        % Compute what are the classes associated to the current split.
        classes_split = unique(y_split);

        % First stopping criterion: when a subset contains only instances
        % of the same class, create a leaf node labeled by this class.
        if numel(classes_split) == 1
            tree{i+1} = {splits(i), classes_split};

        % Second stopping criterion: when a subset contains vectors of
        % different classes, but all the features have been considered,
        % create a leaf node labeled by the most common class among the
        % subset vectors.
        elseif size(X_train, 2) == 1
            tree{i+1} = {splits(i), mode(y_split)};

        % If no stopping criterion is satisfied, recursively build the rest
        % of the tree.
        else
            if X_type(idx) == 1
                X_split = X_train(X_train(:,idx)==splits(i),:);
            elseif X_type(idx) == 2
                if i == 1 % First split: less than or equal to threshold
                    X_split = X_train(X_train(:,idx)<=t(idx),:);
                elseif i == 2 % Second split: more than threshold
                    X_split = X_train(X_train(:,idx)>t(idx),:);
                end
            end

            % Mark the feature used in this node as unavailable for the
            % deeper nodes in this branch of the tree.
            X_split_type = X_type;
            X_split_type(idx) = -1;

            % Build the rest of the tree.
            tree{i+1} = {splits(i), tree_fit(X_split, X_split_type, y_split)};
        end
    end
end

function igr = information_gain_ratio(J)
    % The Information Gain Ratio (IGR) is defined as the ratio between the
    % mutual information and the entropy of the conditioning variable:
    %   I(A; B) / H(B).
    % with I(A; B) mutual information, defined as:
    %   I(A; B) = H(A) - H(A|B)

    % Input J is the joint probability distribution matrix, which is
    % structured like the following example.
    %
    % A, B random variables with possible values {a1, a2, a3} and
    % {b1, b2} respectively.
    %       A=a1    A=a2    A=a3 (M=3 values)
    % B=b1  0.1     0.3     0.1
    % B=b2  0.2     0.2     0.1
    % (N=2 values)

    i = entropy(sum(J, 2)) - conditional_entropy(J);
    igr = i/entropy(sum(J, 1));
end

function h = conditional_entropy(J)
    % Compute the conditional entropy H(A|B).
    % Input J is the joint probability distribution matrix.

    h = 0;
    for i = 1:size(J, 1) % Iterate over N rows (i index)
        for j = 1:size(J, 2) % Iterate over M columns (j index)
            if J(i,j) ~= 0
                condp = J(i,j) / sum(J(:,j));
                h = h + J(i,j) * log2(1/condp);
            end
        end
    end
end

function h = entropy(p)
    % Input p is the vector of probabilities associated to the possible
    % values of the random variable.
    % e.g. A is a discrete random variable, and takes values a_i (with i
    % from 1 to M) with probabilities p(a_i)

    s = p.*log2(p);
    s(isnan(s)) = 0; % Consider 0log0 = 0
    h = -sum(s);
end