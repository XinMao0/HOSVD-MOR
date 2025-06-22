n = 12;    % Dimension of tensor A
r = 7;     % Dimension of reduced tensor S
k = 4;     % Order of tensor A
m = 5;
eps = 1e-14;
r_1 = 0;
r_2 = 0;

% Core tensor
S = rand(r, r, r, r);
B = rand(n,m);

% Orthogonal matrices
U_1 = orth(randn(n, r));
U_k = orth(randn(n, r));

% Pack into cell array
U_list = {U_1, U_1, U_1, U_k};

% Reconstruct full tensor
A = reconstruct_hosvd(S, U_list);
A = tensor(A);
S = tensor(S);
C = B; %%C is the controllability matrix 
for j = 0:n-1
  perm_set = permn(1:size(C,2), k-1); %Generate all permutations with repetition
      % For each permutation
    for i = 1:size(perm_set,1)      
       L = A;
       for q = 1:k-1
         L = ttv(L,C(:,perm_set(i,q)),1);
       end
        C = [C vec(L)];
    end
    [u,s,v] = svd(C,'econ');
    s = diag(s); 
    r_1 = my_chop2(s,eps*norm(s));
    C = u(:,1:r_1);
    if r_1 == n
       break
    end
end

%if r_1 == n
    Br = U_1' * B;  
    Cr = Br;
    for j = 0:r-1
        perm_set = permn(1:size(Cr,2), k-1); % Generate all permutations with repetition
        % For each permutation
        for i = 1:size(perm_set,1)      
            L = S;
            for q = 1:k-1
                L = ttv(L, Cr(:,perm_set(i,q)), 1);
            end
            Cr = [Cr vec(L)];
        end
        [u,s,v] = svd(Cr, 'econ');
        s = diag(s); 
        r_2 = my_chop2(s, eps * norm(s));
        Cr = u(:,1:r_2);
        if r_2 == r
            break
        end
    end
%end
