%clear all;
n = 9; % number of variables
k = 4;
d = 1; 
rand('state',1); 
A = sptensor(@rand,[n-1 n-1],0.01);
B = sptensor(@rand,[n-1 n-1 n-1],0.005);
C = construct_C(double(A),double(B));
Xn = tenmat(C, 1);
s = svd(double(Xn));
fprintf('number of nonzero svs:')
disp(nnz(s));
fprintf('Mode-%d singular values:\n', 1);
disp(s(1:n));

x0 = rand(n,1);
T = 3;
[t, X] = ode45(@(t, x) tensor_rhs(x, C), [0 T], x0);
y = X(size(t,1),:);
output = norm(X(41,:));

[S_full,U_full]=full_hosvd(C);
U_trunc = cell(1, k);
r = zeros(1, k);
U_trunc{1} = U_full{1}(:,1:(n-d));
U_trunc{2} = U_full{2}(:,1:(n-d));
U_trunc{3} = U_full{3}(:,1:(n-d));
U_trunc{k} = U_full{k}(:,1:(n-d));

U_list = {U_trunc{1}',U_trunc{1}',U_trunc{1}',U_trunc{k}'};
S_trunc = reconstruct_hosvd(C, U_list);
C_trunc = reconstruct_hosvd(S_trunc,U_trunc);
fprintf('Tensor error:\n')
norm(C-C_trunc,'fro')/norm(C,"fro")

U_red = {eye(n-d),eye(n-d),eye(n-d),U_trunc{1}'*U_trunc{k}};
C_red = reconstruct_hosvd(S_trunc,U_red);
x_red = U_trunc{1}'*x0;
[t_red, X_red] = ode45(@(t, x) tensor_rhs(x, C_red), [0 T], x_red);
y_red =X_red(size(t_red,1),:)*U_trunc{1}';
mask = y_red ~= 0;
rel_err = norm(y(mask) - y_red(mask), 'fro') / norm(y(mask), 'fro')


function C = construct_C(A, B)
    n = size(A,1);
    C = zeros(n+1, n+1, n+1, n+1);

    for i = 1:n
        for j = 1:n
            for k = 1:n
                C(i,j,k,i) = B(i,j,k);
            end
        end
    end

    for i = 1:n
        for j = 1:n
            C(i,j,n+1,i) = A(i,j);
        end
    end

    C = symmetrize(C);  % from earlier function if you want symmetry
end

function A_sym = symmetrize(A)
    % Make A symmetric in its first three indices
    A_sym = (A + ...
             permute(A, [1,3,2,4]) + ...
             permute(A, [2,1,3,4]) + ...
             permute(A, [2,3,1,4]) + ...
             permute(A, [3,1,2,4]) + ...
             permute(A, [3,2,1,4])) / 6;
end

function [S_full, U_full] = full_hosvd(C)
    k = ndims(C);                 % number of modes
    U_full = cell(1, k);          % store factor matrices
    rmax = zeros(1, k);           % store ranks per mode

    for n_mode = 1:k
        Cn = tenmat(C, n_mode);            % unfold along mode n
        [Un, ~, ~] = svd(double(Cn), 'econ');  % left singular vectors
        U_full{n_mode} = Un;
        rmax(n_mode) = size(Un, 2);
    end

    U_trans = cell(1, k);
    for n_mode = 1:k
        U_trans{n_mode} = U_full{n_mode}'; % transposed factors
    end
    S_full = reconstruct_hosvd(C, U_trans);
end
