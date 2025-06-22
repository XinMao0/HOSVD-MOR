% Parameters
n = 6;      % Dimension of tensor A
r = 3;      % Dimension of reduced tensor S
k = 4;      % Order of tensor A
T = 15;     % Final time


% Core tensor
sz = repmat(r, 1, k);
S = zeros(sz);
for i=1:r
    S(i,i,i,i)=-rand(1)*6;
end

S(1,1,1,1) = -8.2880;
S(2,2,2,2) = -3.2248;
S(3,3,3,3) = -9.7615;

% Orthogonal matrices
U = [-0.1743    0.0129    0.7769;
   -0.0115   -0.4458    0.2735;
   -0.0802    0.0156   -0.5407;
   -0.5370   -0.1316   -0.1081;
   -0.4111    0.8066    0.0856;
    0.7112    0.3646    0.1017];

% Pack into cell array
U_list = {U, U, U, U};

% Reconstruct full tensor
A = reconstruct_hosvd(S, U_list);
x0 = [0.3341    2.8115   -1.2861   -1.1378   -1.2017   -1.8510]';

% Solve the ODE
[t, X] = ode45(@(t, x) tensor_rhs(x, A), [0 T], x0);

% Plot trajectories

subplot(1,2,1);
plot(t, X, 'LineWidth', 4);
xlabel('Time t');
ylabel('State x(t)');
legend(arrayfun(@(i) sprintf('x_%d', i), 1:n, 'UniformOutput', false));
title(sprintf('Trajectories of original system'));
set(gca,'fontsize', 36) 
grid on;

   
% Solve the ODE
z0=U'*x0;

[tr, Z] = ode45(@(t, x) tensor_rhs(x, S), [0 T], z0);

% Plot trajectories
subplot(1,2,2);
plot(tr, Z, 'LineWidth', 4);
xlabel('Time t');
ylabel('State z(t)');
legend(arrayfun(@(i) sprintf('z_%d', i), 1:n, 'UniformOutput', false));
title(sprintf('Trajectories of reduced system'));
set(gca,'fontsize', 36) 
grid on;


% Compute dx symbolically

% Define symbolic state variables 
syms x [n 1] real  % x = [x1; x2; x3; x4; x5; x6]

% Compute dx symbolically
dx = sym(zeros(n,1));

% Generate all (j1, j2, j3) index combinations
[j1, j2, j3] = ndgrid(1:n, 1:n, 1:n);
index_list = [j1(:), j2(:), j3(:)];

% Polynomial expansion
for i = 1:n
    for row = 1:size(index_list, 1)
        jj = index_list(row, :);
        coeff = A(i, jj(1), jj(2), jj(3));
        monomial = coeff * x(jj(1)) * x(jj(2)) * x(jj(3));
        dx(i) = dx(i) + monomial;
    end
end

% Display the polynomial system
disp('The polynomial system dx/dt is:')
for i = 1:n
    fprintf('dx(%d)/dt = %s\n', i, char(vpa(collect(dx(i), x), 6)));
end
