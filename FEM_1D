%% Author   Hardik Kothari
%           Euler Institute
%           Universita della Svizzera italiana
%           Lugano, Switzerland
%           hardik.kothari@usi.ch	

%% Purpose FEM code for solving 1D Possion equation 
% solve -\nabla^2 u = f

clear
clc
close all

%% exact Sol
u_exact = @(x) sin(pi*x).*cos(pi*x);
gradu_exact = @(x) pi*cos(pi*x).^2 - pi*sin(pi*x).^2;

f_rhs = @(x) 4*pi^2*cos(pi*x).*sin(pi*x);

%% basis functions
LagBasis.N = @(xi) [0.5*(1 - xi); 0.5*(1 + xi)];
LagBasis.DN = @(xi) [-0.5; 0.5];

numBasis = 2;

%% refine to show convergence of errors
for numElem = [5 10 20 40 80 160 320 640]
  %% domain = [startPoint, endPoint]
  startPoint= 0;
  endPoint = 1;

  %% create mesh
  NodeArray = (startPoint:1/(numElem):endPoint)';
  ElemArray = [1:numel(NodeArray)-1; 2:numel(NodeArray)]';

  numNodes = numel(NodeArray);

  %% FE Space
  % quadratures
  order = 5;
  Quadrature = getQuadratureRule(order);

  %% allocate space
  M = sparse(numNodes,numNodes);
  K = sparse(numNodes,numNodes);
  F = zeros(numNodes,1);

  %% assembly
  for i_elem = 1:numElem
    localElem = ElemArray(i_elem,:);
    localNodes = NodeArray(localElem',:);

    Ke = zeros(numBasis);
    Me = zeros(numBasis);
    fp = zeros(numBasis,1);

    for i_qp = 1:Quadrature.numQP
      xi = Quadrature.Points(i_qp);
      w =  Quadrature.Weights(i_qp);

      N = LagBasis.N(xi);
      dN = LagBasis.DN(xi);

      Jacobian = dN' * localNodes;
      invJ = inv(Jacobian);
      detJ = abs(det(Jacobian));

      JxW = detJ*w;

      % evaluate rhs at quadrature points
      f_val = f_rhs(N'*localNodes);

      % loop over basis
      for i_basis = 1:numBasis
        for j_basis = 1:numBasis
          Me(i_basis,j_basis) = Me(i_basis,j_basis) + JxW * N(i_basis)*N(j_basis);
          Ke(i_basis,j_basis) = Ke(i_basis,j_basis) + JxW * (invJ*dN(i_basis))*(invJ*dN(j_basis));
        end
        fp(i_basis) = fp(i_basis) + (JxW * f_val * N(i_basis));
      end
    end

    % global assembly
    M(localElem, localElem) = M(localElem, localElem) + Me;
    K(localElem, localElem) = K(localElem, localElem) + Ke;
    F(localElem) = F(localElem) + fp;
  end

  %% apply Dirichlet BC
  % rhs
  F(1) = u_exact(NodeArray(1));
  F(end) = u_exact(NodeArray(end));
  
  % stiffness matrix
  K(1,:)=0;
  K(end,:)=0;
  K(1,1)=1;
  K(end,end)=1;

  %% solve
  x = K\F;

  %% compute error
  L2 = 0;
  H1 = 0;

  for i_elem = 1:numElem
    localElem = ElemArray(i_elem,:);
    localNodes = NodeArray(localElem',:);
    x_local = x(localElem);

    l2_err=0;
    h1_err=0;

    for i_qp = 1:Quadrature.numQP
      xi = Quadrature.Points(i_qp);
      w =  Quadrature.Weights(i_qp);

      N = LagBasis.N(xi);
      dN = LagBasis.DN(xi);

      Jacobian = dN' * localNodes;

      invJ = inv(Jacobian);

      detJ = abs(det(Jacobian));
      JxW = detJ*w;

      u_val = u_exact(N'*localNodes);
      gradu_val = gradu_exact(N'*localNodes);
      x_qp = N'*x_local;
      gradx_qp = invJ*dN'*x_local;

      func_diff = x_qp-u_val;
      grad_diff = gradx_qp - gradu_val;

      l2_err = l2_err + (JxW * (func_diff'*func_diff));
      h1_err = h1_err + (JxW * (grad_diff'*grad_diff));
    end
    L2 = L2 + l2_err;
    H1 = H1 + h1_err;
  end

  i = i+1;
  L2_fin(i,1) = sqrt(L2); %sqrt(error'*M*error);
  H1_fin(i,1) = sqrt(H1); %sqrt(error'*K*error);

  h(i,1) = 1/numElem;
end

loglog(h,L2_fin,"r*--",h,H1_fin,"bo--",'LineWidth',3)
hold on
loglog( h, 0.5*(L2_fin(1)/(h(1)^2))*h.^(2), 'r--',h,0.5*(H1_fin(1)/h(1))*h.^1, 'b--','Linewidth',2)
lgd = legend("L_2 error","H_1 error",'O(h^1)','O(h^2)','Location','southeast');
xlim([min(h) max(h)])
title('Convergence of P1 FEM - Diffusion Equation')
xlabel('Mesh size')  % x-axis label
ylabel('error')                         % y-axis label
lgd.FontSize = 14;

%% Quadrature rule
function Quadrature = getQuadratureRule(order)
switch order
  case 1
    Quadrature.Points = zeros (1,1);
    Quadrature.Weights = zeros (1,1);

    Quadrature.Points (1,1) = 0;
    Quadrature.Weights (1) = 2;

    Quadrature.numQP = 1;

  case {2,3}
    Quadrature.Points = zeros (2,1);
    Quadrature.Weights = zeros (2,1);

    Quadrature.Points (1,1) = -5.7735026918962576450914878050196e-01;
    Quadrature.Points (2,1) = -Quadrature.Points (1,1);
    Quadrature.Weights (1) = 1;
    Quadrature.Weights (2) = 1;
    
    Quadrature.numQP = 2;

  case {4,5}
    Quadrature.Points = zeros (3,1);
    Quadrature.Weights = zeros (3,1);

    Quadrature.Points (1,1) = -7.7459666924148337703585307995648e-01;
    Quadrature.Points (2,1) = 0;
    Quadrature.Points (3,1) = -Quadrature.Points (1,1);

    Quadrature.Weights (1) = 5.5555555555555555555555555555556e-01;
    Quadrature.Weights (2) = 8.8888888888888888888888888888889e-01;
    Quadrature.Weights (3) = Quadrature.Weights (1);

    Quadrature.numQP = 3;
end
end
