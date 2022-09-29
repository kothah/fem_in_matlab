%% Author   Hardik Kothari
%           Euler Institute
%           Universita della Svizzera italiana
%           Lugano, Switzerland
%           hardik.kothari@usi.ch	

%% Purpose: cutFEM code for solving 1D Possion equation 
% solve -\nabla^2 u = f on [0, 1-\epsilon]
% mesh is created on domain [0,1]

% 0 is fitted with mesh
% 1-\epsilon is unfitted, 1-epsilon = h*(numElem-1)+\delta
% mesh size is h
% epsilon is moved from 1e-10 to 1e-1 and from 1e-1 to 9e-1 
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
LagBasis.numBasis = 2;

LagBasis.patchDNElem = @(xi)[-0.5  0.5 0];
LagBasis.patchNNeigh = @(xi)[   0 -0.5 0.5]; 
LagBasis.numpatchBasis = 3;

%% counters
count=0;

%% paramters
gamma = 100;
eps_g = 1e-2;

deltabyh = [logspace(-10,-1),0.2:.1:0.9];

for curr_dbh = deltabyh
  count = count+1;

  numElem = 50; %[5 10 20 40 80 160 320 640]
  % domain = [0,1]
  % mesh = [0, 1 + delta];
  % last element is unfitted

  %delta = 1/numElem;
  startPoint= 0;
  endPoint = 1;% + delta;
  h = (endPoint-startPoint)/numElem;
  delta = curr_dbh*h;

  %% mesh
  NodeArray = (startPoint:(endPoint)/(numElem):endPoint)';
  ElemArray = [1:numel(NodeArray)-1; 2:numel(NodeArray)]';

  numNodes = numel(NodeArray);

  % cut config;
  cutElemArray = ElemArray(end,:);
  cutPoint = NodeArray(end-1) + delta;
  num_uncutElems = numElem - 1;
  num_cutElems = 1;

  %% FE Space
  % quadratures
  order = 5;
  Quadrature = getQuadratureRule(order);

  %% allocate space
  M = sparse(numNodes,numNodes);
  K = sparse(numNodes,numNodes);
  F = zeros(numNodes,1);

  %% assembly for uncutElements
  for i_elem = 1:num_uncutElems
    localElem = ElemArray(i_elem,:);
    localNodes = NodeArray(localElem',:);

    Ke = zeros(LagBasis.numBasis);
    Me = zeros(LagBasis.numBasis);
    fp = zeros(LagBasis.numBasis,1);

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
      for i_basis = 1:LagBasis.numBasis
        for j_basis = 1:LagBasis.numBasis
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

  %% assembly for cut element
  for i_elem = 1:num_cutElems
    localElem = cutElemArray(i_elem,:);
    localNodes = NodeArray(localElem',:);
    orig_area = abs(localNodes(2)-localNodes(1));

    % replacing full element with cut element
    cutNodes = [localNodes(1); cutPoint];
    cut_area = abs(cutNodes(2)-cutNodes(1));

    Ke = zeros(LagBasis.numBasis);
    Me = zeros(LagBasis.numBasis);
    fp = zeros(LagBasis.numBasis,1);

    for i_qp = 1:Quadrature.numQP
      xi = Quadrature.Points(i_qp);
      w =  Quadrature.Weights(i_qp)*(cut_area/orig_area);

      N = LagBasis.N(xi);
      dN = LagBasis.DN(xi);

      % affine map
      % QP in current element
      currN = N'*cutNodes;

      % we have to map it back to reference element
      newXi = inverseFEmap(currN,localNodes,LagBasis.N,LagBasis.DN);

      % evaluate basis at the new quadrature points
      NewN = LagBasis.N(newXi);
      NewdN = LagBasis.DN(newXi);

      Jacobian = NewdN' * localNodes;
      detJ = abs(det(Jacobian));
      JxW = detJ*w;

      % evaluate rhs at quadrature points
      f_val = f_rhs(NewN'*localNodes);

      % loop over basis
      for i_basis = 1:LagBasis.numBasis
        for j_basis = 1:LagBasis.numBasis
          Me(i_basis,j_basis) = Me(i_basis,j_basis) + JxW * NewN(i_basis)*NewN(j_basis);
          Ke(i_basis,j_basis) = Ke(i_basis,j_basis) + JxW * (invJ*NewdN(i_basis))*(invJ*NewdN(j_basis));
        end
        fp(i_basis) = fp(i_basis) + (JxW * f_val * NewN(i_basis));
      end
    end
    M(localElem, localElem) = M(localElem, localElem) + Me;
    K(localElem, localElem) = K(localElem, localElem) + Ke;
    F(localElem) = F(localElem) + fp;
  end

  %% apply Diriclet BC with penalty method
  localElem = cutElemArray(i_elem,:);
  localNodes = NodeArray(localElem',:);
  orig_area = abs(localNodes(2)-localNodes(1));

  % replacing full element with cut element
  cutNodes = [localNodes(1); cutPoint];
  cut_area = abs(cutNodes(2)-cutNodes(1));

  Ke = zeros(LagBasis.numBasis);
  fp = zeros(LagBasis.numBasis,1);

  % affine map
  % QP in current element
  currN = cutPoint;

  % we have to map it back to reference element
  newXi = inverseFEmap(currN,localNodes,LagBasis.N,LagBasis.DN);

  NewN = LagBasis.N(newXi);
  NewdN = LagBasis.DN(newXi);

  Jacobian = NewdN' * localNodes;

  detJ = abs(det(Jacobian));
  JxW = 1;

  u_D = u_exact(NewN'*localNodes);

  for i_basis = 1:LagBasis.numBasis
    for j_basis = 1:LagBasis.numBasis
      Ke(i_basis,j_basis) = Ke(i_basis,j_basis)  + (gamma/h) * NewN(i_basis)*NewN(j_basis);
    end
    fp(i_basis) = fp(i_basis) + ((gamma/h) * u_D * NewN(i_basis));
  end
  K(localElem, localElem) = K(localElem, localElem) + Ke;
  F(localElem) = F(localElem) + fp;

  %% apply exact BC
  F(1) = u_exact(NodeArray(1));
  K(1,:)=0;
  K(1,1)=1;
  
  cond_num_before_stab(count) = condest(K);

  %% apply ghost penalty stabilization 
  patch_nodes = [ElemArray(end-1,:) ElemArray(end,end)];
  patch_NArray = NodeArray(patch_nodes);

  currNArray = NodeArray(ElemArray(end-1,:));
  neighNArray = NodeArray(ElemArray(end,:));
  
  currN = NodeArray(ElemArray(end,1));

  newXiElem  = inverseFEmap(currN,currNArray,LagBasis.N,LagBasis.DN);
  sideDNElem = LagBasis.patchDNElem(newXiElem);
  JacElem  = sideDNElem*patch_NArray;
  gradPhiElem = JacElem\sideDNElem;

  newXiNeigh = inverseFEmap(currN,neighNArray,LagBasis.N,LagBasis.DN);  
  sideDNNeigh = LagBasis.patchNNeigh(newXiNeigh);
  JacNeigh = sideDNNeigh*patch_NArray;
  gradPhiNeigh = JacNeigh\sideDNNeigh;

  diff = gradPhiElem - gradPhiNeigh;
  
  Ke = zeros(LagBasis.numpatchBasis);

  for i_basis = 1:LagBasis.numpatchBasis
    for j_basis = 1:LagBasis.numpatchBasis
      Ke(i_basis,j_basis) = Ke(i_basis,j_basis)  + (eps_g*h) * diff(i_basis)*diff(j_basis);
    end
  end
  K(patch_nodes, patch_nodes) = K(patch_nodes, patch_nodes) + Ke;

  cond_num_after_stab(count) = condest(K);

end
loglog(deltabyh,cond_num_after_stab,'--o')
hold on;
loglog(deltabyh,cond_num_before_stab,'--x')

lgd = legend("\kappa(A_{stab})","\kappa(A)",'Location','northeast');
title('Condition number of the system matrix with and without stabilization')
xlabel('\delta/h','FontSize',14)  % x-axis label
ylabel('\kappa','FontSize',14)                         % y-axis label
lgd.FontSize = 14;

%% Map quadrature points back to reference element
function [new_qp] = inverseFEmap(physical_qp,OrigElem,c_phi,c_dphi)
% cutElem = cut element co-ordinates
% OrigElem = original element co-ordinates
% xi  = qp(:,1);
% eta = qp(:,2);
%
% curr_phi  = c_phi(xi,eta);
%
% %physical_qp = curr_phi*OrigElem;
%
% physical_qp = curr_phi * cutElem;

% initial guess
% point P is on the reference element
% TODO: for 3D
P  = 0;
dp = 1;

count=0;
while norm(dp) > 1e-8  && count < 50

  physical_guess = c_phi(P')' * OrigElem;

  delta = (physical_qp - physical_guess)';

  if( norm(delta,2) <= 1e-8 )
    break;
  end
  curr_dphi = squeeze(c_dphi(P'));

  J = curr_dphi' * OrigElem;
  J = J';

  dp =  J\delta;
  P = P + dp;
  count = count+1;
end
new_qp = P';
end

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
