function [D,Dt] = defDDt3D

D = @(U) ForwardD(U);
Dt = @(X,Y,Z) Dive(X,Y,Z);


function [Dux,Duy,Duz] = ForwardD(U)
% [ux,uy,uz] = D u

Dux = cat(1, diff(U,1,1), U(1,:,:) - U(end,:,:));
Duy = cat(2, diff(U,1,2), U(:,1,:) - U(:,end,:));
Duz = cat(3, diff(U,1,3), U(:,:,1) - U(:,:,end));


function DtXYZ = Dive(X,Y,Z)
% DtXYZ = D_1' X + D_2' Y + D_3' Z

DtXYZ =         cat(1, X(end,:,:) - X(1,:,:), -diff(X,1,1));
DtXYZ = DtXYZ + cat(2, Y(:,end,:) - Y(:,1,:), -diff(Y,1,2));
DtXYZ = DtXYZ + cat(3, Z(:,:,end) - Z(:,:,1), -diff(Z,1,3));
DtXYZ = DtXYZ(:);
