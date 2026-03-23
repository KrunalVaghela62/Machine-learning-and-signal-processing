clc; clear; close all;

%% Symbol set and parameters
S = [-2 -1 1 2];
Delta = 0.33;

R = 3;   % major radius
r = 1;   % minor radius

%% All 2D symbol vectors
[A,B] = meshgrid(S,S);
Sym = [A(:) B(:)]';   % 2 x 16

%% Linear
X_lin = Sym;

%% OFDM (2-IDFT)
F = 1/sqrt(2)*[1 1; 1 -1];
X_ofdm = F * Sym;

%% Modulo
modfun = @(x) x - Delta*round(x/Delta);
Y_lin  = modfun(X_lin);
Y_ofdm = modfun(X_ofdm);

%% --------- Torus distance function ----------
torus_dist = @(u,v) sqrt( ...
    min(abs(u(1)-v(1)), Delta-abs(u(1)-v(1)))^2 + ...
    min(abs(u(2)-v(2)), Delta-abs(u(2)-v(2)))^2 );

%% Compute minimum distances
N = size(Y_lin,2);

dmin_lin = inf;
dmin_ofdm = inf;

idx_lin = [1 1];
idx_ofdm = [1 1];

for i = 1:N
    for j = i+1:N
        dL = torus_dist(Y_lin(:,i), Y_lin(:,j));
        dO = torus_dist(Y_ofdm(:,i), Y_ofdm(:,j));
        
        if dL < dmin_lin
            dmin_lin = dL;
            idx_lin = [i j];
        end
        
        if dO < dmin_ofdm
            dmin_ofdm = dO;
            idx_ofdm = [i j];
        end
    end
end

fprintf('Min distance (linear, torus) = %.6f\n', dmin_lin);
fprintf('Min distance (OFDM, torus)   = %.6f\n', dmin_ofdm);

%% Map square -> donut
map_torus = @(u,v) deal( ...
    (R + r*cos(2*pi*u/Delta)).*cos(2*pi*v/Delta), ...
    (R + r*cos(2*pi*u/Delta)).*sin(2*pi*v/Delta), ...
     r*sin(2*pi*u/Delta) );

[Xl,Yl,Zl] = map_torus(Y_lin(1,:),  Y_lin(2,:));
[Xo,Yo,Zo] = map_torus(Y_ofdm(1,:), Y_ofdm(2,:));

%% Closest pairs in 3D (for visualization)
[i1,j1] = deal(idx_lin(1), idx_lin(2));
[i2,j2] = deal(idx_ofdm(1), idx_ofdm(2));

%% Draw donut surface
[U,V] = meshgrid(linspace(-Delta/2,Delta/2,60));
[Xs,Ys,Zs] = map_torus(U,V);

figure('Color','w');

subplot(1,2,1)
surf(Xs,Ys,Zs,'FaceAlpha',0.2,'EdgeColor','none'); hold on;
plot3(Xl,Yl,Zl,'ro','MarkerSize',10,'MarkerFaceColor','r');
plot3([Xl(i1) Xl(j1)], [Yl(i1) Yl(j1)], [Zl(i1) Zl(j1)], ...
      'k','LineWidth',3);
axis equal; grid on;
title(sprintf('Linear (d_{min}=%.4f)', dmin_lin));
view(3)

subplot(1,2,2)
surf(Xs,Ys,Zs,'FaceAlpha',0.2,'EdgeColor','none'); hold on;
plot3(Xo,Yo,Zo,'bo','MarkerSize',10,'MarkerFaceColor','b');
plot3([Xo(i2) Xo(j2)], [Yo(i2) Yo(j2)], [Zo(i2) Zo(j2)], ...
      'k','LineWidth',3);
axis equal; grid on;
title(sprintf('OFDM (d_{min}=%.4f)', dmin_ofdm));
view(3)
