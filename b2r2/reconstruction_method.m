classdef reconstruction_method
    properties
        x_lambda
        X_ft_lambda
        M
        of
        rho
        w
        Wm
    end
    methods
        function obj = reconstruction_method(x_lambda, X_ft_lambda, M, of, w,Wm)
            if nargin == 0
                obj.x_lambda = [];
                obj.X_ft_lambda = [];
                obj.M = [];
                obj.of = [];
                obj.rho = [];
                obj.w = [];
                obj.Wm=[];
                return;
            end
            obj.x_lambda = x_lambda;
            obj.X_ft_lambda = X_ft_lambda;
            obj.M = M;
            obj.of = of;
            obj.rho = 1 / of;
            obj.w = w;
            obj.Wm = Wm;
        end
        function x_lambda = BBRR(obj, Lambda, L)
            x_lambda = obj.x_lambda;
            rho = obj.rho;
            M = obj.M;
            w = obj.w;
            of = obj.of;
            N = floor(M/2);
            Wm = obj.Wm;

            beta = 1.05;
            decay = 0.999;
            reg = 1e-6;
            momentum = 0.9;
            epsilon = min(1e-3, max(1e-6, 1e-4 * Lambda * log(M + 1) * sqrt(of)));

            Ws=2*Wm*of;
            F = dftmtx(L);
            [~, idx_1] = min(abs(w - 2*pi*Wm));
            [~, idx_2] = min(abs(w - 2*pi*(Ws-Wm)));
            diagonal = ones(L,1);
            diagonal(1:idx_1+1) = 0;
            diagonal(idx_2:end) = 0;
            D = diag(diagonal);
            BB = D * F;
            dev_matrix = BB' * BB;

            if of==2
                step_size=2/L;
            
            else
                step_size = 2 / (idx_2 - idx_1);
            end
            

            mu = step_size;
            mom = momentum;
            change = 0;
            d1 = 0; d2 = 0;

            delta_rec = high_pass(-x_lambda, beta, rho, w,Wm);
            delta_rec(1:floor(L/2)-N) = 0;
            delta_rec(floor(L/2)+N+2:end) = 0;
            delta_rec = double(delta_rec);

            for iteration = 1:1000000
                [delta_rec, change] = PGD(L,x_lambda, delta_rec, change, N, dev_matrix, reg, mu, mom, []);
                mu = decay * mu;
                mom = decay * mom;

                if mod(iteration,5) == 0
                    d1_new = delta_rec(floor(L/2)-N+1);
                    d2_new = delta_rec(floor(L/2)+N+1);
                    if max(abs(d1_new - d1), abs(d2_new - d2)) < epsilon
                        delta_rec = quant_delta(delta_rec, Lambda);
                        if N < 1
                            x_lambda(floor(L/2)+1) = x_lambda(floor(L/2)+1) + delta_rec(floor(L/2)+1);
                            break;
                        else
                            mu = step_size;
                            mom = momentum;
                            x_lambda(floor(L/2)-N+1) = x_lambda(floor(L/2)-N+1) + delta_rec(floor(L/2)-N+1);
                            x_lambda(floor(L/2)+N+1) = x_lambda(floor(L/2)+N+1) + delta_rec(floor(L/2)+N+1);
                            N = N - 1;
                        end
                        d1 = delta_rec(floor(L/2)-N+1);
                        d2 = delta_rec(floor(L/2)+N+1);
                    else
                        d1 = d1_new;
                        d2 = d2_new;
                    end
                end
            end
        end
    end
end
function x_rec_hpf = high_pass(x, beta, rho, w,Wm)
    Ws = 2*Wm*(1/rho);
    X_ft_hpf = fft(ifftshift(x));
    [~, idx_1] = min(abs(w - 2*pi*Wm));
    [~, idx_2] = min(abs(w - (2*pi*(Ws-Wm))));
    X_ft_hpf(1:idx_1) = 0;
    X_ft_hpf(idx_2+1:end) = 0;
    x_rec_hpf = ifftshift(ifft(X_ft_hpf));
end



function [delta_rec, change] = PGD(L,x_lambda, delta_rec, change, N_lambda, dev_matrix, reg, step_size, momentum, idx_del)
    x_lambda = x_lambda(:);
    delta_rec = delta_rec(:);
    vector = x_lambda + delta_rec;
    % Debug print
    % disp(['Size of dev_matrix: ', mat2str(size(dev_matrix))]);
    % disp(['Size of vector: ', mat2str(size(vector))]);
    grad = dev_matrix * vector + reg * delta_rec;
    change = momentum * change + step_size * real(grad);
    delta_rec = delta_rec - change;
    delta_rec(1:floor(L/2) - N_lambda) = 0;
    delta_rec(floor(L/2) + N_lambda + 2:end) = 0;
    if ~isempty(idx_del)
        delta_rec(idx_del) = 0;
    end
end



function delta_quant_rec = quant_delta(delta_rec, Lambda)
    delta_quant_rec = round(delta_rec / (2*Lambda)) * 2 * Lambda;
end

