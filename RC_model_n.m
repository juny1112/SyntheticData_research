% nRC model function

function V_est = RC_model_n(X, t_vec, I_vec, n)
    % parameters X = [R0, R1 … Rn, tau1 … taun]  ← 길이 = 1 + 2n
    R0   = X(1);
    R    = X(2 : n+1);   
    tau  = X(n+2 : 2*n + 1);

    dt   = [1; diff(t_vec)]; % 첫 dt = 1 (무의미·안정화 용도)
    N    = numel(t_vec);
    V_est = zeros(N,1);      
    Vrc   = zeros(n,1);      % 각 RC 전압

    for k = 1:N
        % (0) OCV = 0 가정
        OCV = 0;

        % (1) IR0 
        IR0 = R0 * I_vec(k);

        % (2) Vrc
        alpha = exp(-dt(k) ./ tau);    % n×1 벡터
        if k > 1
            Vrc = Vrc .* alpha + R .* (1 - alpha) .* I_vec(k-1);
        else
            Vrc(:) = 0;                % 초기 단계
        end

        % (3) output voltage
        V_est(k) = OCV + IR0 + sum(Vrc);
    end
end

