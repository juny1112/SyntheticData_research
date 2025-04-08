%% 2RC model function

function V_est = RC_model_2(X, t_vec, I_vec)
    
    % X: [R0, R1, R2, tau1, tau2]
    R0   = X(1);
    R1   = X(2);
    R2   = X(3);
    tau1 = X(4);
    tau2 = X(5);

    % dt 계산
    dt = [1; diff(t_vec)]; % 현재k와 이전k-1사이의 차이 (첫번째 시간간격을 1로 둠)

    N = length(t_vec);
    V_est = zeros(N, 1);

    for k = 1:N
        % (1) OCV = 0 가정
        OCV = 0;

        % (2) R0 전압강하
        IR0 = R0 * I_vec(k);

        % (3) RC1, RC2 업데이트
        alpha1 = exp(-dt(k)/tau1);
        alpha2 = exp(-dt(k)/tau2);


        if k == 1
            % k-1번째 = 0번째 -> Vrc(0)=0, I(0)=I(1) 이라고 가정
            Vrc1 = R1 * I_vec(1) * (1 - alpha1);
            Vrc2 = R2 * I_vec(1) * (1 - alpha2);
        else
            % 이후는 기존 공식
            Vrc1 = Vrc1*alpha1 + R1*(1 - alpha1)*I_vec(k-1);
            Vrc2 = Vrc2*alpha2 + R2*(1 - alpha2)*I_vec(k-1);
        end

        % (4) 최종 전압
        V_est(k) = OCV + IR0 + Vrc1 + Vrc2;
    end
end