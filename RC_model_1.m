% 1RC model function

function V_est = RC_model_1(X, t_vec, I_vec, OCV)
    
    % X: [R0, R1, tau1]
    R0   = X(1);
    R1   = X(2);
    tau1 = X(3);

    % dt 계산
    dt = [0.1; diff(t_vec)]; % 현재k와 이전k-1사이의 차이 (첫번째 시간간격을 1로 둠)

    N = length(t_vec);
    V_est = zeros(N, 1);

    for k = 1:N
        % R0 전압강하
        IR0 = R0 * I_vec(k);

        % RC1 업데이트
        alpha1 = exp(-dt(k)/tau1);
        if k == 1
            % k-1번째 = 0번째 (RC에 대한 전압강하 없음)
            Vrc1 = 0;
        else
            % 이후는 기존 공식
            Vrc1 = Vrc1*alpha1 + R1*(1 - alpha1)*I_vec(k);
        end

        % 최종 전압
        V_est(k) = OCV(k) + IR0 + Vrc1;
    end
end
