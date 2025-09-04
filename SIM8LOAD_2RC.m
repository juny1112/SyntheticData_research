% ======================================================================
%  SIM ±3~±4 구간 기반 2-RC 피팅 + SOC(90·50·20) 전역 통계
% ======================================================================
clc; clear; close all;

% ── 경로 & 파일 리스트 ───────────────────────────────────────────────
folder_SIM = "G:\공유 드라이브\BSL_Data4\HNE_agedcell_2025_processed\SIM_parsed";
sim_files  = dir(fullfile(folder_SIM,"*_SIM.mat"));
exclude_name = "HNE_Driving_45degC_FC_NBR_5_3_SIM.mat";
sim_files = sim_files(~strcmp({sim_files.name}, exclude_name));

% ── fmincon + MultiStart 설정 ────────────────────────────────────────
ms       = MultiStart("UseParallel",true,"Display","off");
startPts = RandomStartPointSet('NumStartPoints',20);
opt      = optimset('display','off','MaxIter',1e3,'MaxFunEvals',1e4,...
                    'TolFun',1e-14,'TolX',1e-15);

% 2-RC 초기추정값 / 경계 / 선형제약 (τ1<τ2)
para0 = [0.003 0.0005 0.0005 10 100]; 
lb    = [0       0       0       0.01  0.01];
ub    = para0*10;
A_lin = [0 0 0 1 -1];  b_lin = 0;

% ── 누적 컨테이너 ────────────────────────────────────────────────────
all_para_hats = struct;
all_rmse      = struct;
all_summary   = struct;   % ← 여기 저장
target_soc    = [90 50 20];

% ── 메인 루프 (파일 단위) ────────────────────────────────────────────
for f = [2 3 7]  % 예: 2,3,7번 파일만
    % 1) load SIM_table
    load(fullfile(folder_SIM,sim_files(f).name),"SIM_table");
    base_raw   = erase(sim_files(f).name,"_SIM.mat");
    base_field = matlab.lang.makeValidName(base_raw);

    % 2) 중심-3 ~ 중심+4 윈도우 인덱스
    win_idx = [];
    for trg = target_soc
        hit = find(SIM_table.SOC1>=trg & SIM_table.SOC2<=trg,1);
        if isempty(hit), continue, end
        win_idx = [win_idx, max(hit-3,1):min(hit+4,height(SIM_table))];
    end
    win_idx = unique(win_idx,'sorted');
    SIM_sel = SIM_table(win_idx,:);
    nSeg    = height(SIM_sel);
    if nSeg==0
        warning("No segments for %s", base_raw);
        continue
    end

    % 3) fitting 결과 저장 배열
    para_hats = zeros(nSeg,5+3);  % 5 params + RMSE + exitflag + iter
    RMSE_list = zeros(nSeg,1);

    % 4) fit each segment
    for s = 1:nSeg
        t = SIM_sel.time{s};   I = SIM_sel.current{s};
        V = SIM_sel.voltage{s}; O = SIM_sel.OCV_vec{s};

        problem = createOptimProblem('fmincon', ...
          'objective',@(p)RMSE_2RC(V,p,t,I,O), ...
          'x0',para0,'lb',lb,'ub',ub, ...
          'Aineq',A_lin,'bineq',b_lin,'options',opt);

        [Pbest, Fval, exitflg, ~, sol] = run(ms,problem,startPts);

        if ~isempty(sol)
            it = sol(find([sol.Fval]==Fval,1)).Output.iterations;
        else
            it = NaN;
        end

        para_hats(s,:) = [Pbest, Fval, exitflg, it];
        RMSE_list(s)   = Fval;
    end

    % 5) 누적
    all_para_hats.(base_field) = para_hats;
    all_rmse.(base_field)      = RMSE_list;

    % ── 이제 SOC 별 summary 테이블로 정리 ──────────────────────────────
    P_all      = para_hats(:,1:5);                        % [nSeg×5]
    midSOC_all = mean([SIM_sel.SOC1, SIM_sel.SOC2],2);    % [nSeg×1]

    % 그룹 마스크
    m90 = midSOC_all>=80;
    m50 = midSOC_all>=40 & midSOC_all<80;
    m20 = midSOC_all<40;

    % 9×5 테이블 초기화
    rowNames = { ...
      'SOC90_Mean','SOC90_Min','SOC90_Max', ...
      'SOC50_Mean','SOC50_Min','SOC50_Max', ...
      'SOC20_Mean','SOC20_Min','SOC20_Max'};
    T = table( ...
       nan(9,1), nan(9,1), nan(9,1), nan(9,1), nan(9,1), ...
       'VariableNames', {'R0','R1','R2','tau1','tau2'}, ...
       'RowNames',     rowNames           ...
    );

    % 채우기
    groups = {m90,m50,m20};
    r = 1;
    for g = 1:3
      idx = groups{g};
      if any(idx)
        block = P_all(idx,:);
        T{r,:} = mean(block,1);
        T{r+1,:} = min (block,[],1);
        T{r+2,:} = max (block,[],1);
      end
      r = r + 3;
    end

    % 저장
    all_summary.(base_field) = T;
    fprintf('[done] %s → fitted %d segs, stored summary\n', base_raw, nSeg);
end

fprintf("모든 파일 처리 완료!\n");

%% ——— SOC별 파라미터 비교 플롯 ——————————————
customLabels = {
  "10℃ 1C 50cyc", ...
  "10℃ 2C 10cyc", ...
  "fresh" ...
};

% ── SOC별 파라미터 비교 플롯 (2-RC) ─────────────────────────────────
cells  = fieldnames(all_summary);
SOC    = [20 50 90];
pNames = {'R0','R1','R2','tau1','tau2'};

for p = 1:numel(pNames)
    figure('Name',pNames{p}+" vs SOC (2-RC)", 'NumberTitle','off');
    hold on;

    % 모든 셀의 Y값을 모아 최댓값 계산용
    Y_all = [];

    for c = 1:numel(cells)
        T = all_summary.(cells{c});
        Y = [
          T{"SOC20_Mean", pNames{p}}, ...
          T{"SOC50_Mean", pNames{p}}, ...
          T{"SOC90_Mean", pNames{p}}
        ];
        plot(SOC, Y, '-o', 'LineWidth',1.5, 'DisplayName', customLabels{c});
        Y_all = [Y_all; Y];  %#ok<AGROW>
    end

    xlabel('SOC (%)');
    ylabel(pNames{p});
    title(pNames{p}+" vs SOC");
    legend('Location','northeast');
    grid on;

    % y축을 0에서 시작하도록 설정
    ymax = max(Y_all(:));
    ylim([0, ymax * 1.05]);  % 상단에 5% 여유 추가
end

% ── RMSE 계산 보조 함수 ──────────────────────────────────────────────
function cost = RMSE_2RC(V_true, para, t, I, OCV)
    V_est = RC_model_2(para,t,I,OCV);
    cost  = sqrt(mean((V_true - V_est).^2));
end
