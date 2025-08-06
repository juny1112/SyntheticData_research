% ======================================================================
%   SIM_rep 기반 2-RC 피팅 + SOC(90·50·30)별 전역 통계
% ----------------------------------------------------------------------
%   • 입력  : *_SIM.mat   (SIM_table , SIM_rep ← 3행)
%   • 출력  : ├ 파일별 2RC 피팅 결과  (base_2RCfit.mat)
%            └ SOC90·50·30 전체 요약  (콘솔)
% ======================================================================
clc; clear; close all;

% ── 경로 ──────────────────────────────────────────────────────────────
folder_SIM = "G:\공유 드라이브\BSL_Data4\HNE_agedcell_2025_processed\SIM_parsed";
sim_files  = dir(fullfile(folder_SIM,"*_SIM.mat"));

% ── 제외 파일 --------------------------------------------------------
exclude_name = "HNE_Driving_45degC_FC_NBR_5_3_SIM.mat";
sim_files = sim_files(~strcmp({sim_files.name}, exclude_name));

% ── MultiStart / fmincon 설정 ---------------------------------------
ms        = MultiStart("UseParallel",true,"Display","off");
startPts  = RandomStartPointSet('NumStartPoints',20);
opt       = optimset('display','off','MaxIter',1e3,'MaxFunEvals',1e4,...
                     'TolFun',1e-14,'TolX',1e-15);

para0 = [0.003 0.0005 0.001 10 100];
lb    = [0 0 0 0.01 0.01];
ub    = para0*10;
A_lin = [0 0 0 1 -1];  b_lin = 0;

% ── 누적 구조체 초기화 ----------------------------------------------
all_para_hats = struct;          % 파일별 파라미터
all_rmse      = struct;          % 파일별 RMSE
soc_pool  = struct('SOC90',[],'SOC50',[],'SOC30',[]);
rmse_pool = struct('SOC90',[],'SOC50',[],'SOC30',[]);

for f = 5%1:numel(sim_files)

    % 1) SIM_rep 로드
    load(fullfile(folder_SIM, sim_files(f).name), "SIM_rep");
    base_raw   = erase(sim_files(f).name,"_SIM.mat");
    base_field = matlab.lang.makeValidName(base_raw);   % 구조체 필드 안전화

    segNames = SIM_rep.Properties.RowNames;        % 'SOC90','SOC50','SOC30'
    nSOC     = numel(segNames);

    para_hats = zeros(nSOC, numel(para0)+2);   % + exitflag iter
    RMSE_list = zeros(nSOC,1);

    figure('Name',[base_raw ' – 2RC fitting'],'NumberTitle','off','Position',[100 100 1600 800]);
    titleStr = strrep(base_raw,'_','\_'); 

    for s = 1:nSOC
        t_vec   = SIM_rep.time   {s};
        I_vec   = SIM_rep.current{s};
        V_true  = SIM_rep.voltage{s};
        OCV_vec = SIM_rep.OCV_vec{s};

        % 2) 피팅
        problem = createOptimProblem('fmincon', ...
            'objective',@(p) RMSE_2RC(V_true,p,t_vec,I_vec,OCV_vec), ...
            'x0',para0,'lb',lb,'ub',ub, ...
            'Aineq',A_lin,'bineq',b_lin,'options',opt);

        [bestP,bestFval,eflag,~,solns] = run(ms,problem,startPts);

        iter = NaN;
        if ~isempty(solns)
            ii = find([solns.Fval]==bestFval,1);
            iter = solns(ii).Output.iterations;
        end

        para_hats(s,:) = [bestP eflag iter];
        RMSE_list(s)   = bestFval;

        % 3) 플롯 (2×4)
        V_fit = RC_model_2(bestP,t_vec,I_vec,OCV_vec);
        V_0 = RC_model_2(para0,t_vec,I_vec,OCV_vec);

        subplot(nSOC,1,s);
        plot(t_vec, V_true,'k', ...
            t_vec, V_fit ,'r', ...
            t_vec, V_0   ,'--b','LineWidth',1.2);
            xlabel('Time (s)'); ylabel('Voltage (V)');
            title(segNames{s}); grid on; legend('True','Fitted','Initial');
    end
    sgtitle([titleStr ' – 2RC fitting (SIM\_rep)'], ...
        'Interpreter','tex');

    % 4) 파일별 구조체 누적
    all_para_hats.(base_field) = para_hats;
    all_rmse.(base_field)      = RMSE_list;

    % 5) SOC별 풀 누적
    for k = 1:nSOC
        tag = segNames{k};                       % 'SOC90' …
        soc_pool.(tag)  = [soc_pool.(tag) ; para_hats(k,1:5)];
        rmse_pool.(tag) = [rmse_pool.(tag);  RMSE_list(k)];
    end

    % % 6) 파일별 저장
    % save(fullfile(folder_SIM,[base_raw '_2RCfit.mat']), ...
    %      'para_hats','RMSE_list');

    fprintf('[done] %s → 평균 RMSE = %.4f mV (nSOC = %d)\n', ...
            base_raw, mean(RMSE_list)*1e3, nSOC);
end

% ── 7) SOC별 전역 통계 ----------------------------------------------
soc_names = fieldnames(soc_pool);
for i = 1:numel(soc_names)
    sc = soc_names{i};
    P  = soc_pool.(sc);          % [N×5]
    R  = rmse_pool.(sc);         % [N×1]
    if isempty(P), continue, end

    mean_p = mean(P,1); std_p = std(P,0,1); min_p = min(P,[],1); max_p = max(P,[],1);
    mean_r = mean(R);   std_r = std(R);     min_r = min(R);      max_r = max(R);

    fprintf('\n>> [%s] 파라미터 요약 (%d 파일)\n', sc, size(P,1));
    fprintf('   [Mean] R0=%.6f R1=%.6f R2=%.6f tau1=%.3f tau2=%.3f RMSE=%.6f\n', mean_p, mean_r);
    fprintf('   [Min ] R0=%.6f R1=%.6f R2=%.6f tau1=%.3f tau2=%.3f RMSE=%.6f\n', min_p,  min_r);
    fprintf('   [Max ] R0=%.6f R1=%.6f R2=%.6f tau1=%.3f tau2=%.3f RMSE=%.6f\n', max_p,  max_r);
    fprintf('   [Std ] R0=%.6f R1=%.6f R2=%.6f tau1=%.3f tau2=%.3f RMSE=%.6f\n', std_p,  std_r);
end


% 보조함수 ======================================================================
function cost = RMSE_2RC(V_true, para, t_vec, I_vec, OCV_vec)
    V_est = RC_model_2(para,t_vec,I_vec,OCV_vec);
    cost  = sqrt( mean( (V_true - V_est).^2 ) );
end