% ======================================================================
%  SIM ±2 구간 기반 2-RC 피팅 + SOC(90·50·30) 전역 통계
% ----------------------------------------------------------------------
%  • 입력 : *_SIM.mat  (SIM_table , SIM_rep 포함)
%  • 각 target SOC(90·50·30)를 통과하는 SIM 1개를 찾고
%    → 그 SIM을 중심으로 앞뒤 2개(최대 5개) 묶어 피팅.
% ======================================================================
clc; clear; close all;

% ── 경로 ----------------------------------------------------------------
folder_SIM = "G:\공유 드라이브\BSL_Data4\HNE_agedcell_2025_processed\SIM_parsed";
sim_files  = dir(fullfile(folder_SIM,"*_SIM.mat"));

exclude_name = "HNE_Driving_45degC_FC_NBR_5_3_SIM.mat";
sim_files = sim_files(~strcmp({sim_files.name},exclude_name));

% ── fmincon 설정 --------------------------------------------------------
ms       = MultiStart("UseParallel",true,"Display","off");
startPts = RandomStartPointSet('NumStartPoints',20);
opt      = optimset('display','off','MaxIter',1e3,'MaxFunEvals',1e4,...
                    'TolFun',1e-14,'TolX',1e-15);


para0 = [0.003 0.0015 20];
lb    = [0 0 0.01];
ub    = para0*10;

% % RMSE 작은 케이스 (2 3 7)-------------------
% para0 = [0.003 0.0005 0.001 10 100]; 
% lb    = [0 0 0 0.01 0.01];
% ub    = para0*10;

% ── 누적 구조체 ---------------------------------------------------------
all_para_hats = struct;   all_rmse = struct;
soc_pool  = struct('SOC90',[],'SOC50',[],'SOC20',[]);
rmse_pool = struct('SOC90',[],'SOC50',[],'SOC20',[]);

target_soc = [90 50 20];

for f = [2 3 7]%1:numel(sim_files)

    % 1) 두 테이블 로드
    load(fullfile(folder_SIM,sim_files(f).name),"SIM_table");
    base_raw   = erase(sim_files(f).name,"_SIM.mat");
    base_field = matlab.lang.makeValidName(base_raw);

    % 2) 중심-3 ~ 중심+4 SIM 인덱스 모으기
    win_idx = [];
    for trg = target_soc
        hit = find(SIM_table.SOC1 >= trg & SIM_table.SOC2 <= trg , 1,'first');
        if isempty(hit),  continue, end

        % 중심-3 ~ 중심+4
        win = max(hit-3,1) : min(hit+4 , height(SIM_table));

        win_idx = [win_idx , win];
    end

    win_idx = unique(win_idx,'sorted');
    SIM_sel = SIM_table(win_idx,:);
    segNames = SIM_sel.Properties.RowNames;   
    nSeg     = numel(segNames);               


    if nSeg==0
        fprintf("[skip] %s → 대상 SOC 구간 없음\n", base_raw);
        continue
    end

    % 3) 결과 배열
    para_hats = zeros(nSeg,numel(para0)+3);
    RMSE_list = zeros(nSeg,1);

    % 4) Figure
    rows = 3; cols = 8;
    figure('Name',[base_raw ' – 1RC fitting'],'NumberTitle','off',...
           'Position',[100 100 1600 900]);
    sgtitle(strrep(base_raw,'_','\_') + " – 1RC fitting",...
            'Interpreter','tex');

    % 5) 구간별 피팅
    for s = 1:nSeg
        t_vec   = SIM_sel.time   {s};
        I_vec   = SIM_sel.current{s};
        V_true  = SIM_sel.voltage{s};
        OCV_vec = SIM_sel.OCV_vec{s};

        problem = createOptimProblem('fmincon',...
            'objective',@(p) RMSE_1RC(V_true,p,t_vec,I_vec,OCV_vec),...
            'x0',para0,'lb',lb,'ub',ub,...
            'options',opt);

        [bestP,bestFval,eflg,~,solns] = run(ms,problem,startPts);
        iter = NaN;  if ~isempty(solns)
            iter = solns(find([solns.Fval]==bestFval,1)).Output.iterations;
        end

        para_hats(s,:) = [bestP bestFval eflg iter];
        RMSE_list(s)   = bestFval;

        V_fit = RC_model_1(bestP,t_vec,I_vec,OCV_vec);
        V_0   = RC_model_1(para0 ,t_vec,I_vec,OCV_vec);

        current_temp = SIM_sel.current(s);

        subplot(rows,cols,s);
        plot(t_vec,V_true,'k',t_vec,V_fit,'r',t_vec,V_0,'--b','LineWidth',1.2);
        xlabel('Time (s)'); ylabel('Voltage (V)');
        title(SIM_sel.Properties.RowNames{s});
        grid on; legend('True','Fitted','Initial','Location','best');
        xlim([t_vec(1) , min(t_vec(1)+200 , t_vec(end))]);


        % subplot(rows,cols,s);
        % yyaxis left
        % plot(t_vec,V_true,'k',t_vec,V_fit,'r-','LineWidth',1.2);
        % xlabel('Time (s)'); ylabel('Voltage (V)');
        % ax = gca;                     
        % ax.YAxis(1).Color = 'r';
        % 
        % yyaxis right
        % plot(t_vec,current_temp{1,1},'b-','LineWidth',0.7)
        % ylabel('current (A)')
        % ax.YAxis(2).Color = 'b'; 
        % title(SIM_sel.Properties.RowNames{s});
        % grid on; legend('True','Fitted','Current','Location','northeast');
        
    end

    set(gcf,"Position",[1600 300 1200 800]);
    
    % 6) 파일별 구조체 누적
    all_para_hats.(base_field) = para_hats;
    all_rmse.(base_field)      = RMSE_list;

    % 7) SOC 풀 누적 (평균 SOC 기준)
    for k = 1:nSeg
        midSOC = mean([SIM_sel.SOC1(k) SIM_sel.SOC2(k)]);
        if     midSOC >= 80, tag = "SOC90";
        elseif midSOC >= 40, tag = "SOC50";
        else                 tag = "SOC20"; end
        soc_pool.(tag)  = [soc_pool.(tag);  para_hats(k,1:numel(para0))];
        rmse_pool.(tag) = [rmse_pool.(tag); RMSE_list(k)];
    end

    fprintf('[done] %s → 평균 RMSE = %.4f mV (seg %d)\n', ...
             base_raw, mean(RMSE_list)*1e3, nSeg);
end

% % 8) SOC별 전역 통계 ----------------------------------------------------
% soc_names = fieldnames(soc_pool);
% for i = 1:numel(soc_names)
%     sc = soc_names{i};  P = soc_pool.(sc);  R = rmse_pool.(sc);
%     if isempty(P), continue, end
%     fprintf('\n>> [%s] 파라미터 요약 (%d seg)\n',sc,size(P,1));
%     fprintf('   Mean: R0=%.6f R1=%.6f tau1=%.3f RMSE=%.6f\n', mean(P,1), mean(R));
%     fprintf('   Min : R0=%.6f R1=%.6f tau1=%.3f RMSE=%.6f\n', min(P,[],1), min(R));
%     fprintf('   Max : R0=%.6f R1=%.6f tau1=%.3f RMSE=%.6f\n', max(P,[],1), max(R));
%     fprintf('   Std : R0=%.6f R1=%.6f tau1=%.3f RMSE=%.6f\n', std(P,0,1),  std(R));
% end
fprintf("\n전체 피팅 완료!\n");

% ── 보조 함수 -----------------------------------------------------------
function cost = RMSE_1RC(V_true, para, t, I, OCV)
    V_est = RC_model_1(para, t, I, OCV);
    cost  = sqrt(mean((V_true - V_est).^2));
end
