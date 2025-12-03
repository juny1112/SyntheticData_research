% ======================================================================
%  (개정) 전체 SIM 기반 2-RC 피팅 + SOC(90·70·50·30) 전역 통계
%  • 입력: *_SIM.mat (SIM_table 필요, 총 32 seg = 8×4 가정)
%  • 그룹핑: 기본=앞에서부터 8개씩 [90,70,50,30], 보조=SOC_center 최근접
%  • 출력: 파일별 요약(12×5: 각 SOC의 Mean/Min/Max), 비교 플롯
% ======================================================================
clc; clear; close all;

% ── 경로 & 파일 리스트 ───────────────────────────────────────────────
% folder_SIM = 'G:\공유 드라이브\BSL_Data4\HNE_SOC_moving_cutoff_5_processed\SIM_parsed';
folder_SIM = 'G:\공유 드라이브\BSL_Data4\HNE_SOC_moving_cutoff_5_processed\SIM_parsed\이름 정렬';
sim_files  = dir(fullfile(folder_SIM,"*_SIM.mat"));
if isempty(sim_files)
    error("SIM 파일을 찾지 못했습니다: %s", folder_SIM);
end

% save_path = fullfile(folder_SIM,'1RC_fitting');
% if ~exist(save_path,'dir'); mkdir(save_path); end

% 저장 경로는 상위(SIM_parsed) 폴더로 고정
save_root = fileparts(folder_SIM);          % -> ...\SIM_parsed
save_path = fullfile(save_root,'1RC_fitting');
if ~exist(save_path,'dir'); mkdir(save_path); end

% ──사용자 입력: 색상 매핑용 '용량/ SOH' (셀 순서와 동일하게 입력) ────────
% 예) QC/40 값이나 SOH(%) 벡터를 입력
Cap_user  = [58.94, 47.97, 52.15, 51.50, 57.29, 53.39];   
Cap_label = 'Capacity (QC/40, Ah)';   % 컬러바 라벨

% ── fmincon + MultiStart 설정 ────────────────────────────────────────
ms       = MultiStart("UseParallel",true,"Display","off");
startPts = RandomStartPointSet('NumStartPoints',20);
opt      = optimset('display','off','MaxIter',1e3,'MaxFunEvals',1e4, ...
                    'TolFun',1e-14,'TolX',1e-15);

% 1-RC 초기추정값 / 경계 / 선형제약 (τ1<τ2)
para0 = [0.003 0.0015 20];
lb    = [0 0 0.01];
% ub    = para0*10;
ub    = [0.05 0.03 300];

% ── 누적 컨테이너 ────────────────────────────────────────────────────
all_para_hats = struct;   % 각 파일: [nSeg × 6] = [R0 R1 tau1 | RMSE exitflag iter]
all_rmse      = struct;   % 각 파일: [nSeg × 1] RMSE
all_summary   = struct;   % 각 파일: 12×4 요약 테이블 (90/70/50/30 × Mean/Min/Max, cols=R0 R1 tau1 RMSE)

% 대표 SOC(정리/플롯 기준)
soc_targets  = [90 70 50 30];
soc_labels   = ["SOC90","SOC70","SOC50","SOC30"];
rowNames     = { ...
  'SOC90_Mean','SOC90_Min','SOC90_Max', ...
  'SOC70_Mean','SOC70_Min','SOC70_Max', ...
  'SOC50_Mean','SOC50_Min','SOC50_Max', ...
  'SOC30_Mean','SOC30_Min','SOC30_Max'};

% ── 메인 루프 (모든 파일 처리) ───────────────────────────────────────
for f = 1:numel(sim_files)
    % 1) load SIM_table
    S = load(fullfile(folder_SIM,sim_files(f).name),"SIM_table");
    if ~isfield(S,"SIM_table")
        warning("SIM_table 없음: %s (건너뜀)", sim_files(f).name);
        continue
    end
    SIM_table = S.SIM_table;

    base_raw   = erase(sim_files(f).name,"_SIM.mat");
    base_field = matlab.lang.makeValidName(base_raw);

    % 2) 이번 실험: 각 파일에 SIM 총 32개(=8×4) 가정
    nSeg = height(SIM_table);
    if nSeg==0
        warning("No SIM rows: %s", base_raw);
        continue
    end

    % ── 그룹 코드 할당 ────────────────────────────────────────────────
    % 우선순위 1) 8개씩 블록 매핑: [1..8]→90, [9..16]→70, [17..24]→50, [25..32]→30
    grp_code = zeros(nSeg,1);    % 1:90, 2:70, 3:50, 4:30
    if nSeg >= 32
        blk = [1 8; 9 16; 17 24; 25 32];
        for g = 1:4
            ii = blk(g,1):min(blk(g,2), nSeg);
            grp_code(ii) = g;
        end
    end

    % 우선순위 2) 예외(32개가 아니거나 순서 불명확) → SOC_center 최근접 매핑
    if any(grp_code==0)
        SOC_center = mean([SIM_table.SOC1 SIM_table.SOC2], 2, 'omitnan');
        miss = isnan(SOC_center);
        if any(miss) && ismember('SOC_vec', SIM_table.Properties.VariableNames)
            try
                SOC_center(miss) = cellfun(@(v) mean(v,'omitnan'), SIM_table.SOC_vec(miss));
            catch
                % 남은 NaN은 그대로 두고 제외
            end
        end
        valid = ~isnan(SOC_center);
        if any(valid)
            [~, gmin] = min(abs(SOC_center(valid) - soc_targets), [], 2);
            vidx = find(valid);
            grp_code(vidx) = gmin;
        end
        % 여전히 0이면(정보 부족) 해당 세그먼트는 후속 통계에서 자동 제외됨
    end

    % 3) 전 세그먼트 피팅
    para_hats = nan(nSeg, 3+3);
    RMSE_list = nan(nSeg, 1);

    % 서브플롯 레이아웃(최대 8열 고정, 행은 자동)
    cols = 8;
    rows = max(1, ceil(nSeg/cols));
    fig = figure('Name',[base_raw ' – 1RC fitting'], 'NumberTitle','off', ...
        'Position',[100 100 1600 900], 'Color','w');
    try
        sgtitle(strrep(base_raw,'_','\_') + " – 1RC fitting", 'Interpreter','tex');
    catch
        % (older MATLAB)
        suptitle(strrep(base_raw,'_','\_') + " – 1RC fitting");
    end

    for s = 1:nSeg
        try
            t = SIM_table.time{s};        % duration
            I = SIM_table.current{s};
            V = SIM_table.voltage{s};
            O = []; 
            if ismember('OCV_vec', SIM_table.Properties.VariableNames)
                O = SIM_table.OCV_vec{s};
            end
            
            problem = createOptimProblem('fmincon', ...
                'objective',@(p)RMSE_1RC(V,p,t,I,O), ...
                'x0',para0,'lb',lb,'ub',ub,'options',opt);

            [Pbest, Fval, exitflg, ~, sol] = run(ms,problem,startPts);
            it = NaN;
            if ~isempty(sol)
                it = sol(find([sol.Fval]==Fval,1)).Output.iterations;
            end

            para_hats(s,:) = [Pbest, Fval, exitflg, it];
            RMSE_list(s)   = Fval;

            % ---- SOC 라벨 생성 ----
            if grp_code(s) >= 1 && grp_code(s) <= numel(soc_targets)
                soc_txt = sprintf('SOC %d', soc_targets(grp_code(s)));
            else
                soc_txt = 'SOC ?';
            end

            % ---- 모델 전압 ----
            V_fit = RC_model_1(Pbest, t, I, O);
            V_ini = RC_model_1(para0 , t, I, O);

            % ---- 서브플롯 ----
            subplot(rows, cols, s);
            plot(t, V, 'k', t, V_fit, 'r', t, V_ini, '--b', 'LineWidth', 1.1);
            grid on;
            xlabel('Time'); ylabel('Voltage (V)');
            ttl = sprintf('Load %d | %s | RMSE=%.2f mV', s, soc_txt, Fval*1e3);
            title(ttl, 'Interpreter','none');
            legend('True','Fitted','Initial','Location','northeast','Box','off');
      

        catch ME
            warning("(%s) seg %d 피팅 실패: %s", base_raw, s, ME.message);
        end

    end
    % === (A) 파일별 피팅 figure 저장 ===
    if isgraphics(fig,'figure')
        savefig(fig, fullfile(save_path, [base_raw '_1RC_fit.fig']));
    else
        warning('(%s) figure 핸들이 유효하지 않아 저장을 건너뜁니다.', base_raw);
    end
    % 필요하면 창 닫기:
    % close(fig);
    

    % 4) SOC(90/70/50/30)별 요약 테이블(12×4) 구성
    T = table( ...
        nan(12,1), nan(12,1), nan(12,1), nan(12,1), ...
        'VariableNames', {'R0','R1','tau1','RMSE'}, ...
        'RowNames',     rowNames );

    P_all = para_hats(:,1:3);

    % 그룹 마스크
    m90 = (grp_code==1);
    m70 = (grp_code==2);
    m50 = (grp_code==3);
    m30 = (grp_code==4);
    groups = {m90,m70,m50,m30};

    r = 1;
    for g = 1:4
        idx = groups{g};
        if any(idx)
            blockP = P_all(idx,:);            % [*, 3] = R0 R1 tau1
            blockE = RMSE_list(idx);          % [*, 1] = RMSE
            block4 = [blockP, blockE];        % [*, 4] = R0 R1 tau1 RMSE

            T{r  ,:} = mean(block4,1,'omitnan');  % Mean
            T{r+1,:} = min (block4,[],1);         % Min
            T{r+2,:} = max (block4,[],1);         % Max
        end
        r = r + 3;
    end

    % 5) 누적 저장
    all_para_hats.(base_field) = para_hats;
    all_rmse.(base_field)      = RMSE_list;
    all_summary.(base_field)   = T;

    % 로그
    fprintf('[done] %s → fitted %d segs, summary(12×4) 저장  |  counts: 90=%d,70=%d,50=%d,30=%d\n', ...
        base_raw, nSeg, nnz(m90), nnz(m70), nnz(m50), nnz(m30));
    
end

fprintf("모든 파일 처리 완료!\n");

save(fullfile(save_path,'1RC_results.mat'), ...
     'all_para_hats','all_summary','-v7.3');
fprintf('1RC 결과 저장 완료: %s\n', fullfile(save_path,'1RC_results.mat'));

% ─────────────────────────────────────────────────────────────────────
% (NEW) 모든 셀 한 장: nCells(행) × 4(SOC) 그리드로 '3번째 Load(200s)'만 표시
% ─────────────────────────────────────────────────────────────────────
% 셀 키를 원래 파일 순서대로 정렬
cells_in_results = fieldnames(all_para_hats);
keys = strings(0);
for f = 1:numel(sim_files)
    base_raw   = erase(sim_files(f).name,"_SIM.mat");
    base_field = matlab.lang.makeValidName(base_raw);
    if ismember(base_field, cells_in_results)
        keys(end+1) = string(base_field); %#ok<AGROW>
    end
end
cells  = cellstr(keys);
nCells = numel(cells);

if nCells==0
    warning('그릴 셀이 없습니다. (all_para_hats 비어있음)');
else
    % 파일 경로 매핑
    all_paths = struct;
    for f = 1:numel(sim_files)
        base_raw   = erase(sim_files(f).name,"_SIM.mat");
        base_field = matlab.lang.makeValidName(base_raw);
        if ismember(base_field, cells)
            all_paths.(base_field) = fullfile(folder_SIM, sim_files(f).name);
        end
    end

    nRows = nCells; nCols = 4;  % 행=셀 개수, 열=SOC 4개
    figAll = figure('Name','ALL CELLS – SOC별 3rd Load (200s)', ...
                    'NumberTitle','off', ...
                    'Position',[50 50 1600 240*nRows], 'Color','w');
    tlAll = tiledlayout(nRows, nCols, 'TileSpacing','compact', 'Padding','compact');

    trueColor = [0.10 0.10 0.10];   % True(두껍게)
    fitColor  = [0.85 0.25 0.15];   % Fitted(점선)

    for r = 1:nRows
        key = cells{r};

        % 데이터 로드
        if ~isfield(all_paths, key), for gg=1:nCols, nexttile; axis off; text(0.5,0.5,[key ' 경로 없음'],'HorizontalAlignment','center'); end; continue; end
        S2 = load(all_paths.(key), "SIM_table");
        if ~isfield(S2,"SIM_table"), for gg=1:nCols, nexttile; axis off; text(0.5,0.5,[key ' SIM_table 없음'],'HorizontalAlignment','center'); end; continue; end
        SIM_table = S2.SIM_table;

        % 이 셀의 결과
        P_hat = all_para_hats.(key);
        RMSEv = all_rmse.(key);

        % grp_code 재구성(루프와 동일)
        nSeg = height(SIM_table);
        grp_code = zeros(nSeg,1);        % 1:90, 2:70, 3:50, 4:30
        if nSeg >= 32
            blk = [1 8; 9 16; 17 24; 25 32];
            for g = 1:4
                ii = blk(g,1):min(blk(g,2), nSeg);
                grp_code(ii) = g;
            end
        end
        if any(grp_code==0)
            SOC_center = mean([SIM_table.SOC1 SIM_table.SOC2], 2, 'omitnan');
            miss = isnan(SOC_center);
            if any(miss) && ismember('SOC_vec', SIM_table.Properties.VariableNames)
                try, SOC_center(miss) = cellfun(@(v) mean(v,'omitnan'), SIM_table.SOC_vec(miss)); catch, end
            end
            valid = ~isnan(SOC_center);
            if any(valid)
                [~, gmin] = min(abs(SOC_center(valid) - soc_targets), [], 2);
                vidx = find(valid);
                grp_code(vidx) = gmin;
            end
        end

        % 열=SOC 4개
        for gg = 1:nCols     % 1:90, 2:70, 3:50, 4:30
            tileIdx = (r-1)*nCols + gg;
            nexttile(tileIdx); hold on; grid on;

            idx_g = find(grp_code==gg);
            if numel(idx_g) >= 3
                sSel = idx_g(3);                  % '3번째' 세그먼트
                tSel = SIM_table.time{sSel};
                ISel = SIM_table.current{sSel};
                VSel = SIM_table.voltage{sSel};
                OSel = [];
                if ismember('OCV_vec', SIM_table.Properties.VariableNames)
                    OSel = SIM_table.OCV_vec{sSel};
                end

                Psel    = P_hat(sSel,1:3);
                V_fitS  = RC_model_1(Psel, tSel, ISel, OSel);
                rmse_mV = RMSEv(sSel)*1e3;

                % 플롯: True(두껍게), Fitted(점선), xlim=200s
                plot(tSel, VSel,  'LineWidth', 2.2, 'Color', trueColor);
                plot(tSel, V_fitS,'LineWidth', 1.4, 'LineStyle','--', 'Color', fitColor);
                if isduration(tSel), xlim([tSel(1), tSel(1) + seconds(200)]);
                else,                xlim([min(tSel), min(tSel) + 200]); end

                % 타이틀(요청 포맷 유지)
                soc_txt = sprintf('SOC %d', soc_targets(gg));
                title(sprintf('Load %d | %s | RMSE=%.2f mV', sSel, soc_txt, rmse_mV), 'Interpreter','none');

                % 보기 좋게 여백
                ymin = min([VSel; V_fitS]); ymax = max([VSel; V_fitS]);
                pad  = 0.02 * max(ymax - ymin, eps); ylim([ymin - pad, ymax + pad]);
            else
                axis off;
                soc_txt = sprintf('SOC %d', soc_targets(gg));
                text(0.5,0.5, sprintf('%s: 3번째 Load 없음', soc_txt), ...
                    'HorizontalAlignment','center','FontSize',11);
            end

            % 라벨: 마지막 행만 x라벨, 첫 열만 y라벨
            if r==nRows, xlabel('Time'); end
            if gg==1,    ylabel('Voltage (V)'); end
        end
    end

    % 전체 제목
    title(tlAll, 'ALL CELLS – SOC별 3rd Load(200s) Overview', 'Interpreter','none');

    % 저장
    savefig(figAll, fullfile(save_path, 'ALLCELLS_SOC_3rdLoad_grid.fig'));
    exportgraphics(figAll, fullfile(save_path, 'ALLCELLS_SOC_3rdLoad_grid.png'), 'Resolution', 200);
end


%% ——— SOC별 파라미터 비교 플롯 (Capacity 색상: 파스텔 빨강→파랑) ————————————
cells  = fieldnames(all_summary);
SOCx   = [30 50 70 90];                 % x축 순서(오름차순)
pNames = {'R0','R1','tau1'};            % 1RC 파라미터

% (필수) 용량 벡터는 셀 순서대로 미리 Cap_user 로 정의해 두세요.
% 예시: Cap_user = [58.94, 47.97, 52.15, 51.50, 57.29, 53.39];
if ~exist('Cap_user','var')
    warning('Cap_user 변수가 정의되어 있지 않습니다. 색상 매핑이 정상적이지 않을 수 있습니다.');
    Cap_user = nan(numel(cells),1);
end
if ~exist('Cap_label','var')
    Cap_label = 'Capacity (QC/40, Ah)';
end

% ---- 용량 검증 & 정규화 ----
if isempty(Cap_user) || numel(Cap_user) < numel(cells)
    warning('Cap_user 길이가 cells 수(%d)와 다릅니다. 앞에서부터 맞는 것만 사용합니다.', numel(cells));
end
capVec = nan(numel(cells),1);
capVec(1:min(numel(Cap_user),numel(cells))) = Cap_user(1:min(numel(Cap_user),numel(cells)));
capMin = min(capVec,[],'omitnan');  
capMax = max(capVec,[],'omitnan');
if ~isfinite(capMin) || ~isfinite(capMax) || capMax<=capMin
    capMin = 0; capMax = 1;  % 안전가드
end

% ---- 파스텔 diverging 컬러맵(채도 유지): 빨강(작음) → 라벤더(중간) → 파랑(큼) ----
Nmap = 256;
anchors = [0.88 0.16 0.24;   % red (low)
           0.83 0.70 0.86;   % lavender (mid)
           0.16 0.38 0.92];  % blue (high)
x  = [0 0.5 1];
xi = linspace(0,1,Nmap)';
cmap = [interp1(x, anchors(:,1), xi, 'pchip'), ...
        interp1(x, anchors(:,2), xi, 'pchip'), ...
        interp1(x, anchors(:,3), xi, 'pchip')];
cmap = min(max(cmap,0),1);
% HSV에서 최소 채도/명도 보정(중간이 과하게 연해지는 것 방지)
hsv = rgb2hsv(cmap);
hsv(:,2) = max(0.35, hsv(:,2));      % 최소 채도 보장
hsv(:,2) = min(1.0, hsv(:,2)*1.20);  % 채도 소폭 증가
hsv(:,3) = max(0.75, hsv(:,3)*0.95); % 너무 밝음 방지
cmap = hsv2rgb(hsv);

% 값 v를 [capMin,capMax] → cmap 인덱스로 매핑
mapColor = @(v) cmap( max(1, min(Nmap, 1 + round((v-capMin)/max(capMax-capMin,eps)*(Nmap-1))) ), : );

% 사용자 맞춤 범례 라벨(셀 배열). 없으면 파일명 사용
customLabels = {'1_신품셀_58.94Ah', '2_1C,150cyc_47.97Ah', '3_급속/US06_52.15Ah', ...
    '4_병렬_51.50Ah', '5_2C,10cyc_57.29Ah', '7_완속급속/US06_53.39Ah'}; 

for p = 1:numel(pNames)
    figure('Name',pNames{p}+" vs SOC (1-RC, colored by "+Cap_label+")", ...
           'NumberTitle','off','Color','w'); hold on;

    Y_all = [];
    for c = 1:numel(cells)
        T = all_summary.(cells{c});
        if isempty(T) || height(T)~=12, continue; end

        y = [ valOrNaN(T,"SOC30_Mean", pNames{p}), ...
              valOrNaN(T,"SOC50_Mean", pNames{p}), ...
              valOrNaN(T,"SOC70_Mean", pNames{p}), ...
              valOrNaN(T,"SOC90_Mean", pNames{p}) ];
        if all(isnan(y)), continue; end
        Y_all = [Y_all; y]; %#ok<AGROW>

        col = mapColor(capVec(c));   % 이 셀의 색상

        % 범례 라벨
        if exist('customLabels','var') && ~isempty(customLabels) && numel(customLabels)>=c
            dname = strrep(customLabels{c}, '_', '\_');   % '_' 이스케이프
        else
            dname = strrep(cells{c}, '_', '\_');
        end

        plot(SOCx, y, '-o', ...
             'LineWidth', 1.8, ...
             'Color', col, ...
             'MarkerFaceColor', col, ...
             'MarkerEdgeColor', col, ...
             'DisplayName', dname);
    end

    xlabel('SOC (%)'); ylabel(pNames{p});
    title(pNames{p}+" vs SOC");
    grid on;

    % y축 범위 자동
    if ~isempty(Y_all)
        ymax = max(Y_all(:), [], 'omitnan');  
        ymin = min(Y_all(:), [], 'omitnan');
        if isfinite(ymax)
            if ~isfinite(ymin) || ymin >= 0, ylim([0, ymax*1.05]);
            else,                            ylim([ymin*0.95, ymax*1.05]);
            end
        end
    end

    % ---- 컬러바(용량 범위) ----
    colormap(cmap);
    cb = colorbar('Location','eastoutside');
    cb.Label.String = Cap_label;
    if exist('clim','builtin') || exist('clim','file')
        clim([capMin capMax]);
    else
        axg = gca; if isprop(axg,'CLim'), axg.CLim = [capMin capMax]; end
    end
    cb.Limits = [capMin capMax];

    legend('Location','best');
    savefig(gcf, fullfile(save_path, [pNames{p} '_vs_SOC_colored.fig']));
    exportgraphics(gcf, fullfile(save_path, [pNames{p} '_vs_SOC_colored.png']), 'Resolution', 200);
end

% ── 보조 함수 ─────────────────────────────────────────────────────────
function cost = RMSE_1RC(V_true, para, t, I, OCV)
    V_est = RC_model_1(para,t,I,OCV);    % 사용자 정의 함수가 경로에 있어야 함
    cost  = sqrt(mean((V_true - V_est).^2));
end

function y = valOrNaN(T, rowName, colName)
    if ismember(rowName, T.Properties.RowNames)
        y = T{rowName, colName};
        if isempty(y), y = NaN; end
    else
        y = NaN;
    end
end
