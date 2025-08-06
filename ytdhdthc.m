% ======================================================================
% 파싱된 SIM 파일에서 17번째부터 8개 구간별 voltage vs time 플롯
% ======================================================================
clc; clear; close all;

% 1) 파싱된 SIM 파일들이 들어있는 폴더 (필요에 따라 수정)
sim_folder = "G:\공유 드라이브\BSL_Data4\HNE_agedcell_2025_processed\SIM_parsed";

% 2) *_SIM.mat 파일 리스트 가져오기
sim_files = dir(fullfile(sim_folder, "*_SIM.mat"));

% 3) 각 파일마다
for f = 2%1:numel(sim_files)
    % 3-1) SIM_table만 로드
    data      = load(fullfile(sim_folder, sim_files(f).name), "SIM_table");
    SIM_table = data.SIM_table;
    nSIM      = height(SIM_table);
    
    % 3-2) 17번째부터 8개 인덱스 계산
    startIdx = 17;
    endIdx   = startIdx + 8 - 1;
    selIdx   = startIdx : min(endIdx, nSIM);
    if isempty(selIdx)
        warning("'%s'에 17번째 SIM이 없습니다. 넘어갑니다.", sim_files(f).name);
        continue
    end
    
    % 3-3) 선택된 각 SIM 세그먼트를 별도 Figure에 그리기
    for k = selIdx
        t = SIM_table.time{k};       % 시간 벡터
        v = SIM_table.voltage{k};    % 전압 벡터
        
        % ← 여기서 Figure 크기를 가로로 길게 설정
        figure('Name', sprintf("%s (SIM %d)", sim_files(f).name, k), ...
            'NumberTitle','off', ...
            'Units','pixels', ...        % 픽셀 단위
            'Position',[100, 100, 1000, 400]);  % [left, bottom, width, height]

        plot(t, v, 'LineWidth', 1.5);
        xlabel('Time [s]');
        ylabel('Voltage [V]');
        title(sprintf("%s — SIM segment %d", sim_files(f).name, k), ...
              'Interpreter','none');
        grid on;
    end
end
