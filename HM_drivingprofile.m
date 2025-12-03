clc; clear; close all;

% ── 경로 지정 (SIM_table 저장된 폴더) ──────────────────────────────────
folder_SIM   = 'G:\공유 드라이브\BSL_Data4\HNE_agedcell_8_processed\SIM_parsed';
save_fig_dir = fullfile(folder_SIM, 'SOC90_SIM_plots_V');   % 그림 저장 폴더 (전압용 폴더로 이름만 살짝 변경)

if ~exist(save_fig_dir, 'dir')
    mkdir(save_fig_dir);
end

% ── 파일 리스트 불러오기 (*_SIM.mat) ──────────────────────────────────
files = dir(fullfile(folder_SIM, '*_SIM.mat'));

% 드라이브 이름 (US06 > UDDS > HWFET > WLTP > city1 > city2 > HW1 > HW2)
driveNames = {'US06','UDDS','HWFET','WLTP','city1','city2','HW1','HW2'};

for f = 1:numel(files)

    % 1) SIM_table 로드
    file_now = fullfile(folder_SIM, files(f).name);
    load(file_now, 'SIM_table');   % 변수 이름 그대로라고 가정

    nSIM = height(SIM_table);
    if nSIM < 8
        fprintf('[skip] %s : SIM 개수(%d) < 8\n', files(f).name, nSIM);
        continue;
    end

    % ── SOC 90 영역: 앞 8개 SIM만 사용 ───────────────────────────────
    idx_SOC90 = 1:8;

    timeCell = SIM_table.time(idx_SOC90);
    voltCell = SIM_table.voltage(idx_SOC90);   % ★ 전압 사용

    % 2) 각 SIM별 상대 시간 (0초 시작) 계산
    t_rel = cell(size(timeCell));
    for k = 1:numel(timeCell)
        t_raw     = timeCell{k};              % 보통 double(sec)일 것
        t_rel{k}  = t_raw - t_raw(1);         % 0초 기준으로 shift
    end

    % 3) 가장 긴 SIM 길이 구해서 x축 통일
    len_vec = cellfun(@(t) t(end), t_rel);
    t_max   = max(len_vec);

    % 4) 플롯 (8개 서브플롯, x축 0~t_max 통일)
    [~, base, ~] = fileparts(files(f).name);
    fig_name = sprintf('%s - SOC90 SIM VOLTAGE profiles', base);

    figure('Name', fig_name, 'Color', 'w', 'Position', [100 100 1200 800]);

    for k = 1:8
        subplot(4,2,k);
        plot(t_rel{k}, voltCell{k}, 'LineWidth', 1.0);  % ★ 전압 플롯
        xlim([0 t_max]);
        grid on;

        title(driveNames{k}, 'Interpreter', 'none');
        ylabel('Voltage [V]');                          % ★ y축 라벨 변경

        if k > 6   % 마지막 줄에만 x축 label
            xlabel('Time [s]');
        end
    end

    sgtitle(sprintf('%s - SOC 90 SIM voltage profiles', base), 'Interpreter', 'none');

    % 5) 그림 저장 (fig + png)
    savefig(fullfile(save_fig_dir, [base '_SOC90_SIM_voltage.fig']));
    exportgraphics(gcf, fullfile(save_fig_dir, [base '_SOC90_SIM_voltage.png']), 'Resolution', 300);

    fprintf('[done] %s : SOC90 8 SIM VOLTAGE plot 저장 완료\n', files(f).name);

end

fprintf('\n=== 모든 파일 처리/플로팅 완료 (Voltage) ===\n');
